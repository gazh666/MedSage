import json
import logging
import mimetypes
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SymptomAnalysisRequest,
    SymptomAnalysisResponse
)
from app.services.llm.gemini_client import gemini_client
from app.services.llm.prompts import (
    MEDICAL_SYSTEM_PROMPT,
    SYMPTOM_ANALYSIS_PROMPT,
    TRIAGE_PROMPT
)
from app.core.config import settings
from app.core.audit_confidence import derive_confidence_band
from app.db.session import get_db
from app.db.models import ChatHistory, AIDecisionAudit
from app.services.agents.nodes.emergency_action import emergency_action_node
from app.services.agents.state import AgentState
from app.core.deps import CurrentUser

router = APIRouter()
logger = logging.getLogger(__name__)
UNAUTHORIZED_RESPONSE = {
    401: {
        "description": "Not authenticated. Provide a valid Bearer access token.",
    }
}

MAX_CONTEXT_MESSAGES = 20


async def _merge_server_history(
    db: AsyncSession,
    session_id: Optional[str],
    user_id: int,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """服务端上下文兜底：前端未携带完整历史时，从 DB 恢复最近会话记录。

    仅在 messages 极短（历史丢失 / API 直连）时用服务端历史补全，避免与前端
    已传来的上下文重复；最后做窗口截断，防止上下文无限增长。
    """
    if not session_id:
        return messages
    try:
        result = await db.execute(
            select(ChatHistory)
            .where(ChatHistory.session_id == session_id, ChatHistory.user_id == user_id)
            .order_by(ChatHistory.created_at.desc())
            .limit(MAX_CONTEXT_MESSAGES)
        )
        rows = result.scalars().all()
        if not rows:
            return messages

        history = [{"role": r.role, "content": r.content} for r in reversed(rows)]
        merged = (history + messages) if len(messages) <= 1 else messages
        return merged[-MAX_CONTEXT_MESSAGES:]
    except Exception:
        return messages

RED_FLAG_KEYWORDS = [
    "chest pain",
    "difficulty breathing",
    "shortness of breath",
    "severe bleeding",
    "unconscious",
    "stroke",
    "seizure",
]


def _detect_red_flags(text: str) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in RED_FLAG_KEYWORDS if keyword in lowered]


def _infer_image_type(file: UploadFile) -> str:
    content_type = file.content_type or ""
    if "png" in content_type or "jpeg" in content_type or "jpg" in content_type or "webp" in content_type:
        return "medical_image"

    guessed = mimetypes.guess_type(file.filename or "")[0] or ""
    if guessed.startswith("image/"):
        return "medical_image"
    return "medical_image"


def _parse_json_or_none(value: Optional[str]) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _is_upload_file(value: Any) -> bool:
    return isinstance(value, (UploadFile, StarletteUploadFile))


async def _parse_chat_request(
    raw_request: Request,
) -> tuple[ChatRequest, Optional[UploadFile], Optional[UploadFile]]:
    content_type = raw_request.headers.get("content-type", "").lower()

    if "multipart/form-data" not in content_type:
        payload = await raw_request.json()
        return ChatRequest.model_validate(payload), None, None

    form = await raw_request.form()
    messages_raw = form.get("messages")
    if isinstance(messages_raw, str):
        parsed_messages = _parse_json_or_none(messages_raw) or []
    else:
        parsed_messages = []

    user_location_raw = form.get("user_location")
    user_location = _parse_json_or_none(user_location_raw) if isinstance(user_location_raw, str) else None

    payload = {
        "messages": parsed_messages,
        "session_id": form.get("session_id"),
        "system_prompt": form.get("system_prompt"),
        "user_location": user_location,
    }

    parsed_request = ChatRequest.model_validate(payload)
    image_file = form.get("image")
    audio_file = form.get("audio")
    return (
        parsed_request,
        image_file if _is_upload_file(image_file) else None,
        audio_file if _is_upload_file(audio_file) else None,
    )


@router.post("/chat", response_model=ChatResponse, responses=UNAUTHORIZED_RESPONSE)
async def chat(
    raw_request: Request,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
   
    try:
        request, image_file, audio_file = await _parse_chat_request(raw_request)

       
        session_id = request.session_id or str(uuid.uuid4())
        
     
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        messages = await _merge_server_history(db, session_id, current_user.id, messages)
        
      
        if not messages:
            raise HTTPException(status_code=422, detail="At least one message is required.")

        latest_user_text = messages[-1]["content"] if messages else ""
        attachment_notes: list[str] = []
        attachment_metadata: dict[str, Any] = {}

        if audio_file is not None:
            from app.services.voice.audio_utils import audio_utils
            from app.services.voice.service import voice_service

            audio_data = await audio_file.read()
            is_valid_audio, audio_error = audio_utils.validate_audio(audio_data)
            if not is_valid_audio:
                raise HTTPException(status_code=400, detail=audio_error or "Invalid audio file.")

            try:
                audio_transcription = await voice_service.transcribe_audio(
                    audio_data,
                    filename=audio_file.filename,
                    content_type=audio_file.content_type,
                )
                if audio_transcription:
                    attachment_notes.append(f"[Audio transcription]\n{audio_transcription}")
                    attachment_metadata["audio"] = {
                        "filename": audio_file.filename,
                        "transcription": audio_transcription,
                    }
            except Exception as transcription_error:
                logger.warning(
                    "Audio transcription failed for chat endpoint: %s",
                    transcription_error,
                )
                attachment_metadata["audio"] = {
                    "filename": audio_file.filename,
                    "transcription_error": str(transcription_error),
                }

        if image_file is not None:
            from app.services.vision.gemini_vision import gemini_vision
            from app.services.vision.image_processor import image_processor

            image_data = await image_file.read()
            is_valid_image, image_error = image_processor.validate_image(image_data)
            if not is_valid_image:
                raise HTTPException(status_code=400, detail=image_error or "Invalid image file.")

            image_type = _infer_image_type(image_file)
            image_analysis = await gemini_vision.analyze_medical_image(
                image_data=image_data,
                image_type=image_type,
                filename=image_file.filename,
                additional_context=latest_user_text or None,
            )
            image_summary = image_analysis.get("findings_summary") or image_analysis.get("full_analysis") or ""
            if image_summary:
                attachment_notes.append(f"[Image analysis summary]\n{image_summary}")
            attachment_metadata["image"] = {
                "filename": image_file.filename,
                "image_type": image_type,
                "severity": image_analysis.get("severity"),
                "confidence": image_analysis.get("confidence"),
            }

        if attachment_notes:
            enriched_text = latest_user_text.strip()
            notes_text = "\n\n".join(attachment_notes).strip()
            messages[-1]["content"] = f"{enriched_text}\n\n{notes_text}".strip() if enriched_text else notes_text
            latest_user_text = messages[-1]["content"]

        red_flags = _detect_red_flags(latest_user_text)
        metadata = None

        if red_flags:
            if request.user_location:
                user_lat = request.user_location.lat
                user_lng = request.user_location.lng
                emergency_state: AgentState = {
                    "patient_id": None,
                    "symptoms": latest_user_text,
                    "user_location": {"lat": user_lat, "lng": user_lng},
                    "age": None,
                    "gender": None,
                    "medical_history": None,
                    "vitals": None,
                    "has_image": False,
                    "image_type": None,
                    "image_analysis": None,
                    "triage_result": None,
                    "symptom_analysis": {"red_flags": red_flags},
                    "rag_context": None,
                    "diagnosis": None,
                    "treatment_plan": None,
                    "is_emergency": True,
                    "emergency_info": {
                        "status": "CRITICAL",
                        "red_flags": red_flags,
                    },
                    "urgency_level": "EMERGENCY",
                    "next_step": "emergency_action",
                    "messages": [],
                    "final_report": None,
                    "confidence": 0.0,
                }
                emergency_result = await emergency_action_node(emergency_state)
                metadata = emergency_result.get("emergency_info")
                response_text = emergency_result.get(
                    "final_report",
                    "CRITICAL: Potential life-threatening condition detected.",
                )
            else:
                metadata = {
                    "status": "CRITICAL",
                    "red_flags": red_flags,
                    "user_location": None,
                    "nearby_facilities": [],
                    "first_aid_instructions": [
                        "Keep the patient calm and seated upright.",
                        "Avoid giving food or drink while awaiting emergency help.",
                    ],
                }
                response_text = (
                    "CRITICAL: Potential life-threatening condition detected.\n"
                    "Call emergency services immediately.\n"
                    "Location access was unavailable, so nearest CHC could not be auto-resolved."
                )
        else:
            system_prompt = request.system_prompt or MEDICAL_SYSTEM_PROMPT
            response_text = await gemini_client.chat(messages, system_prompt)

        if attachment_metadata:
            metadata = {
                **(metadata or {}),
                "input_modalities": attachment_metadata,
            }
        
        if isinstance(response_text, list):
            # Extract text if response_text is a list of dicts (LangChain behavior for some models)
            response_text = "".join([item.get("text", "") for item in response_text if isinstance(item, dict)])
        elif not isinstance(response_text, str):
            response_text = str(response_text)
            
      
        user_msg = ChatHistory(
            session_id=session_id,
            user_id=current_user.id,
            role="user",
            content=messages[-1]["content"] # type: ignore
        )
        db.add(user_msg)
        
        
        assistant_msg = ChatHistory(
            session_id=session_id,
            user_id=current_user.id,
            role="assistant",
            content=response_text
        )
        db.add(assistant_msg)

        audit_record = AIDecisionAudit(
            user_id=current_user.id,
            session_id=session_id,
            source_endpoint="/api/v1/chat/chat",
            decision_type="chat",
            input_summary=messages[-1]["content"],  # type: ignore
            output_summary=response_text[:2000],
            confidence_band=derive_confidence_band(
                urgency_level="EMERGENCY" if red_flags else "ROUTINE",
                red_flags_count=len(red_flags),
                output_summary=response_text[:2000],
            ),
            model_name=settings.GEMINI_MODEL,
            model_version="v1",
            prompt_version="medical-system-v1",
        )
        db.add(audit_record)
        await db.commit()
        
        return ChatResponse(
            message=response_text,
            session_id=session_id,
            timestamp=datetime.now(),
            metadata=metadata,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/analyze-symptoms",
    response_model=SymptomAnalysisResponse,
    responses=UNAUTHORIZED_RESPONSE,
)
async def analyze_symptoms(
    request: SymptomAnalysisRequest,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze symptoms and provide preliminary assessment
    """
    try:
      
        prompt = SYMPTOM_ANALYSIS_PROMPT.format(symptoms=request.symptoms)
        
        
        analysis = await gemini_client.generate(prompt)
        
       
        triage_prompt = TRIAGE_PROMPT.format(symptoms=request.symptoms)
        triage_result = await gemini_client.generate(triage_prompt)
        
       
        severity = "ROUTINE"
        if "EMERGENCY" in triage_result.upper():
            severity = "EMERGENCY"
        elif "URGENT" in triage_result.upper():
            severity = "URGENT"
        
      
        conditions = ["Condition analysis in progress"]

        audit_record = AIDecisionAudit(
            user_id=current_user.id,
            source_endpoint="/api/v1/chat/analyze-symptoms",
            decision_type="symptom_analysis",
            input_summary=request.symptoms,
            output_summary=analysis[:2000] if isinstance(analysis, str) else str(analysis)[:2000],
            confidence_band=derive_confidence_band(
                urgency_level=severity,
                red_flags_count=1 if severity in {"EMERGENCY", "URGENT"} else 0,
                output_summary=triage_result if isinstance(triage_result, str) else str(triage_result),
            ),
            urgency_level=severity.lower(),
            model_name=settings.GEMINI_MODEL,
            model_version="v1",
            prompt_version="symptom-analysis-v1",
        )
        db.add(audit_record)
        await db.commit()
        
        return SymptomAnalysisResponse(
            analysis=analysis,
            severity=severity,
            possible_conditions=conditions,
            recommendations=triage_result
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from sqlalchemy.future import select

@router.get("/history/{session_id}", responses=UNAUTHORIZED_RESPONSE)
async def get_chat_history(
    session_id: str,
    current_user: CurrentUser,
    db: AsyncSession = Depends(get_db),
):
    
    result = await db.execute(
        select(ChatHistory)
        .filter(
            ChatHistory.session_id == session_id,
            ChatHistory.user_id == current_user.id,
        )
        .order_by(ChatHistory.created_at)
    )
    history = result.scalars().all()
    
    return {
        "session_id": session_id,
        "messages": [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.created_at
            }
            for msg in history
        ]
    }
