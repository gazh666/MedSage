from __future__ import annotations
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.session import get_db
from app.db.models import ImageAnalysis, Diagnosis, Patient, User
from app.schemas.agents import DiagnosisRequest, DiagnosisResponse
from app.services.agents.graph import get_medical_agent_graph
from app.services.agents.state import AgentState
from app.core.deps import get_current_active_user

logger = logging.getLogger(__name__)
router = APIRouter()


async def _load_patient_notes(db: AsyncSession, current_user: User, request: DiagnosisRequest) -> Optional[str]:
    """读取患者长期记忆（过敏史 / 慢病史），用于自动注入诊断上下文。"""
    patient_record = None
    if request.patient_id:
        result = await db.execute(
            select(Patient).where(
                Patient.id == request.patient_id,
                Patient.user_id == current_user.id,
            )
        )
        patient_record = result.scalar_one_or_none()

    if patient_record is None:
        latest = await db.execute(
            select(Patient)
            .where(Patient.user_id == current_user.id)
            .order_by(Patient.id.desc())
            .limit(1)
        )
        patient_record = latest.scalar_one_or_none()

    if patient_record is None:
        return None

    notes = []
    if patient_record.allergies:
        notes.append(f"Known allergies: {patient_record.allergies}")
    if patient_record.chronic_conditions:
        notes.append(f"Chronic conditions: {patient_record.chronic_conditions}")
    return "; ".join(notes) if notes else None


async def _enrich_medical_history(db: AsyncSession, current_user: User, request: DiagnosisRequest) -> str:
    """把患者档案信息并入医学史，供 diagnose / agentic-diagnose 两个端点复用。"""
    notes = await _load_patient_notes(db, current_user, request)
    if not notes:
        return request.medical_history or ""
    return f"{request.medical_history}; {notes}" if request.medical_history else notes

@router.post(
    "/diagnose", 
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    summary="Run full diagnostic agent workflow",
    description="Trigger a multi-agent pipeline: Triage -> Symptoms -> Vision -> RAG -> Diagnosis -> Plan"
)
async def multi_agent_diagnosis(
    request: DiagnosisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
  
    try:
        image_analysis_data = None
        if request.image_analysis_id:
            result = await db.execute(
                select(ImageAnalysis).filter(ImageAnalysis.id == request.image_analysis_id)
            )
            image_record = result.scalars().first()
            
            if not image_record:
                logger.warning(f"Image ID {request.image_analysis_id} not found. Proceeding without vision data.")
            else:
                image_analysis_data = image_record.analysis_result
        
        # 注入患者长期记忆（过敏史 / 慢病史）到诊断上下文
        medical_history = await _enrich_medical_history(db, current_user, request)

        initial_state: AgentState = {
            "patient_id": request.patient_id,
            "symptoms": request.symptoms,
            "user_location": request.user_location.model_dump() if request.user_location else None,
            "age": request.age,
            "gender": request.gender,
            "medical_history": medical_history,
            "vitals": request.vitals.model_dump() if request.vitals else None,
            "has_image": image_analysis_data is not None,
            "image_type": image_analysis_data.get('image_type') if image_analysis_data else None,
            "image_analysis": image_analysis_data,
            "lung_sound_path": request.lung_sound_path,
            "lung_sound_result": request.lung_sound_result,
            "urgency_level": "ROUTINE",
            "messages": [],
            "confidence": 0.0,
            "triage_result": None,
            "symptom_analysis": None,
            "rag_context": None,
            "diagnosis": None,
            "treatment_plan": None,
            "is_emergency": False,
            "emergency_info": None,
            "next_step": None,
            "final_report": None
        }
        
        logger.info(f"🚀 Starting agent workflow for patient: {request.patient_id or 'Guest'}")
        
        graph = get_medical_agent_graph()
        final_state = await graph.ainvoke(initial_state)
        
        diagnosis_info = final_state.get('diagnosis') or {}
        treatment_info = final_state.get('treatment_plan') or {}
        emergency_info = final_state.get("emergency_info")
        
      
        workflow_steps = []
        for msg in final_state.get('messages', []):
 
            if hasattr(msg, "content"):
               
                if hasattr(msg, "type") and msg.type == "ai":
                     workflow_steps.append(msg.content)
                elif "assistant" in str(type(msg)).lower():
                     workflow_steps.append(msg.content)
            
           
            elif isinstance(msg, dict):
                if msg.get('role') == 'assistant':
                    workflow_steps.append(msg.get('content', ''))

        logger.info("✅ Medical workflow completed successfully.")

        # Resolve patient record so diagnosis can be persisted and reflected on dashboard/history.
        patient_record: Patient | None = None
        if request.patient_id:
            patient_result = await db.execute(
                select(Patient).where(
                    Patient.id == request.patient_id,
                    Patient.user_id == current_user.id,
                )
            )
            patient_record = patient_result.scalar_one_or_none()

        if patient_record is None:
            latest_patient_result = await db.execute(
                select(Patient)
                .where(Patient.user_id == current_user.id)
                .order_by(Patient.id.desc())
                .limit(1)
            )
            patient_record = latest_patient_result.scalar_one_or_none()

        if patient_record is None:
            patient_record = Patient(
                user_id=current_user.id,
                name=current_user.full_name or current_user.email,
                age=request.age,
                gender=request.gender,
            )
            db.add(patient_record)
            await db.flush()

        urgency = final_state.get('urgency_level', 'ROUTINE')
        severity_map = {
            'EMERGENCY': 'Critical',
            'URGENT': 'High',
            'ROUTINE': 'Low',
            'SELF-CARE': 'Low',
        }
        diagnosis_row = Diagnosis(
            user_id=current_user.id,
            patient_id=patient_record.id,
            symptoms=request.symptoms,
            diagnosis=diagnosis_info.get('primary_diagnosis', 'Emergency escalation triggered' if emergency_info else 'Unable to determine'),
            confidence=float(final_state.get('confidence', 0.0) or 0.0),
            severity=severity_map.get(urgency, 'Low'),
            treatment_plan=treatment_info,
            full_report=final_state.get('final_report', 'Emergency escalation triggered.' if emergency_info else 'Report generation failed.'),
            urgency_level=urgency,
            lung_sound_path=final_state.get('lung_sound_path'),
            lung_sound_result_json=final_state.get('lung_sound_result'),
        )
        db.add(diagnosis_row)
        await db.commit()

        return DiagnosisResponse(
            diagnosis=diagnosis_info.get('primary_diagnosis', 'Emergency escalation triggered' if emergency_info else 'Unable to determine'),
            confidence=final_state.get('confidence', 0.0),
            differential_diagnoses=diagnosis_info.get('differential_diagnoses', []),
            treatment_plan=treatment_info,
            urgency_level=final_state.get('urgency_level', 'ROUTINE'),
            final_report=final_state.get('final_report', 'Emergency escalation triggered.' if emergency_info else 'Report generation failed.'),
            status="CRITICAL" if emergency_info else "OK",
            emergency_info=emergency_info,
            workflow_steps=workflow_steps,
            is_grounded_in_rag=bool(final_state.get('rag_context')),
            citations=final_state.get('rag_context') or []
        )
    
    except Exception as e:
        logger.error(f"❌ Workflow System Failure: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Internal Error: {str(e)}"
        )

@router.post(
    "/agentic-diagnose",
    response_model=DiagnosisResponse,
    status_code=status.HTTP_200_OK,
    summary="Run autonomous (harness + ReAct loop) diagnostic agent",
    description="Same inputs as /diagnose, but driven by a tool harness + ReAct loop: "
    "the model autonomously chooses which tools (lung sound, image, RAG, symptoms, clinics) to call.",
)
async def agentic_diagnosis(
    request: DiagnosisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    try:
        image_analysis_data = None
        if request.image_analysis_id:
            result = await db.execute(
                select(ImageAnalysis).filter(ImageAnalysis.id == request.image_analysis_id)
            )
            image_record = result.scalars().first()
            image_analysis_data = image_record.analysis_result if image_record else None

        # 延迟导入，避免启动期加载 torch/模型相关依赖。
        from app.services.agents.react_agent import ReActDiagnosisAgent

        # 注入患者长期记忆（过敏史 / 慢病史）到诊断上下文
        enriched_history = await _enrich_medical_history(db, current_user, request)
        if enriched_history != (request.medical_history or ""):
            request = request.model_copy(update={"medical_history": enriched_history})

        agent = ReActDiagnosisAgent()
        agent_result = await agent.run(request, image_analysis=image_analysis_data)

        diagnosis_info = agent_result["diagnosis"].model_dump()
        treatment_info = agent_result["treatment_plan"].model_dump()
        urgency = agent_result["urgency_level"]
        is_emergency = bool(agent_result["is_emergency"])

        # Resolve / create patient record for persistence.
        patient_record = None
        if request.patient_id:
            patient_result = await db.execute(
                select(Patient).where(
                    Patient.id == request.patient_id,
                    Patient.user_id == current_user.id,
                )
            )
            patient_record = patient_result.scalar_one_or_none()
        if patient_record is None:
            latest = await db.execute(
                select(Patient).where(Patient.user_id == current_user.id)
                .order_by(Patient.id.desc()).limit(1)
            )
            patient_record = latest.scalar_one_or_none()
        if patient_record is None:
            patient_record = Patient(
                user_id=current_user.id,
                name=current_user.full_name or current_user.email,
                age=request.age,
                gender=request.gender,
            )
            db.add(patient_record)
            await db.flush()

        severity_map = {
            "EMERGENCY": "Critical",
            "URGENT": "High",
            "ROUTINE": "Low",
            "SELF-CARE": "Low",
        }
        diagnosis_row = Diagnosis(
            user_id=current_user.id,
            patient_id=patient_record.id,
            symptoms=request.symptoms,
            diagnosis=diagnosis_info.get("primary_diagnosis", "Unable to determine"),
            confidence=float(diagnosis_info.get("confidence", 0.0) or 0.0),
            severity=severity_map.get(urgency, "Low"),
            treatment_plan=treatment_info,
            full_report=agent_result["final_report"],
            urgency_level=urgency,
            lung_sound_path=request.lung_sound_path,
            lung_sound_result_json=request.lung_sound_result,
        )
        db.add(diagnosis_row)
        await db.commit()

        return DiagnosisResponse(
            diagnosis=diagnosis_info.get("primary_diagnosis", "Unable to determine"),
            confidence=float(diagnosis_info.get("confidence", 0.0) or 0.0),
            differential_diagnoses=diagnosis_info.get("differential_diagnoses", []),
            treatment_plan=treatment_info,
            urgency_level=urgency,
            final_report=agent_result["final_report"],
            status="CRITICAL" if is_emergency else "OK",
            emergency_info=agent_result["emergency_info"],
            workflow_steps=agent_result["workflow_steps"],
            is_grounded_in_rag=bool(agent_result["is_grounded_in_rag"]),
            citations=agent_result["citations"],
        )

    except Exception as e:
        logger.error(f"❌ Agentic Diagnosis System Failure: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal Error: {str(e)}",
        )


@router.get("/health", tags=["System"])
async def get_workflow_health():
    """Endpoint to verify the agent graph is compiled and ready."""
    graph = get_medical_agent_graph()
    if graph:
        return {"status": "ready", "engine": "LangGraph 0.2+", "agents": 6}
    return {"status": "error", "message": "Graph not initialized"}
