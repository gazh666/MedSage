from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.core.config import settings
from typing import Any, List, Dict, Type, TypeVar, Optional
import os

T = TypeVar("T")

# OpenAI 兼容 provider（通义 DashScope / DeepSeek / 智谱 GLM 等国内模型）
_OPENAI_COMPAT_PROVIDERS = {"openai", "dashscope", "qwen", "tongyi", "openai_compatible", "openai-compatible"}


class GeminiClient:
    def __init__(self):
        self.llm: Optional[Any] = None

    @property
    def _provider(self) -> str:
        return (settings.LLM_PROVIDER or "gemini").strip().lower()

    def _is_openai_compat(self) -> bool:
        return self._provider in _OPENAI_COMPAT_PROVIDERS

    def _get_openai_compat_llm(self):
        from langchain_openai import ChatOpenAI

        api_key = settings.LLM_API_KEY or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                f"LLM_API_KEY 未配置（provider={self._provider}）。"
                "请在 .env 中设置 LLM_API_KEY（或 DASHSCOPE_API_KEY）。"
            )
        return ChatOpenAI(
            model=settings.LLM_MODEL or "qwen-plus",
            api_key=api_key,
            base_url=settings.LLM_BASE_URL or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.1,
            max_retries=2,
        )

    def _get_gemini_llm(self):
        if not settings.GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is not configured")

        os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
            max_retries=2,
        )

    def _get_llm(self):
        """Create the LLM client lazily to avoid crashing application startup."""
        if self.llm is not None:
            return self.llm
        self.llm = self._get_openai_compat_llm() if self._is_openai_compat() else self._get_gemini_llm()
        return self.llm

    def _normalize_content(self, content: Any) -> str:
        """Normalize multimodal/structured model payloads into plain text."""
        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str) and text.strip():
                return text
            parts = content.get("parts")
            if isinstance(parts, list):
                normalized = "\n".join(
                    self._normalize_content(part) for part in parts if self._normalize_content(part).strip()
                )
                if normalized.strip():
                    return normalized
            return str(content)

        if isinstance(content, list):
            normalized_parts: list[str] = []
            for part in content:
                normalized = self._normalize_content(part)
                if normalized.strip():
                    normalized_parts.append(normalized)
            if normalized_parts:
                return "\n".join(normalized_parts)
            return ""

        return str(content)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ) -> str:
        llm = self._get_llm()
        lc_messages = []
        
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        
        response = await llm.ainvoke(lc_messages)
        return self._normalize_content(response.content)
    
    async def generate(self, prompt: str) -> str:
        """Standard text generation."""
        llm = self._get_llm()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return self._normalize_content(response.content)

    async def generate_structured(self, prompt: str, response_model: Type[T]) -> T:
        """
        Generates a structured response mapped to a Pydantic model.
        Fixes the AttributeError in agent nodes by using LangChain's native parser.
        """
        llm = self._get_llm()
        # .with_structured_output handles the JSON schema conversion automatically
        structured_llm = llm.with_structured_output(response_model)
        
        response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        return response

    async def invoke_with_tools(self, messages: List[Any], tools: List[Any]) -> Any:
        """带工具绑定的单轮推理（function calling）。

        返回的 AIMessage 可能携带 `tool_calls`（[{name, args, id}, ...]），
        供 ReAct 主循环执行工具。
        """
        llm = self._get_llm().bind_tools(tools)
        response = await llm.ainvoke(messages)
        return response

# Singleton instance
gemini_client = GeminiClient()