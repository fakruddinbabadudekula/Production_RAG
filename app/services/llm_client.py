"""Service module contains llm service using llm interface from rag module"""

from app.rag.interface import AsyncLLMClient
from langchain_core.messages import AIMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from app.core.config import settings
from openai import (
    APIError,
    RateLimitError,
)
import time
import asyncio
import logging
from logging import getLogger
from functools import lru_cache
from langchain_openai import ChatOpenAI

logger = getLogger(__name__)
RETRYABLE_LLM_EXCEPTIONS = (
    ConnectionError,
    asyncio.TimeoutError,
    RateLimitError,
    APIError,
)


@lru_cache()
def get_llm():
    # for now we directly return the chatopenai instance we can later add differnt llms and get according to our needs.
    """Return the chatopenai llm instance."""
    return ChatOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        model=settings.CURRENT_CHAT_MODEL,
        temperature=settings.TEMPERATURE,
        streaming=True,
        timeout=settings.CHAT_MODEL_TIMEOUT,
        max_retries=0,
    )


class LLMClient(AsyncLLMClient):
    def __init__(self):
        self.llm = get_llm()

    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=32),  # 2s, 4s, 8s, 16s, 32s
        retry=retry_if_exception_type(RETRYABLE_LLM_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Raise the original exception after all retries fail
    )
    async def _call_llm_with_retries(self, prompt: str) -> AIMessage:
        try:
            start = time.perf_counter()
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt), timeout=settings.LLM_CALL_ASYNC_TIMEOUT
            )
            duration = time.perf_counter() - start
            logger.info(
                "response_is_generated_successfully", extra={"duration": duration}
            )
            return response
        except RETRYABLE_LLM_EXCEPTIONS as e:
            logger.warning("llm_call_failed_retrying", extra={"error": str(e)})
            raise

    async def call(self, prompt: str) -> AIMessage:
        """Generate a response from the language model.

        Args:
            prompt: str
                Prompt sent to the language model.

        Returns:
            AIMessage:
                Generated response.
        """
        return await self._call_llm_with_retries(prompt=prompt)


@lru_cache(maxsize=1)
def get_llm_client():
    return LLMClient()


llm_client = get_llm_client()
