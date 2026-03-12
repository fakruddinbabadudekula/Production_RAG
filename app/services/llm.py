import logging
import asyncio
from urllib import response
from langchain_core.messages import AIMessage
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
from openai import (
    APIError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)
from app.core.config import settings
from app.core.agent.models import get_llm
import time
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RETRYABLE_LLM_EXCEPTIONS = (
    ConnectionError,
    RateLimitError,
    APIError,
)


class LLMService:
    """For now just implemented the only call the llm with retires later updated to fallback,model selection..."""

    def __init__(self):
        self.llm = get_llm()

    @retry(
        stop=stop_after_attempt(settings.MAX_LLM_CALL_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=32),  # 2s, 4s, 8s, 16s, 32s
        retry=retry_if_exception_type(RETRYABLE_LLM_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Raise the original exception after all retries fail
    )
    async def _call_llm_with_retries(self, final_prompt: str) -> AIMessage:
        try:
            start=time.perf_counter()
            response=await asyncio.wait_for(
                self.llm.ainvoke(final_prompt), timeout=settings.LLM_CALL_ASYNC_TIMEOUT
            )
            duration=time.perf_counter()-start
            logger.info("response_is_generated_successfully. duration= %.2fs",duration)
            return response
        except RETRYABLE_LLM_EXCEPTIONS as e:
            logger.warning(
                "llm_call_failed_retrying  error= %s",str(e)
            )
            raise
        except (asyncio.TimeoutError, TimeoutError, APITimeoutError) as e:
            logger.error(
                f"LLM timed out after {settings.LLM_CALL_ASYNC_TIMEOUT}s "
                f"— not retrying, failing fast"
            )
            raise

    async def call(self, final_prompt: str) -> AIMessage:
        """For now just implemented the calling llm with final prompt and return the response"""
        return await self._call_llm_with_retries(final_prompt=final_prompt)


llm_service = LLMService()
