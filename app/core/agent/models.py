# In this code we explore the different types of models and export 
from functools import lru_cache
from langchain_openai import ChatOpenAI
from app.core.config import settings

@lru_cache()
def get_llm():
  """Return the chatopenai llm instance."""
  return ChatOpenAI(
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
    model=settings.CURRENT_CHAT_MODEL,
    temperature=settings.TEMPERATURE,
    streaming=True,
    timeout=settings.CHAT_MODEL_TIMEOUT,
    max_retries=0
  )
