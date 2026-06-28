"""Rag Module for LLMService Interface"""

from abc import ABC, abstractmethod


class AsyncLLMClient(ABC):
    """An interface for llm client"""
    @abstractmethod
    async def call(self, prompt: str):
        pass

