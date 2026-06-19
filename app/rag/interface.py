from abc import ABC, abstractmethod


class AsyncLLMClient(ABC):
    @abstractmethod
    async def call(self, prompt: str):
        pass

