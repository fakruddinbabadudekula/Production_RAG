from typing import List, Dict, Tuple
from app.agent.retrievers.vector_retriever import Retriever
from app.agent.document_loaders.doc_loader import DocumentLoader
from dataclasses import dataclass
from app.agent.models import get_llm
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    query: str
    response: str
    sources_metadata: List[Dict]


class RAG:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.llm=get_llm()

    async def _final_prompt_with_metadata(
        self, query: str
    ) -> Tuple[str | None, List[Dict] | None]:
        try:
            top_k_docs = await self.retriever.aget_top_k(query=query)
            if not top_k_docs:
                logger.info(f"No relavant docs are found for the query {query}")
                final_prompt, metadata = None, None
            else:
                context, metadata = self._format_context_with_citations(top_k_docs)
                final_prompt = self._create_rag_prompt(query=query, context=context)
        except Exception as e:
            logger.error(f"error while generating top_k docs and metadata {e}")
            raise
        return (final_prompt, metadata)

    async def ainvoke(self, query: str) -> RAGResult:

        try:
            final_prompt, metadata = await self._final_prompt_with_metadata(query=query)
            response = await self.llm.ainvoke(final_prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            logger.info(f"Response is generated successfully for {query}")
            return RAGResult(query=query, response=answer, sources_metadata=metadata)
        except Exception as e:
            logger.error(f"Encounter erro while generating response {e}")
            return RAGResult(
                query=query,
                response=f"I encountered an error while generating the response {e}",
                sources_metadata=[],
            )

    async def astream(self, query: str):
        try:
            logger.info(f"Started astreaming.")
            final_prompt, metadata = await self._final_prompt_with_metadata(query=query)
            logger.info(f"Created final_prompt and metadata")
            if not final_prompt or not metadata:
                logger.info(f"No Sources Found")
                yield {'type': 'source', 'data': []}
                yield {'type': 'data', 'data': 'No Sources Found '}
                yield {'type': 'done'}
            logger.info(f"Started metadata Streaming")
            yield {'type': 'sources', 'data': metadata}
            logger.info(f"Started chunk Streaming")
            async for chunk in self.llm.astream(final_prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if not token.strip():
                    continue
                yield {'type': 'data', 'data': token}
            logger.info(f"Streaming is Done")
            yield {'type': 'done'}

        except Exception as e:
            yield {'type': 'error', 'error': str(e)}

    def _create_rag_prompt(self, query: str, context: str) -> str:
        prompt = f"""You are an AI assistant that answers questions based on provided source material. You must follow these citation rules:

                        CITATION REQUIREMENTS:
                        1. For each factual claim in your answer, include the citation reference number in square brackets [1], [2], etc.
                        2. Only use information from the provided context - do not add external knowledge
                        3. If you cannot find relevant information in the context, say so clearly
                        4. Be precise and accurate in your citations
                        5. When multiple sources support the same point, list all relevant citations like this [1], [2], [3].

                        CONTEXT (with citation references):
                        {context}

                        QUESTION: {query}

                        Please provide a comprehensive answer with proper citations. Make sure every factual statement is supported by a citation reference."""

        return prompt

    def _format_context_with_citations(self, top_k_docs):
        context_parts = []
        source_metadata = []
        for i, doc in enumerate(top_k_docs):
            content_index = f"[{i+1}]"
            doc_content = doc.page_content
            content = f"{content_index} {doc_content}"
            doc_metadata = doc.metadata
            metadata = {
                "index": i + 1,
                "source": doc_metadata.get("source", "No source available"),
                "page": doc_metadata.get("page", "no page number available"),
                "file_path": doc_metadata.get("file_path"),
                "format": doc_metadata.get("format", "no format available"),
                "title": doc_metadata.get("title", "title is not available"),
                "content": doc_content,
            }
            context_parts.append(content)
            source_metadata.append(metadata)
        context = "\n\n".join(context_parts)
        return context, source_metadata
