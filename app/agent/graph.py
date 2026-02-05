# In this file create the langgraph agent class which is retrieve the top k results and give the answer to the user query
from app.agent.retrievers.vector_retriever import Retriever
from langgraph.graph import StateGraph, END, START
from typing import List, Dict, Tuple, Optional
from app.agent.models import get_llm
from functools import lru_cache
from app.schemas.agent import GraphState
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
from app.agent.document_loaders.doc_loader import DocumentLoader
import logging
from app.config import settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@lru_cache()
def get_retriever(vector_dir: str):
    return Retriever(vector_dir=vector_dir)


class Graph:
    def __init__(self):
        self.retriever: Retriever = None
        self.llm: None
        self.user_id: Optional[str] = (
            "test_session_user"  # use this value to store all vectors chat values in under this folder name
        )
        self.session_id: Optional[str] = (
            "test_session"  # use this value for chat seperation vectors
        )
        self.vector_path: Optional[str] = None
        self.graph: Optional[CompiledStateGraph] = None
        self.saver: InMemorySaver = InMemorySaver()

    def _get_graph(self):
        try:
            workflow = StateGraph(GraphState)
            workflow.add_node("chat", self._chat)
            workflow.add_edge(START, "chat")
            workflow.add_edge("chat", END)
            graph = workflow.compile(checkpointer=self.saver)
            return graph
        except Exception as e:
            logger.error(f"{str(e)}")

    async def _initialize_all_variables(self, user_id, session_id):
        self.llm = get_llm()
        logger.info(f"llm is initialized")
        self.user_id, self.session_id = user_id, session_id
        path = f"{settings.VECTOR_FOLDER}{self.user_id}/{self.session_id}"
        logger.info(f"Here is the path for vector DB: {path}")
        self.vector_path = path
        if not self.retriever:
            self.retriever = get_retriever(self.vector_path)
            logger.info("retriever is initialized from the graph")

            # Here document loader is not need i just put here to add the attention is all you need.pdf vectors for first time
            pdf_loader = DocumentLoader()
            chunks = await pdf_loader.process_document(
                path="app/agent/data/attention is all you need.pdf"
            )

            await self.retriever.aadd_documents(docs=chunks)

        self.graph = self._get_graph()
        logger.info(f"All states are initialized successfully in graph.")

    async def _chat(self, state: GraphState):
        final_prompt, metadata = await self._final_prompt_with_metadata(
            query=state["messages"][-1].content
        )
        response = await self.llm.ainvoke(final_prompt)
        logger.info(f"Response is generated successfully for {state['messages']}")
        return {"messages": [response], "metadata": metadata}

    async def get_response_stream(self, query, user_id, session_id):
        if not self.graph:
            await self._initialize_all_variables(user_id=user_id, session_id=session_id)
        try:
            config = {
                "configurable": {"thread_id": session_id},
            }
            async for chunk, _ in self.graph.astream(
                {"messages": [query]}, stream_mode="messages",config=config
            ):
                token =chunk.content if hasattr(chunk, "content") else str(chunk)
                if not token.strip():
                    continue
                yield {
                       'type':"token",
                       'value':token}
            logger.info(f"Chat Streaming is Completed.")
            last_state=self.graph.get_state(config=config)
            # print(last_state)
            logger.info(f"Started Metadata Streaming.")
            metadata=last_state.values['metadata']
            # print(metadata)
            yield {
                'type':"metadata",
                'value':metadata
            }
            logger.info("Metadata streaming is completed")
            logger.info("Streaming is completed")
        except Exception as e:
            logger.error(f"{str(e)}")
            raise
            
    def _get_retriever(self, vector_path) -> Retriever:
        try:
            return Retriever(vector_dir=vector_path)
        except Exception as e:
            logger.error(f"{str(e)}")

    async def _final_prompt_with_metadata(
        self, query: str
    ) -> Tuple[str | None, List[Dict] | None]:
        try:
            top_k_docs = await self.retriever.aget_top_k(query=query)
            if not top_k_docs:
                logger.info(f"No relavant docs are found for the query {query}")
                final_prompt, metadata = query, [{'data':"No data is Found"}]
            else:
                context, metadata = self._format_context_with_citations(top_k_docs)
                final_prompt = self._create_rag_prompt(query=query, context=context)
        except Exception as e:
            logger.error(f"error while generating top_k docs and metadata {e}")
            raise
        return (final_prompt, metadata)

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
