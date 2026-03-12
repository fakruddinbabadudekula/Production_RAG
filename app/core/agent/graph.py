# In this file create the langgraph agent class which is retrieve the top k results and give the answer to the user query
from app.core.agent.retrievers.vector_retriever import Retriever
from langgraph.graph import StateGraph, END, START
from typing import List, Dict, AsyncIterator
from app.services.llm import llm_service
from app.schemas.agent import GraphState
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
import logging
from pathlib import Path
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langchain_core.runnables import RunnableConfig
from functools import lru_cache
from app.utils.graph import get_vector_path
from app.core.exceptions import GraphError, VectorStoreError
from app.core.config import settings


load_dotenv()
logger = logging.getLogger(__name__)


@lru_cache()
def get_retriever(vector_path: Path):
    """Return the retriever Object"""
    return Retriever(vector_dir_path=vector_path)


class Graph:
    def __init__(self):
        """
        LangGraph-based RAG agent for retrieval and response generation.

        Orchestrates document retrieval and LLM-based answer generation
        using a stateful LangGraph workflow.
        Initialize the graph agent for a user session.


        Args:
            user_id: Unique identifier for the user.
            session_id: Unique identifier for the session.

        Raises:
            ValueError: If vector path validation fails.
        """
        self.llm_service = llm_service
        self.saver: InMemorySaver = InMemorySaver()
        self.graph: CompiledStateGraph = self._get_graph()

    def _get_graph(self) -> CompiledStateGraph:
        """Build and compile the LangGraph workflow.

        Returns:
            CompiledStateGraph: Compiled graph with retriever and chat nodes.

        Raises:
            GraphError: If graph compilation fails.
        """
        try:
            workflow = StateGraph(GraphState)
            workflow.add_node("retriever", self._retriever)
            workflow.add_node("chat", self._chat)
            workflow.add_edge(START, "retriever")
            workflow.add_edge("retriever", "chat")
            workflow.add_edge("chat", END)
            graph = workflow.compile(checkpointer=self.saver)
            logger.info("graph_compiled")
            return graph
        except Exception as e:
            raise GraphError(
                "Unkown_excepiton or compilation fails",
                operation="creating_and_compiling_graph",
                original_error=e,
            ) from e

    async def _chat(self, state: GraphState, config: RunnableConfig):
        """Generate an LLM response using retrieved documents.

        Args:
            state: Current graph state containing messages and retrieved documents.

        Returns:
            Dict: Updated state containing generated LLM message.

        Raises:
            GraphError:If LLM invocation fails after retries.
            Exception: If LLM invocation fails with unknown error.
        """
        final_prompt = self._final_prompt_with_sources(
            query=state["messages"][-1].content, sources_data=state["retrieved_docs"]
        )
        user_id, session_id = (
            config["metadata"]["user_id"],
            config["metadata"]["session_id"],
        )
        try:
            response = await self.llm_service.call(final_prompt)
        except settings.RETRYABLE_LLM_EXCEPTIONS as e:
            error_msg = (
                f"Failed to invoke the llm. After {settings.MAX_LLM_CALL_RETRIES} retries "
                f"Query: '{final_prompt[:50]}...'. "
            )
            raise GraphError(
                message=error_msg,
                operation=f"LLM_invoke",
                original_error=e,
                user_id=user_id,
                session_id=session_id,
            ) from e
        except Exception as e:
            raise GraphError(
                message="unkown_error",
                operation="LLM_invoke",
                original_error=e,
                user_id=user_id,
                session_id=session_id,
            ) from e
        return {"messages": [response]}

    async def _retriever(self, state: GraphState, config: RunnableConfig):
        """Retrieve top-k relevant documents for the query.

        Args:
            state: Current graph state containing user messages.

        Returns:
            Dict: Updated state containing retrieved document metadata.

        Raises:
            VectorStoreError: If retrieval fails.
        """
        query = state["messages"][-1].content
        user_id, session_id = (
            config["metadata"]["user_id"],
            config["metadata"]["session_id"],
        )
        vector_path = get_vector_path(user_id=user_id, session_id=session_id)
        retriever = get_retriever(vector_path=vector_path)
        logger.info("got_retriever_successfully vector_path= %s",vector_path)
        top_k_docs = await retriever.aget_top_k(query=query)
        logger.info("retreived_top_k_docs")
        sources_data = self._formate_docs_to_list_dict(top_k_docs=top_k_docs)

        return {"retrieved_docs": sources_data}

    async def get_response_stream(
        self, query: str, user_id: str, session_id: str
    ) -> AsyncIterator[Dict]:
        """Stream retrieved documents and generated response tokens.

        Args:
            query: User query string.
            user_id: User unique id
            session_id: Unique session id for multi conversation

        Yields:
            Dict: Streaming event dictionary containing either:
                - {"type": "top_k_docs", "data": [...]}
                - {"type": "token", "value": str}

        Raises:
            ValueError: If query is empty.
            GraphError: If streaming execution fails after retries
            Exception: If streaming execution fails.
        """
        if not query or not query.strip():
            raise ValueError(f"query should not be empty.")
        try:
            config = {
                "configurable": {"thread_id": session_id},
                "callbacks": [CallbackHandler()],
                "metadata": {"user_id": user_id, "session_id": session_id},
            }
            logger.info("started_graph_streaming")
            async for mode, data in self.graph.astream(
                {"messages": [query]},
                stream_mode=["messages", "updates"],
                config=config,
            ):
                if mode == "updates":
                    if data.get("retriever", None):
                        logger.info(f"started_top_k_streaming")
                        top_k_docs = data.get("retriever").get("retrieved_docs", [])
                        yield {"type": "top_k_docs", "data": top_k_docs}
                        logger.info("complited_top_k_streaming")

                elif mode == "messages":
                    chunk, _ = data
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if not token.strip():
                        continue
                    yield {"type": "token", "value": token}
            logger.info("complited_graph_streaming")
        except GraphError:
            # Re-raise GraphError (already has context)
            raise
        except VectorStoreError:
            raise
        except Exception as e:
            error_msg = f"Streaming failed."
            raise GraphError(
                message=error_msg,
                operation=f"streaming",
                original_error=e,
                user_id=user_id,
                session_id=session_id,
            ) from e

    def _final_prompt_with_sources(
        self, query: str, sources_data: List[Dict] | None
    ) -> str:
        """Create the final prompt including retrieved source context.

        Args:
            query: User query string.
            sources_data: List of retrieved document metadata.

        Returns:
            str: Constructed RAG prompt.
        """
        if not sources_data:
            final_prompt = f"{query} Answer only if you know with certainty, otherwise say you don't know."
        else:
            content = []
            for i, data in enumerate(sources_data):
                content.append(f"{i+1} {data['content']}")
            context = "\n\n".join(content)
            final_prompt = self._create_rag_prompt(query=query, context=context)
        return final_prompt

    @staticmethod
    def _create_rag_prompt(query: str, context: str) -> str:
        """Construct a citation-enforced RAG prompt.

        Args:
            query: User query string.
            context: Concatenated source document content.

        Returns:
            str: Fully formatted RAG prompt with citation rules.
        """
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

    @staticmethod
    def _formate_docs_to_list_dict(top_k_docs):
        """Convert retrieved Document objects into serializable metadata dictionaries.

        Args:
            top_k_docs: List of retrieved LangChain Document objects.

        Returns:
            List[Dict]: List of formatted document metadata dictionaries.
                "index",
                "source",
                "page",
                "file_path",
                "format",
                "title",
                "content"
        """
        source_metadata = []
        if not top_k_docs:
            return []

        for i, doc in enumerate(top_k_docs):
            metadata = {
                "index": i + 1,
                "source": doc.metadata.get("source", "No source available"),
                "page": doc.metadata.get("page", "no page number available"),
                "file_path": doc.metadata.get("file_path"),
                "format": doc.metadata.get("format", "no format available"),
                "title": doc.metadata.get("title", "title is not available"),
                "content": doc.page_content,
            }
            source_metadata.append(metadata)
        return source_metadata
