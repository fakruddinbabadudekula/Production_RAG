# In this file create the langgraph agent class which is retrieve the top k results and give the answer to the user query
from re import S
from langchain_core.documents.base import Document
from langchain_core.messages import AIMessage
from app.agent.retrievers.vector_retriever import Retriever
from langgraph.graph import StateGraph, END, START
from typing import List, Dict, AsyncIterator
from app.agent.models import get_llm
from app.schemas.agent import GraphState
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
import logging
from pathlib import Path
from app.config import settings
from dotenv import load_dotenv

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_retriever(vector_path: Path):
    """Return the retriever Object"""
    return Retriever(vector_dir_path=vector_path)


RETRYABLE_LLM_EXCEPTIONS = (
    ConnectionError,  # Network issues
    TimeoutError,  # API timeout
    # Catch provider-specific errors (RateLimitError, etc.)
    # e.g., openai.RateLimitError, openai.APIConnectionError, etc.
)


class GraphError(Exception):
    """Custom exception for graph operations."""

    def __init__(
        self,
        message: str,
        operation: str,
        original_error: Exception,
        user_id: str,
        session_id: str,
    ):
        full_message = f"{message}. Original error: {type(original_error).__name__}: {str(original_error)}"
        logger.error(
            f"message: {full_message} | operation: {operation} exception: {str(original_error)}| user_id/session_id: {user_id}/{session_id}"
        )
        super().__init__(full_message)
        self.operation = operation
        self.original_error = original_error
        self.user_id = user_id
        self.session_id = session_id


class Graph:
    def __init__(self, user_id: str, session_id: str):
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
        self.llm = get_llm()
        self.user_id = user_id
        self.session_id = session_id
        self.vector_path = self._get_vector_path(user_id=user_id, session_id=session_id)
        self.saver: InMemorySaver = InMemorySaver()
        self.graph: CompiledStateGraph = self._get_graph()
        self.retriever: Retriever = get_retriever(self.vector_path)

    @staticmethod
    def _get_vector_path(user_id: str, session_id: str) -> Path:
        """Sanitize the file path
        raises:
            - ValueError: If any other paths are given
        """
        vector_dir_path = (settings.VECTOR_FOLDER / user_id / session_id).resolve()
        if not vector_dir_path.is_relative_to(settings.VECTOR_FOLDER):
            raise ValueError(
                f"Vector file address must be within the limit.Path=> {vector_dir_path}"
            )
        return vector_dir_path

    def _get_graph(self) -> CompiledStateGraph:
        """Build and compile the LangGraph workflow.

        Returns:
            CompiledStateGraph: Compiled graph with retriever and chat nodes.

        Raises:
            Exception: If graph compilation fails.
        """
        try:
            workflow = StateGraph(GraphState)
            workflow.add_node("retriever", self._retriever)
            workflow.add_node("chat", self._chat)
            workflow.add_edge(START, "retriever")
            workflow.add_edge("retriever", "chat")
            workflow.add_edge("chat", END)
            graph = workflow.compile(checkpointer=self.saver)
            return graph
        except Exception as e:
            logger.error(f"{str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(5),  #Try 5 times (patient retry for expensive LLM calls)
        wait=wait_exponential(multiplier=1, min=2, max=32),  # 2s, 4s, 8s, 16s, 32s
        retry=retry_if_exception_type(RETRYABLE_LLM_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Raise the original exception after all retries fail
    )
    async def _llm_invoke(self, final_prompt: str) -> AIMessage:
        return await self.llm.ainvoke(final_prompt)

    async def _chat(self, state: GraphState):
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
        try:
            response = await self._llm_invoke(final_prompt)
        except RETRYABLE_LLM_EXCEPTIONS as e:
            error_msg = (
                f"Failed to invoke the llm. "
                f"Query: '{final_prompt[:50]}...'. "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            raise GraphError(
                message=error_msg,
                operation=f"LLm_Invoke",
                original_error=e,
                user_id=self.user_id,
                session_id=self.session_id,
            )
        except Exception as e:
            logger.error(
                f"Unknown Error Occured while invoking the llm->{str(e)} for user_id: {self.user_id} in session_id: {self.session_id}"
            )
            raise
        logger.info(f"Response is generated successfully for {state['messages']}")
        return {"messages": [response]}

    async def _retriever(self, state: GraphState):
        """Retrieve top-k relevant documents for the query.

        Args:
            state: Current graph state containing user messages.

        Returns:
            Dict: Updated state containing retrieved document metadata.

        Raises:
            Exception: If retrieval fails.
        """
        query = state["messages"][-1].content
        try:
            top_k_docs = await self.retriever.aget_top_k(query=query)
            sources_data = self._formate_docs_to_list_dict(top_k_docs=top_k_docs)
        except Exception as e:
            error_msg = (
                f"Failed to retrieve documents. "
                f"Query: '{query[:50]}...'. "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            logger.error(error_msg)
            raise GraphError(
                message=error_msg,
                operation=f"Retrieving",
                original_error=e,
                user_id=self.user_id,
                session_id=self.session_id,
            )
        return {"retrieved_docs": sources_data}

    async def get_response_stream(self, query: str) -> AsyncIterator[Dict]:
        """Stream retrieved documents and generated response tokens.

        Args:
            query: User query string.

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
                "configurable": {"thread_id": self.session_id},
            }
            async for mode, data in self.graph.astream(
                {"messages": [query]},
                stream_mode=["messages", "updates"],
                config=config,
            ):
                if mode == "updates":
                    if data.get("retriever", None):
                        logger.info(f"Started Top_k_docs Streaming.")
                        top_k_docs = data.get("retriever").get("retrieved_docs", [])
                        yield {"type": "top_k_docs", "data": top_k_docs}
                        logger.info("Top_k_docs streaming is completed")

                elif mode == "messages":
                    chunk, _ = data
                    # print(type(chunk.content))
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if not token.strip():
                        continue
                    yield {"type": "token", "value": token}
            logger.info(f"Chat Streaming is Completed.")
            logger.info("Streaming is completed")
        except GraphError:
            # Re-raise GraphError (already has context)
            raise
        except Exception as e:
            error_msg = f"Streaming failed: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            raise GraphError(
                message=error_msg,
                operation=f"Streaming",
                original_error=e,
                user_id=self.user_id,
                session_id=self.session_id,
            )

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
            logger.info(f"No relavant docs are found for the query {query}")
            final_prompt = query
        else:
            content = []
            for i, data in enumerate(sources_data):
                content.append(f"{i+1} {data['content']}")
            context = "\n\n".join(content)
            final_prompt = self._create_rag_prompt(query=query, context=context)
        return final_prompt

    def _create_rag_prompt(self, query: str, context: str) -> str:
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

    def _formate_docs_to_list_dict(self, top_k_docs):
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

    async def add_docs(self, docs: List[Document]) -> list[str]:
        """Add documents to the vector store.

        Args:
            docs: List of LangChain Document objects.

        Raises:
            ValueError: If the document list is empty.
            Exception: If adding documents fails.

        Return:
            list[str]: Ids of the docs in vector_db
        """
        if len(docs) == 0 or not docs:
            logger.error(f"Documents list must contain at least one document.")
            raise ValueError(f"Documents list must contain at least one document")
        try:
            return await self.retriever.aadd_documents(docs=docs)
        except Exception as e:
            error_msg = (
                f"Failed to add {len(docs)} documents. "
                f"Error: {type(e).__name__}: {str(e)}"
            )
            logger.error(error_msg)
            raise GraphError(
                message=error_msg,
                operation=f"Adding_Docs",
                original_error=e,
                user_id=self.user_id,
                session_id=self.session_id,
            )
        