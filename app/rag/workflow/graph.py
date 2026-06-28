"""Rag module for graph which is orchestration of rag workflow."""

import time
from ..interface import AsyncLLMClient
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END, START
from langchain_core.vectorstores.base import VectorStoreRetriever
from langgraph.graph import MessagesState
from langchain_core.runnables import RunnableConfig
from typing import List, Dict
from langchain_core.messages import BaseMessage
from logging import getLogger

logger = getLogger(__name__)


class GraphState(MessagesState):
    """State shared between LangGraph nodes"""

    retrieved_docs: List[Dict] | None


class Graph:
    def __init__(self, llm_client: AsyncLLMClient):
        
        """Initialize the RAG graph.
        Args:
            llm_client: AsyncLLMClient Asynchronous language model client used for response generation.
        """
        self.llm_client = llm_client
        self.graph = self._get_graph()

    def _get_graph(self) -> CompiledStateGraph:
        """
        Build and compile the LangGraph workflow.

        Returns:
            CompiledStateGraph: Compiled graph with retriever and chat nodes.
        """

        workflow = StateGraph(GraphState)
        workflow.add_node("retriever", self._retriever)
        workflow.add_node("chat", self._chat)
        workflow.add_edge(START, "retriever")
        workflow.add_edge("retriever", "chat")
        workflow.add_edge("chat", END)
        graph = workflow.compile()
        logger.info("graph_compiled")
        return graph

    async def _chat(
        self, state: GraphState, config: RunnableConfig
    ) -> dict[str, list[dict]]:
        """Generate an LLM response using retrieved documents.

        Args:
            state: Current graph state containing messages and retrieved documents.

        Returns:
            Dict: Updated state containing generated LLM message.

        Raises:
        """
        final_prompt = self._final_prompt_with_sources(
            query=state["messages"][-1].content, sources_data=state["retrieved_docs"]
        )
        response = await self.llm_client.call(final_prompt)
        return {"messages": [response]}

    async def _retriever(
        self, state: GraphState, config: RunnableConfig
    ) -> dict[str, list[dict]]:
        """Retrieve top-k relevant documents for the query.

        Args:
            state: Current graph state containing user messages.

        Returns:
            Dict: Updated state containing retrieved document metadata.

        """
        query = state["messages"][-1].content

        retriever: VectorStoreRetriever = config["configurable"]["retriever"]
        start_time = time.perf_counter()
        docs = await retriever.ainvoke(query)
        duration = time.perf_counter() - start_time
        logger.info(
            "retreived_relavent_docs",
            extra={"count": len(docs), "duration": duration},
        )
        sources_data = self._formate_docs_to_list_dict(top_k_docs=docs)

        return {"retrieved_docs": sources_data}

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

    async def ainvoke(
        self, messages: List[BaseMessage], retriever: VectorStoreRetriever
    )->dict:
        """Async function to invoke the rag graph.
        Args:
            messages: 
                list of messages to pass rag graph.
            retriever:
                a vector_store retreiver to get top_k_docs.
        Returns:
            dict:
                contains= top_k_docs and response."""
        if not messages or len(messages) == 0:
            raise ValueError(f"messages_should_not_be_empty.")
        config = {"configurable": {"retriever": retriever}}
        time_taken = time.perf_counter()
        response = await self.graph.ainvoke(
            {"messages": messages},
            config=config,
        )
        duration = time.perf_counter() - time_taken
        logger.info(
            "graph_completed_the_response_generation",
            extra={"duration": duration},
        )
        return {
            "top_k_docs": response["retrieved_docs"],
            "response": response["messages"][-1],
        }
