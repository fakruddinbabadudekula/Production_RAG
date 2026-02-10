# In this file create the langgraph agent class which is retrieve the top k results and give the answer to the user query
from app.agent.retrievers.vector_retriever import Retriever
from langgraph.graph import StateGraph, END, START
from typing import List, Dict
from app.agent.models import get_llm
from app.schemas.agent import GraphState
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver
import logging
from pathlib import Path
from app.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_retriever(vector_path: str):
    return Retriever(vector_dir=vector_path)


class Graph:
    def __init__(self, user_id: str, session_id: str):
        self.llm = get_llm()
        self.user_id = user_id
        self.session_id = session_id
        self.vector_path = f"{settings.VECTOR_FOLDER}{self.user_id}/{self.session_id}"
        self.saver: InMemorySaver = InMemorySaver()
        self.graph: CompiledStateGraph = self._get_graph()
        self.retriever:Retriever = get_retriever(self.vector_path)

    def _get_graph(self):
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

    async def _chat(self, state: GraphState):
        final_prompt = self._final_prompt_with_sources(
            query=state["messages"][-1].content, sources_data=state["retrieved_docs"]
        )
        response = await self.llm.ainvoke(final_prompt)
        logger.info(f"Response is generated successfully for {state['messages']}")
        return {"messages": [response]}

    async def _retriever(self, state: GraphState):
        query = state["messages"][-1].content
        try:
            top_k_docs = await self.retriever.aget_top_k(query=query)
            sources_data = self._formate_docs_to_list_dict(top_k_docs=top_k_docs)
        except Exception as e:
            logger.error(f"error while generating top_k docs and metadata {e}")
            raise
        return {"retrieved_docs": sources_data}

    async def get_response_stream(self, query):
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
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if not token.strip():
                        continue
                    yield {"type": "token", "value": token}
            logger.info(f"Chat Streaming is Completed.")
            logger.info("Streaming is completed")
        except Exception as e:
            logger.error(f"{str(e)}")
            raise

    def _final_prompt_with_sources(
        self, query: str, sources_data: List[Dict] | None
    ) -> str:
        try:
            if not sources_data:
                logger.info(f"No relavant docs are found for the query {query}")
                final_prompt = query
            else:
                content = []
                for i, data in enumerate(sources_data):
                    content.append(f"{i+1} {data['content']}")
                context = "\n\n".join(content)
                final_prompt = self._create_rag_prompt(query=query, context=context)
        except Exception as e:
            logger.error(f"error while generating top_k docs and metadata {e}")
            raise
        return final_prompt

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

    def _formate_docs_to_list_dict(self, top_k_docs):
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

    async def add_docs(self,docs):
        logger.info(f"Documents are added.")
        await self.retriever.aadd_documents(docs=docs)