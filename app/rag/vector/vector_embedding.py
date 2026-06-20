from functools import lru_cache
from langchain_community.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.documents.base import Document
from pathlib import Path
from typing import List
from logging import getLogger
import time
logger = getLogger(__name__)


class VectorEmbedding:
    def __init__(self):
        pass

    async def aad_documents(
        self, vector_store: VectorStore, vector_dir_path: Path, docs: List[Document]
    )->List[str]:
        start=time.perf_counter()
        if len(docs) == 0 or not docs:
            raise ValueError(f"Docs must be atleast one. Passed empty")
        docs_ids = await vector_store.aadd_documents(docs)
        vector_store.save_local(vector_dir_path)
        logger.info(
            "Successfully_added_docs_to_vector_store",
            extra={
            "count":len(docs),
            "vector_path":vector_dir_path,
            "duration":time.perf_counter()-start}
        )
        return docs_ids

    async def get_retriever(
        self,
        vector_store: VectorStore,
        search_type: str = "similarity",
        search_kwargs: dict = {"k": 5},
    ) -> VectorStoreRetriever:
    
        retriever=vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        return retriever
    
@lru_cache(maxsize=1)
def get_vector_embedding()->VectorEmbedding:
    return VectorEmbedding()

vector_embedding=get_vector_embedding()