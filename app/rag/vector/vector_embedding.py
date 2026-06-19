from functools import lru_cache
from langchain_community.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.documents.base import Document
from pathlib import Path
from typing import List
from logging import getLogger

logger = getLogger(__name__)


class VectorEmbedding:
    def __init__(self):
        pass

    async def aad_documents(
        self, vector_store: VectorStore, vector_dir_path: Path, docs: List[Document]
    )->List[str]:
        if len(docs) == 0 or not docs:
            raise ValueError(f"Docs must be atleast one. Passed empty")
        docs_ids = await vector_store.aadd_documents(docs)
        logger.info(
            "Successfully_added %s docs into_vectorestore_path= %s",
            len(docs),
            vector_dir_path,
        )
        vector_store.save_local(vector_dir_path)
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