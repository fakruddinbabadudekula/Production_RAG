from langchain_community.vectorstores import FAISS
import faiss
import os
from pathlib import Path
from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents.base import Document
import logging
from functools import lru_cache
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache()
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


class Retriever:
    def __init__(self, vector_dir: str):
        self.vector_dir_path = Path(vector_dir)
        self.embeddings = load_embeddings()
        if (self.vector_dir_path / "index.faiss").exists():
            logger.info(f"Existing vectorstore found {self.vector_dir_path.name}")
            self.vector_db = FAISS.load_local(
                self.vector_dir_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
           
        else:
            logger.info(f"Creating vector dir {self.vector_dir_path}")
            os.makedirs(self.vector_dir_path, exist_ok=True)
            self.index = faiss.IndexFlatL2(
                len(self.embeddings.embed_query("hello world"))
            )
            self.vector_db = FAISS(
                embedding_function=self.embeddings,
                index=self.index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.vector_db.save_local(self.vector_dir_path)
            

        self.retriever = self.vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

    async def aadd_documents(self, docs: List[Document]):
        try:
            _=await self.vector_db.aadd_documents(docs)
            logger.info(f"Scessfully added the docs into {self.vector_dir_path}")
        except Exception as e:
            logger.error(
                f"Adding documents throws an error {str(e)} at {self.vector_dir_path}"
            )
            raise

    async def aget_top_k(self, query: str) -> List[Document]:
        try:
            top_docs =await self.retriever.ainvoke(query)
            logger.info(f"Successfully Retrived the top_k docs for {query}")
            return top_docs
        except Exception as e:
            logger.error(f"Error at searching the docs {str(e)}")
            raise
