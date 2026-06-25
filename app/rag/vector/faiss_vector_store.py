from functools import lru_cache
from pathlib import Path
import faiss
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.embeddings import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import os



class FaissStore:
    def load_vector_store(
        self,
        vector_dir_path: Path,
        embedding_client: Embeddings,
    ) -> VectorStore:
        """Load existing vector Store"""

        vector_store = FAISS.load_local(
            vector_dir_path,
            embedding_client,
            allow_dangerous_deserialization=True,
        )

        return vector_store

    def create_vector_store(self, vector_dir_path, embedding_client, embedding_len):

        os.makedirs(vector_dir_path, exist_ok=True)
        index = faiss.IndexFlatL2(embedding_len)
        vector_store = FAISS(
            embedding_function=embedding_client,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.save_local(vector_dir_path)

        return vector_store

@lru_cache(maxsize=1)
def get_module():
    return FaissStore()


faiss_store = get_module()
