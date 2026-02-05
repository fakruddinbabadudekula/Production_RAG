from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from typing import List
from pathlib import Path
import logging
from functools import lru_cache
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@lru_cache()
def get_recursive_splitter(chunk_size,chunk_overlap):
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    return splitter
    
class DocumentLoader():
    
    def __init__(self,chunk_size: int=1000,chunk_overlap: int = 200):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
        self.supported_formats={'.pdf'}
        
        
    async def process_document(self,path:str)->List[Document]:
        file_path=Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        try:
            if file_path.suffix.lower() == ".pdf":
                return await self._process_pdf(file_path)

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")  
            raise   
        
        
               
    async def _process_pdf(self,file_path:Path)->List[Document]:
        logger.info(f"Started Processing document: {file_path.name}")
        pdf_loader=PyMuPDFLoader(file_path=file_path)
        data=await pdf_loader.aload()
        splitter=get_recursive_splitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap)
        docs=splitter.split_documents(data)
        logger.info(f"Processed document: {file_path.name}")
        return docs
        
        
    