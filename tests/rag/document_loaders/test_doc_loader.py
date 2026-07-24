import pytest
from app.rag.document_loaders.doc_loader import DocumentLoader, get_recursive_splitter
from app.rag.document_loaders import doc_loader
from unittest.mock import AsyncMock,Mock
from langchain_core.documents.base import Document


def test_valueerror_on_documentloader_chunk_size():
    with pytest.raises(ValueError):
        DocumentLoader(chunk_size=99)


def test_valueerror_on_documentloader_chunk_overlap():
    with pytest.raises(ValueError):
        DocumentLoader(chunk_size=600, chunk_overlap=700)


def test_document_loader_attributes_default():
    obj = DocumentLoader()
    assert obj.chunk_size == 1000
    assert obj.chunk_overlap == 200


def test_document_loader_attributes():
    obj = DocumentLoader(2000, 500)
    assert obj.chunk_size == 2000
    assert obj.chunk_overlap == 500


def test_validate_file_with_file_not_exist(tmp_path):
    obj = DocumentLoader()
    file = tmp_path / "hello.pdf"
    with pytest.raises(FileNotFoundError):
        obj._validate_file(file)


def test_validate_file_with_unsupport_file(tmp_path):
    obj = DocumentLoader()
    file = tmp_path / "hello.txt"
    file.write_text("hello world")  # Creates the file with the given text
    with pytest.raises(ValueError):
        obj._validate_file(file)


def test_validate_file_with_supported_file_and_existed_file(tmp_path):
    obj = DocumentLoader()
    file = tmp_path / "hello.pdf"
    file.write_text("hello World")
    obj._validate_file(
        file
    )  # why we don't assertion here, because the test method is to test if we pass all correct arguments is it executes correctly or raise an exception.If we any exception raise then test automatically failed.


def test_get_character_splitter_with_same_parms():
    chunk_size, chunk_overlap = 1000, 200
    obj1 = get_recursive_splitter(chunk_size, chunk_overlap)
    obj2 = get_recursive_splitter(chunk_size, chunk_overlap)
    assert obj1 is obj2


def test_get_charackter_splitter_with_differ_params():
    obj1 = get_recursive_splitter(1000, 200)
    obj2 = get_recursive_splitter(1200, 300)
    assert obj1 is not obj2


async def test_process_pdf(monkeypatch, tmp_path):
    file = tmp_path / "hello.pdf"
    dummy_docs = [Document(page_content="This is a large content")]
    
    fake_loader_instance = Mock()
    fake_loader_instance.aload = AsyncMock(return_value=dummy_docs)
    # Since PyMuPDFLoader(...) is a constructor call that returns an instance, and .aload() is called on that instance, you always need two conceptual layers: one for "the class when called," one for "the instance it produces."
    fake_pymupdf_loader_class = Mock(return_value=fake_loader_instance)
    monkeypatch.setattr(doc_loader, "PyMuPDFLoader", fake_pymupdf_loader_class)
    obj = DocumentLoader()
    
    await obj._process_pdf(file)

async def test_process_document_to_check_pdf(tmp_path,monkeypatch):
    file=tmp_path/"hello.pdf"
    file.write_text("some large content")
    # we mock process_pdf method because we already test that method.
    # why we are here mock the DocuemntLoader obj instead of directly to the class, because of some complexity required to the class method around passing the self object to the class.
    dummy_docs = [Document(page_content="This is a large content")]
    obj=DocumentLoader()
    monkeypatch.setattr(obj, "_process_pdf", AsyncMock(return_value=dummy_docs))
    result=await obj.process_document(file)
    assert result == dummy_docs
    
async def test_process_document_to_check_unsupported_formate(tmp_path):
    obj=DocumentLoader()
    file=tmp_path/"hello.txt"
    file.write_text("hello from text file")
    with pytest.raises(ValueError):
        await obj.process_document(tmp_path/"hello.txt")