from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
import os

class DocumentProcessor:
    def process(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        content = "\n".join([doc.page_content for doc in docs])
        return content