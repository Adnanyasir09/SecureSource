from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import os

def create_vector_store(repo_path):
    # Load documents
    loader = DirectoryLoader(repo_path, glob="**/*.py", loader_cls=TextLoader)
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Use FAISS instead of Chroma
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store, docs
