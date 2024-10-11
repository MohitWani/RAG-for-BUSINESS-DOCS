from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
import os


def load_document(file_path):
    document = PyPDFLoader(file_path).load()
    print("Document Loaded successfully...")
    return document

def splitter(document):
    doc_split = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap = 50,
    )

    split_docs = doc_split.split_documents(documents=document)

    print("Chunks are created successfully...")
    return split_docs


def create_vectorstore(split_docs):

    embedding = GPT4AllEmbeddings()
    
    persist_directory = 'db'
    vectordb = Chroma.from_documents(split_docs, embedding, persist_directory=persist_directory)

    os.makedirs(persist_directory = 'db', exist_ok=True)

    vectordb.persist()
    return "Vector database is Saved."