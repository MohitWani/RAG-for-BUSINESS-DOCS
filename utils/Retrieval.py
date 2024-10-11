from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
import os
import shutil
import tempfile


def load_document(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        loader = PyPDFLoader(tmp_file.name)
        document = loader.load()
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

def delete_vector_store_if_exists(persist_directory):
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)  # Remove the directory and all its contents
        print(f"Existing vector store at '{persist_directory}' deleted successfully.")


def create_vectorstore(split_docs, filename):
    embedding = GPT4AllEmbeddings()

    location = 'D:\my_projects\{sample set}\database'
    persist_directory = os.path.join(location,filename)
    
    if os.path.exists(persist_directory):
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        vectordb.add_documents(split_docs)  # Add new vectors to the existing store
    else:
        vectordb = Chroma.from_documents(split_docs, embedding, persist_directory=persist_directory)

    os.makedirs(persist_directory, exist_ok=True)

    vectordb.persist()  
    return persist_directory