import streamlit as st
from utils.Retrieval import load_document, splitter, create_vectorstore
from io import BytesIO
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from utils.Generation import multi_query_retriever, reciprocal_rank_fusion, generation_step, model

pdf_file = st.file_uploader("Upload YOUR Document", type=("pdf"))

if pdf_file is not None:
    pdf_bytes = BytesIO(pdf_file.read())
    document = load_document(pdf_bytes)
    split = splitter(document)
    filename = str(pdf_file.name)
    persist_dir = create_vectorstore(split,filename)

query = st.text_input("Ask any query about doc")

if st.button("ASK"):
    llm = model()
    embedding = GPT4AllEmbeddings()
    db = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = db.as_retriever()

    retrieved_doc = multi_query_retriever(llm, retriever, query)
    ranking = reciprocal_rank_fusion(retrieved_doc)
    result = generation_step(llm, ranking, query)
    st.write(result)