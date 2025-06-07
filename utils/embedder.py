from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import os

def create_faiss_index(all_docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for doc in all_docs:
        for chunk in splitter.split_text(doc):
            chunks.append(Document(page_content=chunk))

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local("embeddings/faiss_index")
    with open("embeddings/docstore.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return vectorstore
