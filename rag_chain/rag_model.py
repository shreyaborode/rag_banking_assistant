from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def build_rag_pipeline():
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.load_local("embeddings/faiss_index", embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-3.5-turbo")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa
