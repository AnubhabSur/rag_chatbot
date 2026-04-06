from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import time
import os

COLLECTION_NAME = "knowledge_base"
PERSISTENT_PATH= "/data/chroma_db"

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_vectorstore():
    os.makedirs(PERSISTENT_PATH, exist_ok= True)

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory= PERSISTENT_PATH
    )

def ingest_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(docs)

    vectorstore = get_vectorstore()
    for i, chunk in enumerate(chunks):
        for attempt in range(3):
            try:
                vectorstore.add_documents([chunk])
                break
            except Exception as e:
                print(f"Retry {attempt+1} for chunk {i}: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

    vectorstore.persist()
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")
    return vectorstore