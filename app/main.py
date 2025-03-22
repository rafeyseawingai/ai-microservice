from fastapi import FastAPI
from app.sdk.document_processor import DocumentProcessor
from app.sdk.query_processor import QueryProcessor
from app.sdk.embedding_retriever import EmbeddingRetriever
from app.sdk.answer_generator import AnswerGenerator
from app.sdk.logger import Logger

app = FastAPI(title="Medicare RAG AI API", version="1.0")

@app.get("/")
def home():
    return {"message": "Medicare RAG API is running!"}
