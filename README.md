# Install and Run the AI Microservice  
Follow these steps to install and run the FastAPI-based AI microservice.

## 1. Install Dependencies  
Run the following command to install all required packages:  
```powershell
py -m pip install fastapi uvicorn sentence-transformers faiss-cpu numpy transformers torch slowapi python-dotenv
```

## 2. Start the FastAPI Server  
Navigate to your project directory:  
```powershell
cd ai-microservice
```
Then, run the server using Uvicorn:  
```powershell
py -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- `--host 0.0.0.0` makes it accessible over the network.  
- `--port 8000` runs the API on port 8000.  
- `--reload` enables hot reloading during development.  

## 3. Access the API  
Once the server is running, open a browser and go to:  
- Swagger UI (API Documentation): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)  
- Redoc API Documentation: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)  

## 4. Test the API  
Using `curl`:  
```powershell
curl -X 'POST' 'http://127.0.0.1:8000/ask' `
     -H 'Content-Type: application/json' `
     -H 'X-API-Key: your-secret-key' `
     -d '{"qid": "Q1"}'
```
Using Python:  
```python
import requests

url = "http://127.0.0.1:8000/ask"
headers = {"X-API-Key": "your-secret-key"}
data = {"qid": "Q1"}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

## 5. Run FastAPI with Docker (Optional)  
If you prefer Docker, build and run the container:  
```powershell
docker build -t ai-microservice .
docker run -p 8000:8000 -e API_KEY="your-secret-key" ai-microservice
```

## TODO

```python
import os
import json
import numpy as np
import faiss
import time
import logging
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY", "default-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key")

# Configure rate limiter
limiter = Limiter(key_func=get_remote_address)

# File Paths
DOC_PATH = "/mnt/data/medicare_comparison.md"
QUERY_PATH = "/mnt/data/queries.json"
LOG_FILE = "app.log"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# Load Document and Queries Once
with open(DOC_PATH, "r", encoding="utf-8") as file:
    DOCUMENT_TEXT = file.read()
with open(QUERY_PATH, "r", encoding="utf-8") as file:
    QUERIES = json.load(file)

# Initialize FastAPI with Swagger UI & Redoc
app = FastAPI(
    title="Medicare RAG AI API",
    version="1.0",
    description="An AI-powered API for Medicare-related question answering using RAG (Retrieval-Augmented Generation).",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # Redoc UI
)

# Add rate limiting middleware
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Middleware for logging request and response data
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_body = await request.body()
        logging.info(f"Request: {request.method} {request.url} - Body: {request_body}")
        response = await call_next(request)
        process_time = time.time() - start_time
        logging.info(f"Response: {response.status_code} - Time taken: {process_time:.2f}s")
        return response

app.add_middleware(LoggingMiddleware)

# CORS Middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Logger:
    """Logs execution time for key steps."""
    @staticmethod
    def log(message, start_time):
        time_spent = time.time() - start_time
        log_message = f"{message} - Time spent: {time_spent:.2f} seconds"
        print(log_message)
        logging.info(log_message)

class DocumentProcessor:
    def __init__(self, text, max_tokens=80):
        self.text = text
        self.max_tokens = max_tokens
        self.chunks = self.chunk_text()

    def chunk_text(self):
        sentences = self.text.split("\n")
        chunks, chunk = [], ""
        for sentence in sentences:
            if len(chunk.split()) + len(sentence.split()) <= self.max_tokens:
                chunk += sentence + " "
            else:
                chunks.append(chunk.strip())
                chunk = sentence + " "
        if chunk:
            chunks.append(chunk.strip())
        return chunks

class QueryProcessor:
    def __init__(self, queries):
        self.queries = queries

    def get_query(self, qid):
        for query in self.queries:
            if query["id"].upper() == qid.upper():
                return query["text"]
        return None

class EmbeddingRetriever:
    def __init__(self, document_chunks):
        start_time = time.time()
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.chunks = document_chunks
        self.embeddings = np.array(self.model.encode(self.chunks))
        self.index = self.create_faiss_index()
        Logger.log("Embedding generation & FAISS indexing completed", start_time)

    def create_faiss_index(self):
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.embeddings)
        return index

    def retrieve_context(self, query, top_k=2):
        start_time = time.time()
        query_embedding = np.array(self.model.encode([query])).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        Logger.log("Context retrieval completed", start_time)
        return [self.chunks[i] for i in indices[0] if i >= 0], indices[0].tolist()

class AnswerGenerator:
    def __init__(self):
        self.llm = pipeline("text-generation", model="google/flan-t5-xl")

    def generate_answer(self, query, context):
        start_time = time.time()
        if not context:
            return "No relevant information found in the document."
        prompt = f"Answer the query based only on the given context.\n\nContext:\n{context}\n\nQuery: {query}\nAnswer:"
        response = self.llm(prompt, max_length=100, do_sample=True)[0]["generated_text"].strip()
        Logger.log("Answer generation completed", start_time)
        return response

# Load models once
doc_processor = DocumentProcessor(DOCUMENT_TEXT)
query_processor = QueryProcessor(QUERIES)
retriever = EmbeddingRetriever(doc_processor.chunks)
generator = AnswerGenerator()

class QueryRequest(BaseModel):
    qid: str

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@app.post("/ask", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def ask_question(request: QueryRequest):
    query_text = query_processor.get_query(request.qid)
    if not query_text:
        raise HTTPException(status_code=404, detail="Query ID not found")
    context, source_indices = retriever.retrieve_context(query_text)
    answer = generator.generate_answer(query_text, "\n".join(context))
    return {
        "query_id": request.qid,
        "query_text": query_text,
        "answer": answer,
        "source_chunks": source_indices,
        "source_text": context
    }

logging.info("FastAPI service with request/response logging and rate limiting initialized successfully")
```

## Additional Thoughts

### **FastAPI Production Checklist for AI & Gen AI**

---

#### **1. Code Structure & Best Practices**  
- Organize the project with MVC (Model-View-Controller) or Service Layers  
- Use Pydantic for request validation & type safety  
- Implement dependency injection for database and services  
- Write unit tests using `pytest` and ensure test coverage  

---

#### **2. AI Model & Optimization**  
- Use ONNX or TorchScript for model optimization  
- Load AI models asynchronously and keep them in memory efficiently  
- Implement batching for inference if applicable  
- Use vector databases like FAISS, Milvus, or Weaviate for Gen AI  

---

#### **3. Performance & Scalability**  
- Use Gunicorn with `uvicorn.workers.UvicornWorker` for multi-process execution  
- Enable FastAPI caching using `fastapi-cache2` or Redis  
- Set up asynchronous database queries using `SQLAlchemy Async` or `Tortoise ORM`  
- Enable pagination for large data responses  
- Optimize AI model inference using TensorRT, ONNX, or OpenVINO  

---

#### **4. Security Hardening**  
- Use JWT authentication with `fastapi-users`  
- Validate request payloads with Pydantic models  
- Set up CORS policies to restrict unauthorized access  
- Implement rate limiting using `slowapi` to prevent abuse  
- Sanitize user inputs to prevent SQL injection & XSS  
- Use HTTPS (TLS) with proper certificates (Let’s Encrypt or Cloudflare)  

---

#### **5. Logging & Monitoring**  
- Use structured logging with `loguru`  
- Enable request tracing with `OpenTelemetry`  
- Monitor API health using Prometheus & Grafana  
- Set up Sentry for error tracking  
- Use FastAPI middleware to log request/response data  

---

#### **6. Deployment & Infrastructure**  
- Use Docker with a lightweight image (`python:3.11-slim`)  
- Create a Docker Compose setup with Redis, Postgres, etc.  
- Use Kubernetes or Docker Swarm for scaling  
- Optimize startup with `gunicorn -k uvicorn.workers.UvicornWorker`  
- Store secrets in AWS Secrets Manager, Vault, or `.env` files  

---

#### **7. CI/CD & Versioning**  
- Use GitHub Actions or GitLab CI/CD for automated deployment  
- Implement automatic API versioning using `versioned-fastapi`  
- Automate unit tests, linting, and security scans in CI/CD  
- Deploy with Zero Downtime using rolling updates or blue-green deployment  

---

#### **8. Database & Storage**  
- Use PostgreSQL or MongoDB for structured/unstructured data  
- Implement caching layers using Redis for frequent queries  
- Store AI models & vector embeddings in S3, MinIO, or a dedicated DB  
- Optimize database queries with indexes & connection pooling  

---

#### **9. API Documentation & Observability**  
- Use Swagger UI & Redoc (FastAPI provides them by default)  
- Add custom OpenAPI schemas for better documentation  
- Integrate OpenTelemetry for distributed tracing  
- Use GraphQL (strawberry-graphql) if flexibility is needed  

---

### **Bonus for Gen AI APIs**  
- Async WebSockets for real-time AI responses  
- Token-based request pricing (if monetizing AI API)  
- LangChain for AI-powered workflows  
- Hugging Face Model Inference integration if required  

---

### **Final Steps**  
- Test APIs using Postman or `locust.io` for load testing  
- Deploy on AWS, GCP, Azure, or DigitalOcean  
- Use Terraform or Ansible for Infrastructure as Code (IaC)  
- Set up auto-scaling for high-traffic AI requests  
