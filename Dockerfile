FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn sentence-transformers faiss-cpu numpy transformers torch python-dotenv
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
