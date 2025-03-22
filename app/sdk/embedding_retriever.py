import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.sdk.logger import Logger

class EmbeddingRetriever:
    def __init__(self, document_chunks):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.chunks = document_chunks
        self.embeddings = np.array(self.model.encode(self.chunks))
        self.index = self.create_faiss_index()
        Logger.log("Embedding generation & FAISS indexing completed")

    def create_faiss_index(self):
        dimension = self.embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(self.embeddings)
        return index

    def retrieve_context(self, query, top_k=2):
        query_embedding = np.array(self.model.encode([query])).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)
        Logger.log("Context retrieval completed")
        return [self.chunks[i] for i in indices[0] if i >= 0], indices[0].tolist()
