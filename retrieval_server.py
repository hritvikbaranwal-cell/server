import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

# ----------------------------
# Load API Keys
# ----------------------------

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ----------------------------
# Configuration
# ----------------------------

INDEX_NAME = "wiki-rag"
EMBEDDING_MODEL = "jina-embeddings-v3"
DIMENSION = 1024

# ----------------------------
# Clients
# ----------------------------

jina_client = OpenAI(
    api_key=JINA_API_KEY,
    base_url="https://api.jina.ai/v1"
)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ----------------------------
# FastAPI App
# ----------------------------

app = FastAPI(
    title="RAG Retrieval Server"
)

# ----------------------------
# Request Model
# ----------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 20

# ----------------------------
# Health Endpoint
# ----------------------------

@app.get("/")
def root():
    return {
        "message": "Retrieval Server Running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# ----------------------------
# Retrieval Endpoint
# ----------------------------

@app.post("/retrieve")
def retrieve(req: QueryRequest):

    try:

        # Step 1 : Create query embedding

        embedding_response = jina_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[req.query],
            dimensions=DIMENSION
        )

        query_vector = embedding_response.data[0].embedding

        # Step 2 : Search Pinecone

        results = index.query(
            vector=query_vector,
            top_k=req.top_k,
            include_metadata=True,
            include_values=False
        )

        retrieved_chunks = []

        for match in results.matches:

            metadata = match.metadata or {}

            retrieved_chunks.append({

                "score": float(match.score),

                "doc_id": metadata.get("doc_id"),

                "chunk_index": metadata.get("chunk_index"),

                "text": metadata.get("text")

            })

        return {

            "query": req.query,

            "total_chunks": len(retrieved_chunks),

            "chunks": retrieved_chunks

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )