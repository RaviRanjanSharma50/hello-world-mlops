from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from backend import hybrid_search_pipeline

app = FastAPI(title="Hybrid RAG API", description="Production-ready RAG with RRF + BGE reranker")

class SearchRequest(BaseModel):
    query: str
    k: int = 5

class SearchResponse(BaseModel):
    results: List[str]

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    results = hybrid_search_pipeline(req.query, top_n=req.k)
    return SearchResponse(results=results)
