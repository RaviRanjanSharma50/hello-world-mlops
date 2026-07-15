from dotenv import load_dotenv
load_dotenv()

from FlagEmbedding import BGEM3FlagModel
from langchain_pymupdf import PyMuPDFLoader   # updated import (no deprecation warning)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

# ===== Load and Split Documents =====
loader = PyMuPDFLoader(r"D:\Projects\Conversational_RAG\Docs\Gen AI.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = splitter.split_documents(docs)

# Add metadata
for i, split in enumerate(splits):
    split.metadata["index"] = i
    split.metadata["filename"] = "Gen AI.pdf"
    split.metadata["page_number"] = split.metadata.get("page", None)

# ===== Embeddings + Chroma =====
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_db = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    collection_name="gen_ai_collection",
    persist_directory="./chroma_db"
)

# ===== BM25 Index =====
corpus = [doc.page_content for doc in splits]
bm25 = BM25Okapi([d.split() for d in corpus])

# ===== Keyword Search =====
def keyword_search(query: str, k: int = 3):
    return [doc.page_content for doc in splits if query.lower() in doc.page_content.lower()][:k]

# ===== Reciprocal Rank Fusion =====
def reciprocal_rank_fusion(query: str, k: int = 5, fusion_k: int = 60):
    bm25_results = bm25.get_top_n(query.split(), corpus, n=k)
    vector_results = vector_db.similarity_search(query, k=k)
    keyword_results = keyword_search(query, k=k)

    ranked_lists = {
        "bm25": bm25_results,
        "vector": [doc.page_content for doc in vector_results],
        "keyword": keyword_results
    }

    scores = {}
    for method, docs in ranked_lists.items():
        for rank, doc in enumerate(docs):
            scores[doc] = scores.get(doc, 0) + 1 / (fusion_k + rank + 1)

    fused_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in fused_results[:k]]

# ===== Local Reranker (BGE) =====
reranker = BGEM3FlagModel("BAAI/bge-reranker-large", use_fp16=True)

def rerank_with_bge(query: str, docs: list, top_n: int = 5):
    pairs = [(query, d) for d in docs]
    scores = reranker.compute_score(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]

# ===== Hybrid Search Pipeline =====
def hybrid_search_pipeline(query: str, top_n: int = 5):
    fused_docs = reciprocal_rank_fusion(query, k=top_n)
    return rerank_with_bge(query, fused_docs, top_n=top_n)
