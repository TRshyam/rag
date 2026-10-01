import json
from pathlib import Path
import argparse
from qdrant_client import QdrantClient
from src.embedding.embedder import Embedder
from src.embedding.models import AVAILABLE_MODELS

_PROCESSED_CACHE = {}

def _get_fallback_chunk_metadata(collection_name: str, paper_id: str, chunk_id: str, processed_base_dir: str = "data/processed") -> dict:
    """Retrieve chunk text and page from processed JSON files if not stored in Qdrant payload."""
    strategies = ["semantic", "recursive", "markdown", "fixed"]
    strategy = None
    for s in strategies:
        if collection_name.endswith(f"_{s}"):
            strategy = s
            break
            
    candidate_strategies = [strategy] if strategy else strategies
    for strat in candidate_strategies:
        cache_key = (strat, paper_id)
        if cache_key not in _PROCESSED_CACHE:
            processed_file = Path(processed_base_dir) / strat / f"{paper_id}.json"
            if processed_file.exists():
                try:
                    with open(processed_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    chunks = {c.get("chunk_id"): c for c in data.get("chunks", [])}
                    _PROCESSED_CACHE[cache_key] = chunks
                except Exception:
                    _PROCESSED_CACHE[cache_key] = {}
            else:
                _PROCESSED_CACHE[cache_key] = {}
                
        chunks = _PROCESSED_CACHE.get(cache_key, {})
        if chunk_id in chunks:
            return chunks[chunk_id]
            
    return {}

def query_qdrant(query_text: str, model_key: str, collection_name: str, top_k: int = 5, qdrant_url: str = "http://localhost:6333"):
    # Initialize the embedder
    model_id = AVAILABLE_MODELS.get(model_key)
    if not model_id:
        print(f"Model {model_key} not found. Available models: {list(AVAILABLE_MODELS.keys())}")
        return
        
    print(f"Using model: {model_key} ({model_id})")
    embedder = Embedder(model_id=model_id)
    
    # Generate query embedding
    print(f"Generating embedding for query: '{query_text}'...")
    query_vector = embedder.embed_query(query_text)
    
    # Connect to Qdrant
    print(f"Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url)
    
    # Check if collection exists
    if not client.collection_exists(collection_name=collection_name):
        print(f"Collection {collection_name} does not exist.")
        return
        
    # Search
    print(f"Searching for top {top_k} results in collection {collection_name}...")
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k
        )
        search_result = response.points
    else:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
    
    # Display results
    print("\nSearch Results:")
    for i, result in enumerate(search_result, 1):
        payload = result.payload or {}
        paper = payload.get("paper") or payload.get("paper_id") or "Unknown"
        page = payload.get("page") if payload.get("page") is not None else payload.get("page_number")
        section = payload.get("section") or payload.get("section_name")
        chunk_text = payload.get("text") or payload.get("chunk")
        chunk_id = payload.get("chunk_id")
        
        # If text/page was not stored in Qdrant payload during ingestion, fall back to processed data
        if (chunk_text is None or page is None) and chunk_id:
            meta = _get_fallback_chunk_metadata(collection_name, paper, chunk_id)
            if meta:
                if chunk_text is None:
                    chunk_text = meta.get("text")
                if page is None:
                    page = meta.get("page")
                if section is None:
                    section = meta.get("section")
        
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   Paper: {paper}")
        if page is not None:
            print(f"   Page: {page}")
        if section:
            print(f"   Section: {section}")
        if chunk_text:
            print(f'   Chunk: "{chunk_text}"')
        elif chunk_id:
            print(f"   Chunk ID: {chunk_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="The query text")
    parser.add_argument("--model", type=str, default="Qwen3-Embedding-0.6B", help="Model key from AVAILABLE_MODELS")
    parser.add_argument("--collection", type=str, default="Qwen3_Embedding_0_6B_semantic", help="Qdrant collection name")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    parser.add_argument("--qdrant-url", type=str, default="http://localhost:6333", help="Qdrant URL")
    
    args = parser.parse_args()
    query_qdrant(args.query, args.model, args.collection, args.top_k, args.qdrant_url)
