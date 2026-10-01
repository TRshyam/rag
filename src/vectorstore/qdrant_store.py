import json
import argparse
from pathlib import Path
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

def store_embeddings_in_qdrant(embeddings_dir: str = "data/embeddings", processed_dir: str = "data/processed", qdrant_url: str = "http://localhost:6333"):
    client = QdrantClient(url=qdrant_url)
    base_dir = Path(embeddings_dir)
    processed_base = Path(processed_dir)
    
    if not base_dir.exists():
        print(f"Directory {base_dir} does not exist.")
        return

    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"Processing model: {model_name}")
        
        for strategy_dir in model_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
                
            strategy = strategy_dir.name
            
            # Use model and strategy to name the collection
            # Qdrant collection names must follow a specific pattern
            collection_name = f"{model_name.replace('-', '_').replace('.', '_')}_{strategy}"
            
            print(f"  Strategy: {strategy} -> Collection: {collection_name}")
            
            # Since we don't know the vector size upfront, we'll read the first file
            files = list(strategy_dir.glob("*.json"))
            if not files:
                continue
                
            # Find vector size
            vector_size = None
            for file_path in files:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                embeddings = data.get("embeddings", [])
                if embeddings and len(embeddings) > 0:
                    vector_size = len(embeddings[0].get("embedding", []))
                    break
                    
            if not vector_size:
                print(f"    No valid embeddings found to determine size.")
                continue
                
            # Create collection if it doesn't exist
            if not client.collection_exists(collection_name=collection_name):
                print(f"    Creating collection with size {vector_size}...")
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
            
            # Upload points
            for file_path in sorted(files):
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                paper_id = data.get("paper_id", "")
                embeddings = data.get("embeddings", [])
                if not embeddings:
                    continue
                    
                # Load chunk metadata (text, page, section) from processed files
                processed_file = processed_base / strategy / f"{paper_id}.json"
                chunk_meta_map = {}
                if processed_file.exists():
                    try:
                        with open(processed_file, "r", encoding="utf-8") as pf:
                            pdata = json.load(pf)
                        for c in pdata.get("chunks", []):
                            chunk_meta_map[c.get("chunk_id")] = c
                    except Exception as e:
                        print(f"    Warning: Could not read {processed_file}: {e}")

                print(f"    Uploading {len(embeddings)} points for paper {paper_id}...")
                
                points = []
                for i, item in enumerate(embeddings):
                    chunk_id = item.get("chunk_id", f"{paper_id}_{i}")
                    embedding = item.get("embedding", [])
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
                    meta = chunk_meta_map.get(chunk_id, {})
                    
                    payload = {
                        "text": meta.get("text", ""),
                        "paper_id": paper_id,
                        "chunk_id": chunk_id,
                        "page": meta.get("page"),
                        "section": meta.get("section")
                    }
                    
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                    )
                
                # Upload in batches
                client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
    print(f"\nFinished storing embeddings in Qdrant at {qdrant_url}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", default="data/embeddings")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    args = parser.parse_args()
    
    store_embeddings_in_qdrant(args.embeddings_dir, args.processed_dir, args.qdrant_url)
