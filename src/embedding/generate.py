import json
import argparse
from pathlib import Path

from src.embedding.models import AVAILABLE_MODELS
from src.embedding.embedder import Embedder

def generate_and_save_embeddings(data_dir: str = "data/processed", output_dir: str = "data/embeddings"):
    base_dir = Path(data_dir)
    out_dir = Path(output_dir)

    for model_display_name, model_id in AVAILABLE_MODELS.items():
        print(f"\n==============================================")
        print(f"Generating embeddings for {model_display_name} ({model_id})")
        print(f"==============================================")
        
        embedder = Embedder(model_id=model_id)
        
        for strategy_dir in base_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
                
            strategy = strategy_dir.name
            print(f"\nStrategy: {strategy}")
            
            strat_out_dir = out_dir / model_display_name / strategy
            strat_out_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in sorted(strategy_dir.glob("*.json")):
                paper_id = file_path.stem
                output_file = strat_out_dir / f"{paper_id}.json"
                
                if output_file.exists():
                    print(f"  Skipping {paper_id}, already processed.")
                    continue
                    
                print(f"  Processing {paper_id}...")
                
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                chunks = data.get("chunks", [])
                if not chunks:
                    print(f"  No chunks found in {paper_id}.")
                    continue
                    
                texts = [chunk.get("text", "") for chunk in chunks]
                chunk_ids = [chunk.get("chunk_id", "") for chunk in chunks]
                
                try:
                    embeddings = embedder.embed_documents(texts)
                    
                    output_data = {
                        "paper_id": paper_id,
                        "model": model_display_name,
                        "strategy": strategy,
                        "embeddings": [
                            {"chunk_id": cid, "embedding": emb}
                            for cid, emb in zip(chunk_ids, embeddings)
                        ]
                    }
                    
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False)
                except Exception as e:
                    print(f"  Error processing {paper_id} with {model_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory containing processed JSON files")
    parser.add_argument("--output-dir", type=str, default="data/embeddings", help="Directory to save embeddings")
    args = parser.parse_args()
    
    generate_and_save_embeddings(args.data_dir, args.output_dir)
