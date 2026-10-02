import json
import sys
import re
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient
from langchain_ollama import ChatOllama

from src.embedding.embedder import Embedder
from src.embedding.models import AVAILABLE_MODELS

# In-memory cache for fallback chunk metadata from data/processed
_PROCESSED_CACHE: Dict[tuple, dict] = {}


def clean_golden_answer(text: str) -> str:
    """
    Remove citation markers like [cite: 2], [cite: 1, 2], etc. from golden answers.
    Keeps the answer clean while preserving actual text.
    """
    cleaned = re.sub(r"\[cite:\s*[^\]]+\]", "", text, flags=re.IGNORECASE)
    # Clean any double spaces left behind
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


@dataclass
class GoldenQuestion:
    """Represents a validated benchmark question with ground truth metadata."""
    id: str
    question: str
    expected_answer: str
    source_document: str
    source_file: str
    page: int
    question_type: str
    evidence: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenQuestion":
        raw_answer = data.get("answer", "")
        cleaned_answer = clean_golden_answer(raw_answer)
        return cls(
            id=data.get("id", ""),
            question=data.get("question", ""),
            expected_answer=cleaned_answer,
            source_document=data.get("source", ""),
            source_file=data.get("source_file", ""),
            page=int(data.get("page", 0)),
            question_type=data.get("type", "General"),
            evidence=data.get("evidence"),
        )


def load_golden_dataset(file_path: str | Path) -> List[GoldenQuestion]:
    """Load and parse the golden questions JSON dataset."""
    path = Path(file_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    if not isinstance(raw_items, list):
        raise ValueError(f"Expected a JSON list of items, got {type(raw_items).__name__}")

    return [GoldenQuestion.from_dict(item) for item in raw_items]


def _get_fallback_chunk_metadata(
    collection_name: str,
    paper_id: str,
    chunk_id: str,
    processed_base_dir: Path = PROJECT_ROOT / "data" / "processed"
) -> dict:
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
            processed_file = processed_base_dir / strat / f"{paper_id}.json"
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


class QdrantRAGPipeline:
    """
    RAG Pipeline:
    golden_questions.json -> Qwen3-Embedding-0.6B -> Qdrant (Top-5) -> Prompt -> Qwen3.5:4B -> Generated Answer
    """

    def __init__(
        self,
        embed_model_key: str = "Qwen3-Embedding-0.6B",
        llm_model_name: str = "qwen3.5:4b",
        collection_name: str = "Qwen3_Embedding_0_6B_semantic",
        qdrant_url: str = "http://localhost:6333",
        top_k: int = 5,
        temperature: float = 0.0,
    ):
        self.embed_model_key = embed_model_key
        self.embed_model_id = AVAILABLE_MODELS.get(embed_model_key)
        if not self.embed_model_id:
            raise ValueError(f"Unknown embedding model: {embed_model_key}")

        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.top_k = top_k
        self.llm_model_name = llm_model_name

        print(f"Initializing Embedder: {embed_model_key} ({self.embed_model_id})...")
        self.embedder = Embedder(model_id=self.embed_model_id)

        print(f"Connecting to Qdrant at {qdrant_url} (collection: {collection_name})...")
        self.qdrant_client = QdrantClient(url=qdrant_url)
        if not self.qdrant_client.collection_exists(collection_name=collection_name):
            raise RuntimeError(f"Qdrant collection '{collection_name}' does not exist.")

        print(f"Initializing LLM: {llm_model_name} via Ollama...")
        self.llm = ChatOllama(
            model=llm_model_name,
            temperature=temperature,
            reasoning=False,
            num_ctx=8192,
            num_predict=1024
)

    def retrieve_top_k(self, query: str) -> List[Dict[str, Any]]:
        """Generate query vector using Qwen3-Embedding-0.6B and retrieve Top-K chunks from Qdrant."""
        query_vector = self.embedder.embed_query(query)

        if hasattr(self.qdrant_client, "query_points"):
            response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=self.top_k,
            )
            points = response.points
        else:
            points = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=self.top_k,
            )

        retrieved_chunks = []
        for idx, pt in enumerate(points, 1):
            payload = pt.payload or {}
            paper = payload.get("paper") or payload.get("paper_id") or "Unknown"
            page = payload.get("page") if payload.get("page") is not None else payload.get("page_number")
            text = payload.get("text") or payload.get("chunk")
            chunk_id = payload.get("chunk_id")

            # Fallback if text is not in payload
            if (text is None or page is None) and chunk_id:
                fallback = _get_fallback_chunk_metadata(self.collection_name, paper, chunk_id)
                if fallback:
                    if text is None:
                        text = fallback.get("text")
                    if page is None:
                        page = fallback.get("page")

            retrieved_chunks.append({
                "rank": idx,
                "score": round(float(pt.score), 4),
                "text": text or "",
                "chunk_id": chunk_id,
                "source_file": payload.get("source_file") or (f"{paper}.pdf" if not paper.endswith(".pdf") else paper),
                "page": page,
            })

        return retrieved_chunks

    def build_prompt(self, chunks: List[Dict[str, Any]], question: str) -> str:
        """Construct context and prompt for Qwen3.5:4B."""
        context_parts = []
        for c in chunks:
            page_info = f" (Page {c['page']})" if c.get("page") is not None else ""
            context_parts.append(f"--- Chunk {c['rank']}{page_info} ---\n{c['text']}")

        context_str = "\n\n".join(context_parts)
        return (
            f"Context:\n{context_str}\n\n"
            f"Question:\n{question}\n\n"
            "Answer using only the context."
        )

    def generate_answer(self, prompt: str) -> str:
        """Send prompt to Qwen3.5:4B and return cleaned answer."""
        response = self.llm.invoke(prompt)
        raw_content = response.content if hasattr(response, "content") else str(response)
        cleaned = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL)
        return cleaned.strip()

    def process_question(self, gq: GoldenQuestion) -> Dict[str, Any]:
        """Run full RAG pipeline for one golden question."""
        # 1. Retrieve Top-5 chunks
        retrieved_chunks = self.retrieve_top_k(gq.question)

        # 2. Build prompt
        prompt = self.build_prompt(retrieved_chunks, gq.question)

        # 3. Generate answer using Qwen3.5:4B
        generated_answer = self.generate_answer(prompt)

        # 4. Compute retrieval match against ground truth
        exact_rank = None
        near_rank = None
        doc_rank = None
        for c in retrieved_chunks:
            c_page = c.get("page")
            c_file = c.get("source_file")
            is_doc = (c_file == gq.source_file)
            if is_doc:
                if doc_rank is None:
                    doc_rank = c.get("rank")
                if c_page == gq.page and exact_rank is None:
                    exact_rank = c.get("rank")
                elif c_page is not None and gq.page is not None and abs(c_page - gq.page) <= 1 and near_rank is None:
                    near_rank = c.get("rank")

        if exact_rank is not None:
            retrieval_match = f"Exact Page (Rank {exact_rank})"
        elif near_rank is not None:
            retrieval_match = f"Adjacent Page (Rank {near_rank})"
        elif doc_rank is not None:
            retrieval_match = f"Doc Only (Rank {doc_rank})"
        else:
            retrieval_match = "Doc Miss"

        return {
            "question_id": gq.id,
            "question": gq.question,
            "golden_answer": gq.expected_answer,
            "retrieved_chunks": retrieved_chunks,
            "generated_answer": generated_answer,
            "retrieval_match": retrieval_match,
        }


def display_result(result: Dict[str, Any]):
    """Print readable summary to console."""
    print("=" * 80)
    print(f"QUESTION ID:      {result['question_id']}")
    print(f"Question:         {result['question']}")
    print(f"Golden Answer:    {result['golden_answer']}")
    print(f"Generated Answer: {result['generated_answer']}")
    print("-" * 80)
    print(f"Retrieved Chunks ({len(result['retrieved_chunks'])}):")
    for c in result["retrieved_chunks"]:
        snippet = (c["text"][:110] + "...") if len(c["text"]) > 110 else c["text"]
        print(f"  Rank {c['rank']} (Score: {c['score']:.4f}) [Page {c['page']}]: \"{snippet}\"")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Pipeline: golden_questions.json -> Qwen3-Embedding-0.6B -> Qdrant -> Qwen3.5:4B -> rag_evaluation_results.json"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="evaluation/gemini-code-1790832398351.json",
        help="Path to golden questions JSON dataset",
    )
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Target question ID (e.g. blip2_001). If omitted and --all is not set, defaults to blip2_001.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Loop through all questions in the golden dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate when running multiple",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="Qwen3_Embedding_0_6B_semantic",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="Qwen3-Embedding-0.6B",
        help="Embedding model key from AVAILABLE_MODELS",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="qwen3.5:4b",
        help="Ollama LLM model name for generation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/rag_evaluation_results.json",
        help="Path to save output JSON",
    )
    args = parser.parse_args()

    # Load dataset
    questions = load_golden_dataset(args.dataset)
    print(f"Loaded {len(questions)} golden questions from {args.dataset}")

    # Determine which questions to run
    if args.all:
        target_questions = questions
        if args.limit:
            target_questions = target_questions[: args.limit]
    elif args.question_id:
        target_questions = [q for q in questions if q.id == args.question_id]
        if not target_questions:
            print(f"Error: Question ID '{args.question_id}' not found.")
            sys.exit(1)
    else:
        # Default first target: blip2_001
        default_id = "blip2_001"
        target_questions = [q for q in questions if q.id == default_id]
        if not target_questions:
            target_questions = [questions[0]]
        print(f"No target specified. Defaulting to single question target: {target_questions[0].id}")

    # Initialize RAG Pipeline
    pipeline = QdrantRAGPipeline(
        embed_model_key=args.embed_model,
        llm_model_name=args.llm_model,
        collection_name=args.collection,
        top_k=args.top_k,
    )

    # Output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Execute
    results = []
    print(f"\nRunning RAG pipeline for {len(target_questions)} question(s)...")
    for idx, gq in enumerate(target_questions, 1):
        print(f"\n[{idx}/{len(target_questions)}] Processing {gq.id}...")
        res = pipeline.process_question(gq)
        display_result(res)
        results.append(res)

        # Progressive save to ensure real-time persistence
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nCompleted! Saved all {len(results)} result(s) to: {output_path}")


if __name__ == "__main__":
    main()
