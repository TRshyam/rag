# src/chunking/chunking.py

from src.chunking.fixed_chunk import fixed_chunking 
from src.chunking.recursive_chunk import recursive_chunking
from src.chunking.markdown_chunk import markdown_chunking
from src.chunking.semantic_chunk import semantic_chunking

from langchain_ollama import OllamaEmbeddings

def create_chunks(
    text: str,
    strategy: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    embedding_model: str ="qwen3-embedding"
):
    """
    Main chunking controller.

    Available strategies:

        fixed
        recursive
        markdown
        semantic
    """

    strategy = strategy.lower().strip()

    # --------------------------------
    # Fixed
    # --------------------------------

    if strategy == "fixed":

        return fixed_chunking(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # --------------------------------
    # Recursive
    # --------------------------------

    elif strategy == "recursive":

        return recursive_chunking(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    # --------------------------------
    # Markdown
    # --------------------------------

    elif strategy == "markdown":

        return markdown_chunking(text)

    # --------------------------------
    # Semantic
    # --------------------------------

    elif strategy == "semantic":
        embeddings = OllamaEmbeddings(model=embedding_model)
        return semantic_chunking(text=text, embedding_model=embeddings)

    # --------------------------------
    # Invalid strategy
    # --------------------------------

    else:

        raise ValueError(
            f"Unknown strategy: {strategy}. "
            f"Use fixed, recursive, markdown, or semantic."
        )