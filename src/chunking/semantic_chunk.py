from langchain_experimental.text_splitter import SemanticChunker
import time


def semantic_chunking(
    text: str,
    embedding_model
):
    """
    Semantic chunking with detailed progress logging.
    """

    print("\n" + "=" * 60)
    print("STARTING SEMANTIC CHUNKING")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1: Validate text
    # ---------------------------------------------------------
    print("[1/7] Checking input text...")

    if not text or not text.strip():
        print("      ❌ Text is empty.")
        return []

    text = text.strip()

    print(f"      ✓ Text received")
    print(f"      Characters: {len(text):,}")
    print(f"      Approx words: {len(text.split()):,}")

    # ---------------------------------------------------------
    # STEP 2: Validate embedding model
    # ---------------------------------------------------------
    print("\n[2/7] Checking embedding model...")

    if embedding_model is None:
        raise ValueError(
            "embedding_model is required for semantic chunking"
        )

    print(f"      ✓ Embedding model: {type(embedding_model).__name__}")

    # ---------------------------------------------------------
    # STEP 3: Create SemanticChunker
    # ---------------------------------------------------------
    print("\n[3/7] Creating SemanticChunker...")

    start_time = time.time()

    splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95
    )

    elapsed = time.time() - start_time

    print(f"      ✓ SemanticChunker created")
    print(f"      Time: {elapsed:.2f} seconds")
    print(f"      Breakpoint threshold: 95 percentile")

    # ---------------------------------------------------------
    # STEP 4: Estimate sentence count
    # ---------------------------------------------------------
    print("\n[4/7] Preparing document...")

    # This is only an approximate count
    sentence_count = text.count(".") + text.count("?") + text.count("!")

    print(f"      Approx sentences: {sentence_count:,}")
    print("      SemanticChunker will now calculate embeddings...")
    print("      ⚠️ This may take some time for large documents.")

    # ---------------------------------------------------------
    # STEP 5: Semantic chunking
    # ---------------------------------------------------------
    print("\n[5/7] Running semantic chunking...")

    start_time = time.time()

    documents = splitter.create_documents([text])

    elapsed = time.time() - start_time

    print(f"      ✓ Semantic chunking completed")
    print(f"      Time: {elapsed:.2f} seconds")
    print(f"      Documents/chunks generated: {len(documents)}")

    # ---------------------------------------------------------
    # STEP 6: Clean chunks
    # ---------------------------------------------------------
    print("\n[6/7] Cleaning chunks...")

    chunks = []

    for i, document in enumerate(documents, start=1):

        content = document.page_content.strip()

        if content:
            chunks.append(content)

        print(
            f"      Chunk {i}/{len(documents)} "
            f"→ {len(content):,} characters"
        )

    # ---------------------------------------------------------
    # STEP 7: Summary
    # ---------------------------------------------------------
    print("\n[7/7] Semantic chunking summary")

    print(f"      ✓ Final chunks: {len(chunks)}")

    if chunks:
        avg_size = sum(len(chunk) for chunk in chunks) / len(chunks)

        print(f"      Average chunk size: {avg_size:,.0f} characters")
        print(f"      Smallest chunk: {min(len(c) for c in chunks):,} characters")
        print(f"      Largest chunk: {max(len(c) for c in chunks):,} characters")

    print("=" * 60)
    print("SEMANTIC CHUNKING COMPLETE")
    print("=" * 60)

    return chunks