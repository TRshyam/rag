from pathlib import Path
import json
import argparse

from src.ingestion.pdf_loader import extract_pdf
from src.ingestion.cleaner import clean_text
from src.chunking.chunking import create_chunks


def main():

    print("Starting PDF extraction...")
    # =====================================================
    # SELECT CHUNKING STRATEGY HERE
    # =====================================================

    strategy = "semantic" 

    # Available:
    # "fixed"
    # "recursive"
    # "markdown"
    # "semantic"

    print(f"Chunking strategy: {strategy}")

    pdf_path = Path("data/raw")
    output_path = Path(f"data/processed/{strategy}")

    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_path.glob("*.pdf"))

    print(f"Found {len(pdf_files)} papers")


    # =====================================================
    # PROCESS EACH PDF
    # =====================================================

    for pdf_file in pdf_files:

        print(f"\nProcessing: {pdf_file.name}")

        # -------------------------------------------------
        # 1. Extract PDF
        # -------------------------------------------------

        pages = extract_pdf(pdf_file)

        print(
            f"Extracted {len(pages)} pages "
            f"from {pdf_file.name}"
        )

        # -------------------------------------------------
        # 2. Clean text
        # -------------------------------------------------

        cleaned_pages = []

        for page in pages:

            cleaned_text = clean_text(
                page["text"]
            )

            cleaned_pages.append({
                **page,
                "text": cleaned_text
            })

        # -------------------------------------------------
        # 3. Chunk text
        # -------------------------------------------------

        chunks = []

        for page in cleaned_pages:

            page_chunks = create_chunks(
                text=page["text"],
                strategy=strategy,
                chunk_size=500,
                chunk_overlap=50
            )

            for chunk_index, chunk in enumerate(page_chunks):

                chunks.append({
                    "chunk_id": (
                        f"{pdf_file.stem}_"
                        f"p{page['page']}_"
                        f"c{chunk_index}"
                    ),

                    "paper_id": pdf_file.stem,

                    "page": page["page"],

                    "text": chunk
                })

        # -------------------------------------------------
        # 4. Create final JSON
        # -------------------------------------------------

        paper_data = {

            "paper_id": pdf_file.stem,

            "filename": pdf_file.name,

            "total_pages": len(cleaned_pages),

            "total_chunks": len(chunks),

            "chunking_strategy": strategy,

            "chunks": chunks
        }

        # -------------------------------------------------
        # 5. Save JSON
        # -------------------------------------------------

        output_file = (
            output_path /
            f"{pdf_file.stem}.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                paper_data,
                f,
                ensure_ascii=False,
                indent=4
            )

        # -------------------------------------------------
        # 6. Print statistics
        # -------------------------------------------------

        print(
            f"Pages extracted: "
            f"{len(cleaned_pages)}"
        )

        print(
            f"Chunks created: "
            f"{len(chunks)}"
        )

        print(
            f"Saved: {output_file}"
        )


if __name__ == "__main__":
    main()
