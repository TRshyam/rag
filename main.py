from pathlib import Path
import json

from src.ingestion.pdf_loader import extract_pdf
from src.ingestion.cleaner import clean_text

def main():
    print("Starting PDF extraction...")

    pdf_path = Path("data/raw")

    output_path = Path("data/processed")

    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(pdf_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} papers")

    for pdf_file in pdf_files:

        cleaned_pages = []

        pages = extract_pdf(pdf_file)
        print(f"Extracted {len(pages)} pages from {pdf_file.name}")

        for page in pages:
            cleaned_text = clean_text(page['text'])
            cleaned_pages.append({**page, 'text': cleaned_text})

        paper_data={
            "paper_id": pdf_file.stem,
            "filename": pdf_file.name,
            "total_pages": len(cleaned_pages),
            "pages": cleaned_pages
        }

        output_file = output_path / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(paper_data, f, ensure_ascii=False, indent=4)

        print(f"Saved extracted data to {output_file}")
        print(f"Pages extracted: {len(cleaned_pages)}")

        # for page in pages[:2]:
        #     print(f"Page {page['page']}:")
        #     print(page['text'][:500])  # Print first 200 characters of the page text
        #     print("...")  # Indicate that the text is truncated

if __name__ == "__main__":
    main()
    