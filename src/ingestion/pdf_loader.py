import pymupdf


def extract_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text()

        pages.append({
            "page": page_number + 1,
            "text": text
        })

    doc.close()

    return pages