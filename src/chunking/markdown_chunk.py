from langchain_text_splitters import MarkdownHeaderTextSplitter


def markdown_chunking(text: str):
    """
    Markdown / section-aware chunking.

    Splits using:

    # Header
    ## Header
    ### Header
    """

    if not text or not text.strip():
        return []

    headers = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3")
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False
    )

    documents = splitter.split_text(text)

    chunks = []

    for document in documents:

        content = document.page_content.strip()

        if content:
            chunks.append(content)

    return chunks