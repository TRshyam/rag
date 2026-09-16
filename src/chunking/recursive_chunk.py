from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_chunking(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 150
):
    """
    Recursive / structure-aware chunking.

    Splitting priority:

    Paragraph
        ↓
    Line
        ↓
    Sentence
        ↓
    Word
        ↓
    Character
    """

    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(text)