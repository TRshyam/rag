import re


def clean_text(text):
    """
    Clean extracted PDF text while preserving
    meaningful research-paper content.
    """

    # 1. Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # 2. Remove spaces at the beginning/end of lines
    text = "\n".join(line.strip() for line in text.splitlines())

    # 3. Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Fix words broken by a line break
    # Example:
    # "transfor-\nmer" → "transformer"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # 5. Replace remaining single line breaks with spaces
    # while preserving paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 6. Remove spaces before punctuation
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    # 7. Remove excessive spaces again
    text = re.sub(r" {2,}", " ", text)

    return text.strip()

