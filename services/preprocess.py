import re

def clean_text(text: str) -> str:
    """
    Clean and normalize input text.

    Steps:
    - Convert to lowercase
    - Remove URLs
    - Remove extra spaces
    - Remove special characters
    """

    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
if __name__ == "__main__":
    sample = "Breaking News!!! Visit https://abc.com NOW!!!   "
    print("Original:", sample)
    print("Cleaned :", clean_text(sample))