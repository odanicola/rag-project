from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_documents():
    documents = []
    print(len(list(DATA_DIR.glob("*.txt"))), "file teks ditemukan")
    for file_path in DATA_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        documents.append({
            "source": file_path.name,
            "text": text
        })
    return documents


def chuck_text(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    documents = load_documents()
    for document in documents:
        chunks = chuck_text(document["text"])
        print(f"Document: {document['source']}, Chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            # Print first 50 characters of each chunk
            print(f"Chunk {i + 1}: {chunk[:50]}...")
            print(chunk)
