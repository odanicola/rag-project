from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"

STORAGE_DIR.mkdir(exist_ok=True)

EMBEDDINGS_FILE = STORAGE_DIR / "embeddings.npy"
CHUNKS_FILE = STORAGE_DIR / "chunks.json"


model = SentenceTransformer("all-MiniLM-L6-v2")


def load_documents():

    documents = []

    for file_path in DATA_DIR.glob("*.txt"):

        text = file_path.read_text(encoding="utf-8")

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text):

    chunks = []

    paragraphs = text.split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:
            chunks.append(paragraph)

    return chunks


def build_chunks():

    all_chunks = []

    documents = load_documents()

    for document in documents:

        chunks = chunk_text(document["text"])

        for chunk in chunks:

            all_chunks.append({
                "source": document["source"],
                "text": chunk
            })

    return all_chunks


def main():

    print("Loading documents...")

    chunks = build_chunks()

    print(f"Total chunks: {len(chunks)}")

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    print("Embedding shape:", embeddings.shape)

    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    with open(
        CHUNKS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\nSaved:")
    print(EMBEDDINGS_FILE)
    print(CHUNKS_FILE)


if __name__ == "__main__":
    main()
