from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage"

STORAGE_DIR.mkdir(exist_ok=True)

INDEX_FILE = STORAGE_DIR / "faiss.index"
CHUNKS_FILE = STORAGE_DIR / "chunks.json"
EMBEDDINGS_FILE = STORAGE_DIR / "embeddings.npy"


model = SentenceTransformer("all-MiniLM-L6-v2")


def load_documents():

    documents = []

    for file_path in DATA_DIR.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

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

        chunks = chunk_text(
            document["text"]
        )

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

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    # FAISS expects float32
    embeddings = embeddings.astype(
        "float32"
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    # Normalize vectors so that
    # inner product == cosine similarity
    faiss.normalize_L2(
        embeddings
    )

    # Save embeddings for debugging/comparison
    np.save(
        EMBEDDINGS_FILE,
        embeddings
    )

    # Number of dimensions
    dimension = embeddings.shape[1]

    # IndexFlatIP = inner product search
    index = faiss.IndexFlatIP(
        dimension
    )

    # Add vectors to FAISS
    index.add(embeddings)

    print(
        "FAISS vectors:",
        index.ntotal
    )

    # Save FAISS index
    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Save metadata/text
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
    print("-", INDEX_FILE)
    print("-", CHUNKS_FILE)
    print("-", EMBEDDINGS_FILE)


if __name__ == "__main__":
    main()
