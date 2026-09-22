from pathlib import Path
import json

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STORAGE_DIR = PROJECT_ROOT / "storage"

INDEX_FILE = STORAGE_DIR / "faiss.index"
CHUNKS_FILE = STORAGE_DIR / "chunks.json"


model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def load_knowledge_base():

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        chunks = json.load(file)

    return index, chunks


def retrieve(query, top_k=3):

    index, chunks = load_knowledge_base()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    # Same normalization used during ingestion
    faiss.normalize_L2(
        query_embedding
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        # FAISS returns -1 if no result exists
        if index_id == -1:
            continue

        results.append({
            "source": chunks[index_id]["source"],
            "text": chunks[index_id]["text"],
            "score": float(score)
        })

    return results
