from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_DIR = PROJECT_ROOT / "storage"

EMBEDDINGS_FILE = STORAGE_DIR / "embeddings.npy"
CHUNKS_FILE = STORAGE_DIR / "chunks.json"

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_knowledge_base():
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    return embeddings, chunks


def retrieve(query, top_k=5):
    embeddings, chunks = load_knowledge_base()

    # print("embeddings", embeddings)
    # print("chuncks", chunks)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    # print("query: ", query)
    # print("query_embedding: ", query_embedding)

    similarities = cosine_similarity(query_embedding, embeddings)[0]
    ranked_indices = similarities.argsort()[::-1]
    results = []
    for index in ranked_indices[:top_k]:
        results.append({
            "source": chunks[index]["source"],
            "text": chunks[index]["text"],
            "score": float(similarities[index])
        })

    return results
