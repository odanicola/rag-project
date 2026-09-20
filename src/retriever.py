from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_documents():
    """Load documents from the data directory."""
    documents = []
    for file_path in DATA_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        documents.append({
            "source": file_path.name,
            "text": text
        })
    return documents


def chunk_text(text, chunk_size=80, overlap=20):
    """Chunk text into smaller pieces."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_knowledge_base():
    """Build a knowledge base from documents."""
    knowledge_base = []
    documents = load_documents()
    for document in documents:
        chunks = chunk_text(document["text"])
        for chunk in chunks:
            knowledge_base.append({
                "source": document["source"],
                "text": chunk
            })
    return knowledge_base


def retrieve(query, top_k=3):
    """Retrieve the most relevant chunks for a given query."""
    knowledge_base = build_knowledge_base()
    texts = [item["text"] for item in knowledge_base]
    embeddings = model.encode(texts)
    query_embedding = model.encode([query])
    similarities = cosine_similarity(
        query_embedding, embeddings
    )[0]

    ranked_indices = similarities.argsort()[::-1]
    results = []
    for index in ranked_indices[:top_k]:
        results.append({
            "source": knowledge_base[index]["source"],
            "text": knowledge_base[index]["text"],
            "score": float(similarities[index])
        })

    return results


if __name__ == "__main__":
    query = "How many months must I work before taking leave?"
    results = retrieve(query)
    for result in results:
        print(f"Source: {result['source']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text']}\n")
