import os

import requests
from dotenv import load_dotenv

from retriever import retrieve


load_dotenv()


BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")


def generate_answer(question):

    results = retrieve(question, top_k=3)

    print("\n=== RETRIEVED CONTEXT ===")

    for i, result in enumerate(results, start=1):
        print(f"\nResult #{i}")
        print("Source :", result["source"])
        print("Score  :", round(result["score"], 4))
        print("Text   :", result["text"])

    context = "\n\n".join(
        f"Source: {result['source']}\n{result['text']}"
        for result in results
    )

    prompt = f"""
You are a question-answering assistant.

Use ONLY the provided context to answer the question.

Rules:
1. Answer the exact question being asked.
2. Do not provide unrelated information.
3. Do not combine different facts unless necessary.
4. If the answer is not explicitly available in the context,
   say: "The information is not available in the provided documents."

Context:

{context}

Question:

{question}

Answer:
"""

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0,
            "stream": False,
        },
        timeout=60,
    )

    print("STATUS:", response.status_code)
    print("CONTENT-TYPE:", response.headers.get("content-type"))

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":

    question = input("Question: ")

    answer = generate_answer(question)

    print("\nAnswer:")
    print(answer)
