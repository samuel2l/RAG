"""
Simple RAG pipeline with Pinecone + OpenAI.

What it does:
1. Takes a local document path.
2. Extracts text from PDF/TXT/MD/CSV/JSON/HTML.
3. Splits the text into chunks.
4. Embeds chunks with OpenAI.
5. Stores chunks in Pinecone.
6. Retrieves relevant chunks for a question.
7. Answers using only the retrieved document context.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader


load_dotenv()

# Keep this blank as requested.
# You can also set PINECONE_API_KEY in a .env file instead.
PINECONE_API_KEY = ""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-documents")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "document-rag")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_DIMENSION = 1536


openai_client = OpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


def require_keys() -> None:
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY. Add it to your .env file.")

    if not PINECONE_API_KEY:
        raise ValueError(
            "Missing PINECONE_API_KEY. Put your key in rag.py or set it from env."
        )


def create_index_if_needed() -> None:
    existing_indexes = [index["name"] for index in pinecone_client.list_indexes()]

    if PINECONE_INDEX in existing_indexes:
        return

    pinecone_client.create_index(
        name=PINECONE_INDEX,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)

    return "\n".join(pages).strip()


def extract_text(path: str) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)

    if suffix in {".txt", ".md", ".csv", ".json", ".html", ".htm"}:
        return file_path.read_text(encoding="utf-8", errors="replace").strip()

    raise ValueError(
        f"Unsupported file type: {suffix}. Use PDF, TXT, MD, CSV, JSON, or HTML."
    )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def document_id(path: str) -> str:
    absolute_path = str(Path(path).resolve())
    digest = hashlib.sha256(absolute_path.encode("utf-8")).hexdigest()[:16]
    return f"doc-{digest}"


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


def index_document(path: str, force: bool = False) -> dict[str, Any]:
    require_keys()
    create_index_if_needed()

    index = pinecone_client.Index(PINECONE_INDEX)
    doc_id = document_id(path)
    title = Path(path).name

    existing = index.fetch(ids=[f"{doc_id}-0"], namespace=NAMESPACE)

    if existing.vectors and not force:
        return {
            "doc_id": doc_id,
            "title": title,
            "chunks": 0,
            "skipped": True,
        }

    text = extract_text(path)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError("No text could be extracted from the document.")

    embeddings = embed_texts(chunks)

    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append(
            {
                "id": f"{doc_id}-{i}",
                "values": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunk,
                    "title": title,
                    "source": str(Path(path).resolve()),
                },
            }
        )

    index.upsert(vectors=vectors, namespace=NAMESPACE)

    return {
        "doc_id": doc_id,
        "title": title,
        "chunks": len(vectors),
        "skipped": False,
    }


def retrieve_context(
    question: str,
    doc_id: str,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    index = pinecone_client.Index(PINECONE_INDEX)

    question_embedding = embed_texts([question])[0]

    results = index.query(
        vector=question_embedding,
        top_k=top_k,
        namespace=NAMESPACE,
        include_metadata=True,
        filter={"doc_id": {"$eq": doc_id}},
    )

    matches = []
    for match in results.get("matches", []):
        metadata = match.get("metadata", {})
        matches.append(
            {
                "score": match.get("score"),
                "text": metadata.get("text", ""),
                "title": metadata.get("title", ""),
                "source": metadata.get("source", ""),
                "chunk_index": metadata.get("chunk_index"),
            }
        )

    return matches


def build_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    context_text = "\n\n---\n\n".join(
        f"Chunk {item['chunk_index']}:\n{item['text']}"
        for item in contexts
    )

    return f"""
Answer the question using only the document context below.

If the answer is not in the context, say:
"I could not find that in the document."

Document context:
{context_text}

Question:
{question}

Answer:
""".strip()


def answer_question(
    question: str,
    doc_id: str,
    top_k: int = 4,
) -> str:
    contexts = retrieve_context(question, doc_id=doc_id, top_k=top_k)

    if not contexts:
        return "No relevant context found in the document."

    prompt = build_prompt(question, contexts)

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Answer only from the provided context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response.choices[0].message.content.strip()

    sources = "\n".join(
        f"- {item['title']} | chunk {item['chunk_index']} | score {item['score']:.4f}"
        for item in contexts
    )

    return f"{answer}\n\nSources:\n{sources}"


def interactive_chat(doc_id: str, top_k: int) -> None:
    print("\nDocument is ready. Ask questions about it.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        answer = answer_question(question, doc_id=doc_id, top_k=top_k)
        print(f"\n{answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a document using RAG.")
    parser.add_argument("document", help="Path to the document")
    parser.add_argument(
        "-q",
        "--question",
        help="Ask one question and exit",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of document chunks to retrieve",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index the document even if it was already indexed",
    )

    args = parser.parse_args()

    result = index_document(args.document, force=args.force)

    if result["skipped"]:
        print(f"Document already indexed: {result['title']}")
    else:
        print(f"Indexed {result['chunks']} chunks from {result['title']}")

    if args.question:
        print()
        print(answer_question(args.question, doc_id=result["doc_id"], top_k=args.top_k))
    else:
        interactive_chat(doc_id=result["doc_id"], top_k=args.top_k)


if __name__ == "__main__":
    main()