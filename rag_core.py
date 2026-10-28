"""Core building blocks for the RAG demo: PDF loading, chunking, embeddings,
Chroma vector store, retrieval, and answer generation with Claude."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

# ----------------------------------------------------------------- config ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # characters


@dataclass
class Chunk:
    text: str
    source: str   # file name
    page: int     # 1-based page number
    index: int    # chunk number within the page


@dataclass
class Hit:
    chunk: Chunk
    score: float  # cosine similarity, higher = more relevant


# ------------------------------------------------------------ PDF loading ---
def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...] for every page that has text."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


# --------------------------------------------------------------- chunking ---
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window chunker that prefers to break at sentence boundaries."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Try to end on a sentence boundary in the second half of the window.
            window_start = start + size // 2
            cut = max(text.rfind(". ", window_start, end), text.rfind("? ", window_start, end),
                      text.rfind("! ", window_start, end))
            if cut != -1:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # always make progress
    return chunks


def build_chunks(pdf_path: Path) -> list[Chunk]:
    chunks = []
    for page_no, text in load_pdf(pdf_path):
        for idx, piece in enumerate(chunk_text(text)):
            chunks.append(Chunk(piece, pdf_path.name, page_no, idx))
    return chunks


# ------------------------------------------------------------- embeddings ---
@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer  # lazy: slow import
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    vectors = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ----------------------------------------------------------- vector store ---
def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # we store normalized vectors
    )


def _chunk_id(c: Chunk) -> str:
    return hashlib.sha1(f"{c.source}|{c.page}|{c.index}".encode()).hexdigest()


def ingest_pdfs(pdf_dir: Path, reset: bool = False) -> dict:
    """Load every PDF in pdf_dir, embed its chunks, and upsert into Chroma."""
    pdfs = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir!s}")

    collection = get_collection(reset=reset)
    stats = {}
    for pdf in pdfs:
        chunks = build_chunks(pdf)
        if not chunks:
            stats[pdf.name] = 0
            continue
        collection.upsert(  # upsert => re-running ingest is safe
            ids=[_chunk_id(c) for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embed([c.text for c in chunks]),
            metadatas=[{"source": c.source, "page": c.page, "index": c.index} for c in chunks],
        )
        stats[pdf.name] = len(chunks)
    return stats


# -------------------------------------------------------------- retrieval ---
def retrieve(question: str, k: int = 4) -> list[Hit]:
    collection = get_collection()
    if collection.count() == 0:
        raise RuntimeError("The vector store is empty. Run `python ingest.py` first.")
    res = collection.query(query_embeddings=embed([question]), n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        chunk = Chunk(doc, meta["source"], meta["page"], meta["index"])
        hits.append(Hit(chunk, score=1.0 - dist))
    return hits


# ------------------------------------------------------------- generation ---
SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite the passages you rely on like [1] or [2][3]. "
    "If the context does not contain the answer, say you don't know based on the "
    "provided documents. Be concise and do not invent facts."
)


def build_prompt(question: str, hits: list[Hit]) -> str:
    blocks = [
        f"[{i}] (source: {h.chunk.source}, page {h.chunk.page})\n{h.chunk.text}"
        for i, h in enumerate(hits, start=1)
    ]
    return "Context:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def generate_answer(question: str, hits: list[Hit]) -> str:
    import anthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


def answer(question: str, k: int = 4) -> tuple[str, list[Hit]]:
    hits = retrieve(question, k=k)
    return generate_answer(question, hits), hits
