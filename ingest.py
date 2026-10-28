"""Step 1: read PDFs, chunk them, embed the chunks, and store them in Chroma."""

import argparse
from pathlib import Path

from rag_core import CHROMA_DIR, ingest_pdfs


def main():
    p = argparse.ArgumentParser(description="Index PDFs into a local Chroma database.")
    p.add_argument("--pdf-dir", default="data", help="Folder containing .pdf files (default: data)")
    p.add_argument("--reset", action="store_true", help="Wipe the existing collection first")
    args = p.parse_args()

    print(f"Indexing PDFs from '{args.pdf_dir}' (first run downloads the embedding model)...")
    stats = ingest_pdfs(Path(args.pdf_dir), reset=args.reset)
    for name, n in stats.items():
        print(f"  {name}: {n} chunks")
    print(f"Done. {sum(stats.values())} chunks stored in '{CHROMA_DIR}/'.")


if __name__ == "__main__":
    main()
