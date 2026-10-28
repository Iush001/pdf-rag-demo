"""Step 2: ask questions. Retrieves relevant chunks from Chroma, then asks Claude."""

import argparse

from rag_core import answer


def show(question: str, k: int, show_context: bool):
    reply, hits = answer(question, k=k)
    print(f"\n{reply}\n")
    print("Sources:")
    for i, h in enumerate(hits, start=1):
        print(f"  [{i}] {h.chunk.source}, page {h.chunk.page}  (similarity {h.score:.2f})")
        if show_context:
            print(f"      {h.chunk.text[:300]}...")
    print()


def main():
    p = argparse.ArgumentParser(description="Ask questions about your indexed PDFs.")
    p.add_argument("question", nargs="*", help="Question (omit for interactive mode)")
    p.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    p.add_argument("--show-context", action="store_true", help="Print retrieved text snippets")
    args = p.parse_args()

    if args.question:
        show(" ".join(args.question), args.k, args.show_context)
        return

    print("Interactive mode. Type a question, or 'exit' to quit.")
    while True:
        try:
            q = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit", "q"}:
            break
        if q:
            show(q, args.k, args.show_context)


if __name__ == "__main__":
    main()
