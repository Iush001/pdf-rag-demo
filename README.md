# 📄 PDF RAG Demo

A small, fully working **Retrieval-Augmented Generation (RAG)** example. It loads PDFs, splits them into chunks, creates embeddings, stores them in a **Chroma** vector database, and answers questions with **Claude** — citing the PDF and page each answer came from.

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`, running locally, so no embedding API key is needed
- **Vector DB:** [Chroma](https://www.trychroma.com/), persisted to a local folder
- **LLM:** Anthropic Claude via the official Python SDK
- **PDF parsing:** `pypdf`

## How it works

```
 PDFs ──► extract text ──► chunk (800 chars, 150 overlap) ──► embed ──► Chroma
                                                                         │
 Question ──► embed ──► top-k similar chunks ◄───────────────────────────┘
                              │
                              ▼
              prompt = context + question ──► Claude ──► answer + sources
```

1. **Ingest** (`ingest.py`): reads every PDF in `data/`, splits each page into overlapping chunks, embeds them, and upserts them into Chroma with `source` and `page` metadata.
2. **Ask** (`ask.py`): embeds your question, retrieves the top-k most similar chunks, and sends them to Claude with instructions to answer only from that context and cite passages.

## Project structure

```
rag-demo/
├── data/                  # put your PDFs here (3 sample PDFs generated for you)
├── rag_core.py            # loading, chunking, embeddings, Chroma, retrieval, LLM call
├── ingest.py              # CLI: build the vector index
├── ask.py                 # CLI: ask questions (single or interactive)
├── make_sample_pdfs.py    # regenerates the sample PDFs
├── requirements.txt
├── .env.example
└── .gitignore
```

## Quick start

### 1. Install

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key

```bash
cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

Get a key at <https://console.anthropic.com/>.

### 3. (Optional) Generate sample PDFs

The `data/` folder already contains three sample PDFs with **fictional** facts (a coffee company handbook, a drone manual, and a gardening guide). Because the facts are made up, a correct answer proves the model used your documents. To regenerate them:

```bash
python make_sample_pdfs.py
```

To use your own documents, drop any `.pdf` files into `data/`.

### 4. Build the index

```bash
python ingest.py
```

The first run downloads the embedding model (~90 MB). Re-running is safe; chunks are upserted. Use `--reset` to wipe and rebuild.

### 5. Ask questions

```bash
# one-off
python ask.py "How many days of paid leave do new employees get?"

# interactive
python ask.py

# retrieve more chunks and show the retrieved text
python ask.py -k 6 --show-context "How long does the Zephyr-9 battery last?"
```

Example output:

```
New employees receive 18 days of paid leave per year [1].

Sources:
  [1] aurora_coffee_handbook.pdf, page 2  (similarity 0.71)
  [2] aurora_coffee_handbook.pdf, page 1  (similarity 0.38)
```

### More sample questions

| Question | Source document |
|---|---|
| How long can the Zephyr-9 fly on one battery? | `zephyr9_drone_manual.pdf` |
| What does three amber blinks mean on the drone? | `zephyr9_drone_manual.pdf` |
| When should I plant carrots according to the lunar guide? | `lunar_gardening_guide.pdf` |
| How much is the Bean Fund reimbursement? | `aurora_coffee_handbook.pdf` |
| Who won the 2010 World Cup? | *Not in the documents, so the model should say it doesn't know* |

## Configuration

Set these in `.env` (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | none (required) | Your Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Claude model used for answers |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `CHROMA_DIR` | `chroma_db` | Where Chroma persists data |
| `COLLECTION_NAME` | `pdf_docs` | Chroma collection name |

## Limitations & ideas for next steps

- **Scanned PDFs** have no extractable text; add OCR (e.g. `pytesseract`) first.
- Chunking is character-based; try token-based or heading-aware splitting for better retrieval.
- Add a **reranker** or hybrid (BM25 + vector) search to improve precision.
- Add **conversation memory** so follow-up questions work.
- Wrap `rag_core.answer()` in a **FastAPI** or **Streamlit** UI.
- If you change `EMBEDDING_MODEL`, re-run `python ingest.py --reset`, since vectors from different models aren't compatible.

## Troubleshooting

- **`ANTHROPIC_API_KEY is not set`**: create `.env` from `.env.example`.
- **`The vector store is empty`**: run `python ingest.py` before `ask.py`.
- **Model not found error**: set `ANTHROPIC_MODEL` to a model available on your account.

## License

MIT. Use it however you like.
