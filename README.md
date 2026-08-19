# MovieScout AI

Hybrid semantic movie search with BM25, dense retrieval, Reciprocal Rank Fusion,
optional HyDE expansion and cross-encoder reranking. This is an information
retrieval system—not a full RAG question-answering application.

## Architecture

```text
TMDB -> clean one document/movie -> dense + sparse vectors -> Qdrant
query -> BM25 + dense -> RRF -> confidence router
                              | easy: results
                              ` hard: HyDE -> retrieve -> rerank -> results
```

The index intentionally stores **one searchable document per movie** because
movie overviews are short. There is no passage chunking or MaxP aggregation.
TMDB rating is displayed as metadata and never mixed into relevance ranking.

## Current evaluation

The existing 200-query benchmark is retained for historical comparison, but it
is not a reliable test set: it covers 50 movies with four templated queries per
movie, and some queries overlap closely with indexed overviews. Treat results in
[`evaluation/`](evaluation/) as preliminary, not as evidence of generalization.

A credible next benchmark should use human-written paraphrases, typos, incomplete
descriptions and graded relevance (0–3), with separate locked development/test
sets. Report Recall@K, Precision@K, MRR@10, nDCG@10, p50/p95 latency, API error
rate, Groq calls/cost, and repeated HyDE runs. Router/HyDE claims should only be
made after an ablation against BM25, dense, RRF and reranking baselines.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Fill `.env` with your own credentials. No credentials are bundled with the
project. If a key has ever been committed, revoke it before publishing and purge
it from Git history with an appropriate history-rewrite tool.

```bash
python -m pipeline.ingest
python -m pipeline.clean
python -m pipeline.dual_embedding_qdrant
streamlit run ui/app_final.py
```

Re-index after upgrading because numeric `release_year` is now part of each
Qdrant payload and is used for year-range filters.

Alternatively:

```bash
docker compose up --build
```

## Testing and quality

```bash
pip install -e ".[dev]"
ruff check pipeline retrieval ui tests
pytest -q
```

CI runs the same checks on pushes and pull requests.

## Language and limitations

- The current dense and cross-encoder models are English models. Unicode input
  is preserved, but Vietnamese retrieval quality is not claimed. Use English
  descriptions until a multilingual model is evaluated and the index rebuilt.
- HyDE needs Groq and is non-deterministic; service failures fall back to the
  original query.
- Qdrant must be available and populated before search works.
- The confidence-router thresholds are provisional pending a dev/test ablation.
- The dataset is a TMDB snapshot and inherits TMDB coverage and metadata biases.

## Repository map

- [`pipeline/`](pipeline/) — TMDB ingestion, cleaning and one-document indexing
- [`retrieval/`](retrieval/) — hybrid retrieval, RRF, routing and reranking
- [`ui/app_final.py`](ui/app_final.py) — Streamlit interface
- [`tests/`](tests/) — deterministic unit tests
- [`evaluation/`](evaluation/) — legacy benchmark artifacts and reports

## Security

Never commit `.env`. `.env.example` contains variable names only. Rotate the
previously exposed TMDB key; deleting it from the current file does not revoke it
or remove it from earlier Git commits.
