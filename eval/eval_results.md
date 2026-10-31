# Retrieval eval — rebuild demo set

- Run: 2026-10-31T11:30:19+00:00
- QA set: 20 hand-written questions on public course notes (this repo only)
- top_k: 6 · embeddings: see vectorstore/manifest.json

| metric | hits | rate |
|---|---|---|
| file hit rate | 20/20 | 100% |
| page hit rate | 15/20 | 75% |

## Page-level misses

- q01: What is the difference between supervised and unsupervised learning?
- q03: What does it mean for a model to generalize to unseen data?
- q05: Explain what a decision tree is.
- q11: What are keyword arguments in Python function calls?
- q19: What is retrieval-augmented generation and why is it useful?

Page-level metric: expected_page must be among the top-k retrieved chunks. Questions whose answer text spans more than one adjacent page often stick to the neighbouring page; the file-level metric above is the primary signal.