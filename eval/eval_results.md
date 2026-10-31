# Retrieval eval

- Run: 2026-10-31T13:20:15+00:00
- QA set: 30 curated questions on public documents
  (course notes + PubMed abstracts / MedlinePlus pages)
- top_k: 6 · embeddings: see vectorstore/*/manifest.json

| metric | hits | rate |
|---|---|---|
| file hit rate (overall) | 30/30 | 100% |
| page hit rate (pages exist) | 15/20 | 75% |

## Per domain

| domain | file hits | rate |
|---|---|---|
| course | 20/20 | 100% |
| medical | 10/10 | 100% |

## Page-level misses (course domain)

- q01: What is the difference between supervised and unsupervised learning?
- q03: What does it mean for a model to generalize to unseen data?
- q05: Explain what a decision tree is.
- q11: What are keyword arguments in Python function calls?
- q19: What is retrieval-augmented generation and why is it useful?

Page-level metric: expected_page must be among the top-k retrieved chunks (only scored where real pages exist, i.e. PDFs). Questions whose answer text spans more than one adjacent page often stick to the neighbouring page; the file-level metric above is the primary signal.