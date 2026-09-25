# CLAUDE.md — kratikos-content-classifier-lab

Experiments for the category/subcategory classifier of `kratikos-ai-backend` (sibling checkout at
`../kratikos-ai-backend`). Read [README.md](README.md) first: the hypothesis, the candidates and the
rules live there.

- **Never read `.env` or print a credential.** Scripts load it through `src/lab/config.py`.
- **Read only, dev only.** Use `lab.config.get_engine()` / `get_qdrant()`; never open a writable
  connection and never point at production.
- **One folder per experiment** under `experiments/NNN-short-name/`, with a README stating the
  question, the method and the measured result, so the conclusion survives the code.
- Pulled data and trained models go to `data/` and `models/` (gitignored), never into git.
- Code and docs in English; category/subcategory display names in pt-BR.
- Format with isort then black.
