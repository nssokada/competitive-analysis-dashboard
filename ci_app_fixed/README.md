
# Competitive Intelligence Platform (Refactor)

Modularized, production-ready refactor of your Streamlit app.

## What's inside
- `app.py` – entry point with a simple router
- `theme.py` – CSS & Plotly layout
- `core/`
  - `state.py` – session-state keys & dataclass
  - `utils.py` – parsing/formatting helpers
  - `io.py` – data loaders, trial normalization, search blob
- `components/`
  - `publications.py` – grouped publications renderer
  - `company_extras.py` – funding/partnerships/M&A/news renderer
- `views/`
  - `overview.py`, `programs.py`, `companies.py`, `clinical_trials.py`, `compare.py`, `cluster.py`

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
- File-upload happens on **Overview** and caches trials/publications in session.
- All pages defensively handle missing data/columns.
- Company extras support both legacy top-level fields and a new `info`-list schema.
- Programs view includes full-text search over key fields.
