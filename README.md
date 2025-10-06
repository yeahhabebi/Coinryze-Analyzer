# Coinryze-Analyzer
Educational project to analyze **Coinryze colour trading game data**,  run predictions, and visualize results.

# CoinryzeAnalyzer

Run locally:
- streamlit: `streamlit run streamlit/dashboard.py`
- backend: `uvicorn backend.app:app --reload`
- worker: `python fetcher/fetch_coinryze.py`

Or run full stack:
- `docker-compose up --build`
