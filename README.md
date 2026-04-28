# FinSight Agent

FinSight Agent is an AI-powered personal finance assistant built with FastAPI and Streamlit.

## Features

- Transaction cleaning and preparation pipeline
- Category classification for spending analysis
- Forecasting (30-day spend projection)
- Anomaly detection with severity and explanations
- RAG-style financial Q&A (when OpenAI + LlamaIndex are configured)
- API + UI for interactive usage

## Project Structure

```text
data/
agents/
api/
ui/
eval/
orchestrator.py
test_agents.py
requirements.txt
render.yaml
```

## Quick Start (Local)

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure environment variables in `.env` (already scaffolded in this repo).
4. Run the backend:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
5. Run the frontend:
   ```bash
   python -m streamlit run ui/app.py
   ```

## Required Environment Variables

- `OPENAI_API_KEY` (required for full RAG answers)
- `PLAID_CLIENT_ID` (only for Plaid data ingestion)
- `PLAID_SECRET` (only for Plaid data ingestion)

Optional:
- `FINSIGHT_API_URL`
- `FINSIGHT_MONTHLY_INCOME`
- `FINSIGHT_BUDGETS` (JSON object)

## Render Deployment

This repository includes `render.yaml` configured for a Python web service:

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Python runtime pinned in `runtime.txt`

## Verification

Run:

```bash
python test_agents.py
```

Expected: all tests pass.
