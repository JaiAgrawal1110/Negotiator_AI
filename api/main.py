"""
FastAPI backend - all endpoints, localhost:8000.
TODO (Week 11-12): orchestrate env/agent/nlp/llm/memory layers.
"""
from fastapi import FastAPI

app = FastAPI(title="NegotiateAI API")

@app.get("/health")
def health():
    return {"status": "ok"}
