# NegotiateAI — Freelancer Negotiation Agent

RL + NLP + LLM real-time coaching system for freelance payment negotiations.
See project blueprint PDF for full spec.

## Setup
pip install -r requirements.txt
cp .env.example .env  # add your GROQ_API_KEY

## Structure
- env/        Gymnasium simulation environment + archetypes
- agent/      PPO training + saved policies
- nlp/        Sentiment + archetype classification
- llm/        Groq client simulator + script generator
- memory/     ChromaDB vector memory + RAG
- api/        FastAPI backend
- frontend/   React app

## Status: Week 1, Day 1 — defining archetype prompt templates
