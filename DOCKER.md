# Running Negotiator AI with Docker

## Layout expected

```
Negotiator_AI/
├── .env                     <- GROQ_API_KEY=... (NOT copied into the image, injected at runtime)
├── docker-compose.yml
├── Dockerfile               <- backend
├── requirements.txt
├── api/
├── env/
├── memory/
├── llm/
├── nlp/
├── agent/policy/ppo_negotiation.zip   <- your trained model
└── frontend/
    ├── Dockerfile           <- frontend
    ├── nginx.conf
    └── ... (the rest of the React app)
```

## First run

```powershell
docker compose up --build
```

- Backend: http://localhost:8000 (docs at http://localhost:8000/docs)
- Frontend: http://localhost:3000

The first build will be slow (torch + transformers + chromadb are heavy) —
that's expected, not a hang. Subsequent builds are cached and much faster
unless requirements.txt changes.

## Things that will bite you if skipped

1. **`.env` must exist at the project root** with `GROQ_API_KEY=...` before
   running `docker compose up`. It's referenced via `env_file` in
   docker-compose.yml — not baked into the image, so you can rotate the key
   without rebuilding.

2. **`agent/policy/ppo_negotiation.zip` must exist before building the
   backend image**, or the API will boot fine but 503 on the first
   `/negotiation/{id}/step` call (same lazy-load behavior as running it
   without Docker).

3. **Rebuilding the frontend after changing the API's host/port**: Vite
   bakes `VITE_API_URL` into the JS bundle at *build* time, not at
   container start. If you ever move the backend off `localhost:8000`,
   you need to rebuild the frontend image with a new build arg:
   ```powershell
   docker compose build --build-arg VITE_API_URL=http://localhost:8080 frontend
   ```
   Editing docker-compose.yml's `args:` value and re-running
   `docker compose up --build` also works.

4. **Memory persistence**: `./memory` and `./agent/policy` are mounted as
   volumes specifically so `docker compose down && docker compose up --build`
   doesn't wipe your client history or force a fresh model copy every time.
   Don't delete the `memory/` folder on the host unless you actually want a
   clean slate.

## Stopping

```powershell
docker compose down
```

Add `-v` only if you intentionally want to also drop any anonymous volumes
(there aren't any here by default — the named mounts above are bind mounts
to your own folders, so they're untouched either way).
