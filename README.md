# multi-ai-research-agent

## Running with Docker

The app itself is containerized; PostgreSQL (Supabase) and Redis (Upstash) are
hosted externally and are not part of this setup -- the container connects out
to them over the network using the same `DATABASE_URL`/`REDIS_URL` as a local
run.

**Prerequisite:** a `.env` file must exist in the project root with real
values (`OPENROUTER_API_KEY`, `TAVILY_API_KEY`, `DATABASE_URL`, `REDIS_URL`,
and optionally `GROQ_API_KEY`) before starting the container -- it is read at
container start via `env_file` and is never copied into the image itself.

```bash
# Build the image
docker compose build

# Start the app (reachable at http://localhost:8000, docs at /docs)
docker compose up
```

### Running migrations against the container setup

Migrations aren't run automatically on container startup. Run them as a
one-off command against the same `.env`-configured database instead:

```bash
docker compose run --rm app alembic upgrade head
```

(Full setup, architecture, and usage docs land in a later milestone.)