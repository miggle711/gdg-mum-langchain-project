# Docker Setup Guide

This guide walks through setting up and running the project with Docker.

## Prerequisites

### Check Docker Installation

Verify Docker Desktop is installed and running:

```bash
docker --version
docker compose --version
```

If these commands fail, install [Docker Desktop](https://www.docker.com/products/docker-desktop) for your OS.

### Verify Docker is Running

On macOS/Windows, open Docker Desktop app. On Linux, ensure the Docker daemon is running:

```bash
sudo systemctl start docker
```

## Environment Setup

Before running Docker, create a `.env` file **in `backend/`** (not the project root — that's the only `.env` file the app actually reads, per `backend/app/config.py`):

```bash
cd backend
cp .env.example .env
```

Then edit `backend/.env` and add your Google API key:

```
GOOGLE_API_KEY=your_actual_api_key_here
```

**Without this, the backend will fail to start** with a Pydantic validation error for the missing `google_api_key` setting.

### The `docker compose up` gotcha: `.env` alone isn't enough

`docker-compose.yml` does **not** read `backend/.env` directly. It passes config into the backend container via its `environment:` block, using `${GOOGLE_API_KEY}`-style references — which Docker Compose resolves from **your shell's environment at the moment you run `docker compose up`**, not from any `.env` file. Having the right values in `backend/.env` is necessary for non-Docker local runs and for the seed/utility scripts, but `docker compose up` on its own won't pick them up.

Before running `docker compose up`, export the values your shell needs (once per terminal session, or add to your shell profile):

```bash
export GOOGLE_API_KEY=$(grep GOOGLE_API_KEY backend/.env | cut -d= -f2 | tr -d '"')
export JWT_SECRET_KEY=$(grep JWT_SECRET_KEY backend/.env | cut -d= -f2 | tr -d '"')
docker compose up --build
```

Skipping this doesn't always fail loudly — some vars (`JWT_SECRET_KEY`) are required and `docker compose` refuses to start without them, but others silently fall back to an empty string or default, which can manifest later as confusing runtime errors (e.g. Langfuse traces failing to export with a 401) rather than an obvious startup crash.

See [LANGCHAIN_SETUP.md](LANGCHAIN_SETUP.md#google-api-key-setup) for how to get your API key.

## Running Docker Compose

From the project root:

```bash
docker compose up --build
```

This will:
1. Build the backend image (FastAPI + LangChain)
2. Build the frontend image (Angular + nginx)
3. Start both containers on the Docker network
4. Expose the frontend on `http://localhost`

**First run takes 3-5 minutes** (dependencies install, Angular builds). Subsequent runs are faster.

### What to Expect

**Backend logs** show FastAPI starting:
```
backend-1  | INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Frontend logs** show nginx starting:
```
frontend-1 | /docker-entrypoint.sh: Configuration complete; ready for connections
```

Open `http://localhost` in your browser. The chat panel should appear in the bottom-right.

## Common Issues

### Port Already in Use

If you see `Error response from daemon: ... bind: address already in use`, another service is using port 80 or 8000.

**Solution:** Kill existing containers:
```bash
docker compose down
docker compose up --build
```

Or check what's using the ports:
```bash
lsof -i :80      # Check port 80
lsof -i :8000    # Check port 8000
```

### Backend Crashes with API Key Error

**Error:** A Pydantic validation error mentioning `google_api_key` (or the backend container exits immediately after a `docker compose up`).

**Solution:** Two separate things to check:

1. Verify `backend/.env` exists and contains `GOOGLE_API_KEY=...`:

   ```bash
   cat backend/.env
   ```

   If missing, create it as described in [Environment Setup](#environment-setup).
2. Verify you've **exported** the same values into your shell before running `docker compose up` — see [the gotcha above](#the-docker-compose-up-gotcha-env-alone-isnt-enough). Having a correct `backend/.env` alone is not sufficient for `docker compose up`.

### Frontend Shows "Failed to start chat"

**Error:** Chat panel displays "Failed to start chat. Please refresh and try again."

**Causes:**
- Backend not running (check logs: `docker compose logs backend`)
- Wrong API key (backend can't reach Google)
- Backend crashed on startup

**Solution:**
```bash
docker compose logs backend  # Check backend logs
docker compose down
docker compose up --build    # Restart with fresh build
```

### 404 on /api/chat

**Error:** Browser console shows `POST /api/chat 404`

**Cause:** nginx isn't proxying to backend correctly

**Solution:** Check `frontend/chatbot-ui/nginx.conf` has the rewrite rule:
```nginx
location /api/ {
    rewrite ^/api(/.*)$ $1 break;
    proxy_pass http://backend:8000;
}
```

### Docker Compose Can't Find Backend Service

**Error:** `docker.errors.ImageNotFound` or `getaddrinfo failed`

**Cause:** Services aren't on the same Docker network

**Solution:** Verify `docker-compose.yml` has a `networks` section and both services reference it.

## Stopping Docker

```bash
docker compose down
```

This stops and removes containers but keeps images. To also remove images:

```bash
docker compose down --rmi all
```

## Rebuilding After Code Changes

If you change code, rebuild and restart:

```bash
docker compose down
docker compose up --build
```

The `--build` flag ensures fresh images are created.

## Viewing Logs

View all logs:
```bash
docker compose logs
```

View only backend:
```bash
docker compose logs backend
```

View only frontend:
```bash
docker compose logs frontend
```

Follow logs in real-time:
```bash
docker compose logs -f
```

## Development Workflow

1. Make code changes
2. Run `docker compose down && docker compose up --build`
3. Test at `http://localhost`
4. Check logs if issues: `docker compose logs backend`
5. Repeat

For faster iteration on frontend, you can run Angular dev server locally instead of in Docker (see frontend/chatbot-ui README).
