# Deployment Guide — WiniCari AI API

How to stand up the AI layer as a standalone service for the PHP/HTML platform to call.
See `docs/PHP_INTEGRATION.md` for the REST contract itself, and
`docs/DATA_PIPELINE_REPORT.md` for how the underlying models/data are produced.

Scope of this deployment: **delay/ETA, GPS fallback, anomaly detection** (RAG chatbot
excluded — `ENABLE_CHATBOT=false`, see that report's rationale). Nothing here is
autonomous — retraining, redeploying, and rebuilding the reference DB are all commands a
human runs deliberately, by design.

## Architecture

```
PHP platform  --HTTPS + X-API-Key-->  Caddy (TLS)  -->  api container (FastAPI, uvicorn)
                                                              |
                                                    reads models/, data/ (read-only,
                                                    static, admin-refreshed artifacts)
                                                              |
                                                    live queries: MongoDB (GPS pings,
                                                    only for GPS fallback + the new
                                                    real-time anomaly endpoint)
```

`data/reference/winicari_reference.db` (the SQLite reference DB) is one of those static,
read-only artifacts — never written to or queried live in place of MongoDB. It's refreshed
the same way the trained models are: by re-running the offline pipeline locally and
shipping the updated file to the server.

## 1. One-time server setup

1. Provision a small Linux VPS (any provider works — DigitalOcean, Hetzner, OVH, etc. — the
   architecture below doesn't depend on which one). 2 vCPU / 4GB RAM is a reasonable
   starting point for CPU-only inference across 3 modules.
2. Point a DNS A record for your API subdomain (e.g. `api.yourdomain.tn`) at the server —
   needed for Caddy's automatic HTTPS.
3. Install Docker + the Compose plugin.
4. `git clone` this repo onto the server (or pull a release tarball — whichever you
   prefer; there's no build-on-server step beyond `docker compose build`).

**Open question you need to confirm before going further**: is the real production
MongoDB (`Historique_pos` especially) already reachable over the network from wherever
this server will be, or does it currently only listen on localhost the way this
development environment's does? If it's localhost-only, either open it up with proper
auth/firewalling, or co-locate this API on the same host as MongoDB.

## 2. Configure `.env` on the server

```bash
MONGO_URL=mongodb://<real-mongo-host>:27017
API_KEY=<generate a long random secret, e.g. `openssl rand -hex 32`>
ALLOWED_ORIGINS=https://your-php-platform-domain.tn
API_DOMAIN=api.yourdomain.tn
ENABLE_CHATBOT=false
GROQ_API_KEY=            # unused while chatbot is disabled, leave blank
```

Give the `API_KEY` value to the PHP team out-of-band (not via a ticket/chat that gets
logged in plaintext forever) — it's what `docs/PHP_INTEGRATION.md` expects in the
`X-API-Key` header.

## 3. Deploy the static artifacts

These are built **locally**, never on the server:

```bash
conda activate bus-intelligence
python -m src.build_reference_db --with-trips     # refresh data/reference/winicari_reference.db
python -m src.train_pipeline                       # refresh models/ (writes models_version.json)
```

Then copy the result to the server (rsync shown; scp works too):
```bash
rsync -avz models/ data/reference/ data/processed/foundation_arrivals_full.parquet \
      data/processed/line_distances.parquet \
      user@server:/path/to/winicari/models/ ...  # match the compose volume paths
```

## 4. Start the stack

```bash
docker compose up -d --build
```

This starts `api` (the FastAPI service) and `caddy` (reverse proxy/TLS). MongoDB is **not**
started by this command — it's opt-in (see the "local-mongo" profile in
`docker-compose.yml`) for the case where you don't have a MongoDB elsewhere yet:
```bash
docker compose --profile local-mongo up -d --build
```

Verify:
```bash
curl https://api.yourdomain.tn/health          # no key needed, should show models_loaded: true
curl -H "X-API-Key: $API_KEY" https://api.yourdomain.tn/api/options   # 200 with a key, 401 without
```

## 5. Updating code (CI/CD)

- **CI** (`.github/workflows/ci.yml`) runs automatically on every push/PR: builds the
  image, boots it, and smoke-tests `/health`. It doesn't deploy anything — just catches
  broken builds before you try to deploy them.
- **CD** (`.github/workflows/deploy.yml`) is **manual-trigger only** (`workflow_dispatch`)
  — go to the Actions tab on GitHub and click "Run workflow" when you actually want to
  push a code change live. It builds the image, pushes it to GitHub Container Registry,
  SSHes into the server, pulls the new image, and restarts the `api` container. No commit
  ever auto-deploys.
- Required GitHub Actions secrets for CD: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`
  (private key with SSH access to the server), `DEPLOY_PATH` (absolute path to this repo
  on the server, e.g. `/home/deploy/winicari`).

## 6. Updating models/data (retraining)

Retraining is **never** automated — always a deliberate local command (see
`docs/DATA_PIPELINE_REPORT.md` §8), then manually shipped:

```bash
# locally
python -m src.build_reference_db [--with-trips]
python -m src.train_pipeline

# ship to server (same rsync as step 3), then:
docker compose restart api
```

`models/models_version.json` (written by `train_pipeline.py`) records the git commit and
each module's training metrics at build time — `/health`'s `model_version` field always
tells you exactly which artifact set the running container is serving. Keep the previous
`models/`/`data/` directories around on the server (e.g. rename to `models_previous/`
before overwriting) so a bad deploy can be rolled back by pointing the compose volume
mounts back and restarting — no registry service, just directories.

## 7. Monitoring

- `logs/api.log` (rotating, 10MB × 5 files) — one line per request: method, path, query
  params, status code, latency, and the model commit that served it. Mounted to
  `./logs/` on the host via `docker-compose.yml`, so it survives container restarts/redeploys.
- `/health`'s `uptime_seconds` and `request_count` give a cheap sanity check without any
  metrics stack.
- Docker's built-in `HEALTHCHECK` (in `Dockerfile.api`) means `docker ps` shows
  `(healthy)`/`(unhealthy)` directly, and `restart: unless-stopped` will restart the
  container if it starts failing health checks.

## 8. Local development (not part of the production deployment)

The Streamlit dashboard and Jupyter notebooks are dev-only tools, kept out of the
production compose file entirely:
```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml --profile local-mongo up
```
Leave `API_KEY` unset in `.env` for this stack — the dashboard's HTTP client doesn't send
an API key header (internal tool, not the PHP-facing path). Install
`requirements-dev.txt` (not `requirements.txt`) in your local conda env if you're running
things outside Docker for notebook work.
