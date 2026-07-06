# Deployment Guide — WiniCari AI API

How to stand up the AI layer as a standalone service for the PHP/HTML platform to call.
See `docs/PHP_INTEGRATION.md` for the REST contract itself, and
`docs/DATA_PIPELINE_REPORT.md` for how the underlying models/data are produced.

Scope of this deployment: **delay/ETA, GPS fallback, anomaly detection** (RAG chatbot
excluded — `ENABLE_CHATBOT=false`, see that report's rationale). Nothing here is
autonomous — retraining, redeploying, and rebuilding the reference DB are all commands a
human runs deliberately, by design.

## Why a separate service, not direct integration into the PHP platform?

The short version: **the AI layer is Python, the platform is native PHP, and there is no
practical way to run these specific models inside PHP.** In more detail:

- **No ML runtime in PHP.** The 3 modules in scope are built on scikit-learn
  (`HistGradientBoostingRegressor`), PyTorch (LSTM delay model, LSTM autoencoder for
  anomaly detection, the Kalman-correction LSTM), Prophet (which itself wraps Stan via
  `cmdstanpy`), and `filterpy`'s Kalman filter — none of which have a PHP equivalent. A
  trained model isn't portable data PHP can just "read": a `.joblib` file is a Python
  object graph (pickle format, scikit-learn class instances and all), a `.pt` file is
  PyTorch's own tensor/state-dict serialization. There's no PHP library that understands
  either format, and reimplementing gradient-boosted tree traversal, LSTM forward-pass
  matrix math, and Kalman filter linear algebra in PHP from scratch isn't a realistic
  option — that's rebuilding a numerical computing stack PHP was never designed to have.
- **PHP's request lifecycle doesn't fit "load once, serve many."** The API's `ModelManager`
  loads every artifact (HistGBM, 2 LSTMs, 65 Prophet models, the IF/LSTM-AE anomaly
  ensembles, the foundation parquet, the reference DB) exactly **once**, at process
  startup, and keeps it all resident in memory for the life of that process (see
  `src/api/main.py`'s `lifespan`) — that's why a prediction request is fast. Classic PHP
  (PHP-FPM workers, the model implied by "native PHP") is fundamentally
  shared-nothing/request-scoped: nothing persists in memory between requests by default.
  Re-loading gigabytes of model artifacts from disk on every single page view isn't
  workable — you'd need something unusual (a long-running PHP process manager like
  RoadRunner or Swoole) to even approach what a normal Python process does for free, and
  that still wouldn't solve the "no PyTorch/scikit-learn in PHP" problem above.
- **"Native" narrows this further.** A native PHP + HTML/CSS platform (as opposed to one
  built on a framework with a rich package ecosystem) typically also means no Composer
  dependency management pulling in exotic bindings, no compiled PHP extensions for things
  like ONNX Runtime, and hosting that may not permit long-running background processes at
  all — all of which would be prerequisites for any in-process alternative.
- **Independent deployment/scaling matters too**, separate from the language issue: even
  if the language barrier didn't exist, coupling a multi-GB Python ML stack into the same
  process/server as the PHP app would mean a PHP deploy could break the AI layer (or vice
  versa), and neither could be restarted, scaled, or rolled back independently.

Given all that, HTTP/REST is the natural integration boundary — it's language-agnostic,
PHP already knows how to make HTTP requests (`curl`, same as it would to any third-party
API), and it lets the AI layer keep its models loaded in memory the way it needs to,
completely independent of how the PHP platform is hosted. See `docs/PHP_INTEGRATION.md`
for exactly what that contract looks like from the PHP side.

## Architecture

Two deployment shapes are supported, controlled by Compose profiles in `docker-compose.yml`
— pick based on whether you're sharing a VPS that already runs something else, or starting
from a blank machine.

**Shape A — sharing an existing VPS** (recommended if the company already has one running
the PHP platform, and likely MongoDB too — this also conveniently answers the "is MongoDB
reachable" question for free, since it'd already be on the same box):

```
PHP platform (same VPS)  --HTTP, localhost-->  existing web server (Apache/nginx,
                                                already serving the PHP app + TLS)
                                                        |
                                                proxies a subdomain/path to:
                                                        |
                                          api container, bound to 127.0.0.1:8000 ONLY
                                          (never exposed directly to the internet)
                                                        |
                                          reads models/, data/ (read-only, static,
                                          admin-refreshed artifacts)
                                                        |
                                          live queries: MongoDB on localhost (GPS pings,
                                          only for GPS fallback + real-time anomaly)
```

**Shape B — fresh, dedicated VPS** (nothing else running on it):

```
PHP platform (elsewhere)  --HTTPS + X-API-Key-->  Caddy (TLS, ports 80/443)  -->  api container
                                                                                        |
                                                                          same models/data/Mongo
                                                                          setup as shape A
```

In both shapes, `data/reference/winicari_reference.db` (the SQLite reference DB) is a
static, read-only artifact — never written to or queried live in place of MongoDB. It's
refreshed the same way the trained models are: by re-running the offline pipeline locally
and shipping the updated file to the server.

## 1. One-time server setup

### Shape A — sharing the existing VPS (do this first, it's simpler)

1. Check the box actually has spare capacity before committing to this
   (`free -h`, `nproc`, `df -h`) — the AI layer's heaviest cost is at startup (loading
   torch + 65 Prophet models + the LSTM/IF artifacts into memory, roughly ~1-1.5GB
   resident) plus brief CPU bursts per prediction request, not sustained heavy load.
   `docker-compose.yml` caps the `api` container at `mem_limit: 2g` / `cpus: 1.5` as a
   safety net so it can't starve the PHP app or MongoDB if something misbehaves — adjust
   those numbers to what's genuinely spare.
2. Confirm MongoDB's actual bind address on that box (`mongod.conf` or however it was
   started) — if it's `127.0.0.1`/`localhost`, `MONGO_URL=mongodb://localhost:27017` in
   `.env` just works, no firewall changes needed at all since the API container will run
   on the same host.
3. Install Docker + the Compose plugin if not already present.
4. `git clone` this repo onto the server into its own directory (or pull a release
   tarball) — it doesn't need to live anywhere near the PHP app's own directory.
5. Pick a subdomain or path for the AI API (e.g. `api.yourdomain.tn`, or
   `yourdomain.tn/ai-api/`) and add a reverse-proxy rule to the **existing** web server
   pointing at `127.0.0.1:8000` (the `api` container's exposed loopback port) — it keeps
   using whatever TLS cert setup it already has (certbot, etc.), no new one needed.

   nginx example (as a new `server`/`location` block alongside the existing PHP config):
   ```nginx
   location /ai-api/ {
       proxy_pass http://127.0.0.1:8000/;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
   }
   ```
   Apache example (`mod_proxy` must be enabled):
   ```apache
   ProxyPass /ai-api/ http://127.0.0.1:8000/
   ProxyPassReverse /ai-api/ http://127.0.0.1:8000/
   ```
   Whichever path/subdomain you pick becomes `WINICARI_API_URL` in
   `docs/PHP_INTEGRATION.md`'s PHP snippets.

With this shape, `docker compose up -d --build` (no `--profile` flags) is enough — `caddy`
and `mongodb` both stay off since the existing web server and existing MongoDB already do
those jobs.

### Shape B — fresh, dedicated VPS

1. Provision a small Linux VPS (any provider — DigitalOcean, Hetzner, OVH, etc.). 2 vCPU /
   4GB RAM is a reasonable starting point for CPU-only inference across 3 modules.
2. Point a DNS A record for your API subdomain (e.g. `api.yourdomain.tn`) at the server —
   needed for Caddy's automatic HTTPS.
3. Install Docker + the Compose plugin.
4. `git clone` this repo onto the server.
5. Confirm the real production MongoDB is reachable over the network from this new
   server (it almost certainly is NOT by default, since it currently listens on localhost
   per this project's dev environment) — open it up with proper auth/firewalling
   (ideally a private network/VPC between the two servers, not the open internet).

Start with `docker compose --profile standalone-tls up -d --build` — this also starts
`caddy`, which claims ports 80/443 and gets its own Let's Encrypt cert for `API_DOMAIN`.

## 2. Configure `.env` on the server

```bash
MONGO_URL=mongodb://<real-mongo-host>:27017   # shape A: usually mongodb://localhost:27017
API_KEY=<generate a long random secret, e.g. `openssl rand -hex 32`>
ALLOWED_ORIGINS=https://your-php-platform-domain.tn
API_DOMAIN=api.yourdomain.tn      # only used by caddy -- irrelevant in shape A, no caddy there
ENABLED_MODULES=                  # empty = delay,fallback,anomaly (see next section)
ENABLE_CHATBOT=false
GROQ_API_KEY=            # unused while chatbot is disabled, leave blank
```

### Choosing which modules to load — `ENABLED_MODULES`

Whoever deploys this repo picks the modules; nothing else changes. Valid names:
`delay`, `fallback`, `anomaly`, `chatbot` (comma-separated), or `all`.

```bash
ENABLED_MODULES=                    # unset/empty -> delay,fallback,anomaly (historic default;
                                    #   chatbot still controlled by ENABLE_CHATBOT for back-compat)
ENABLED_MODULES=anomaly             # anomaly-only (free-tier deployments, see below)
ENABLED_MODULES=delay,anomaly       # any subset
ENABLED_MODULES=all                 # everything incl. chatbot (needs GROQ_API_KEY)
```

What a disabled module costs: **nothing**. The heavy libraries (torch, prophet,
chromadb/sentence-transformers) are imported lazily inside each module's `load()`, so a
module that isn't in the list never brings its stack into memory. Its endpoints return
`503` with an explicit "module disabled" message; `/health` reports both
`enabled_modules` (config) and `models` (actually loaded) so you can tell a disabled
module from a failed load at a glance. Dev is unaffected: with nothing set you get the
same three modules as before, and the dashboard tabs for any disabled module just show
their normal "API unavailable" state.

One nuance for slim installs: with `requirements-anomaly.txt` (no torch), the anomaly
module itself degrades gracefully — Isolation Forest handles live scoring, and the LSTM
scores served from `trips_scored.parquet` are the ones precomputed at training time.
Install the full `requirements.txt` if you want live LSTM-AE rescoring too.

## 2bis. Hosting without a company VM (free tiers)

If the company can't provide a VM, two realistic free paths, in order of preference:

**Oracle Cloud "Always Free" (recommended — it IS the VM, just yours):** the Always
Free tier includes up to 4 ARM (Ampere A1) OCPUs + 24 GB RAM permanently — not the
30-day $300 trial credit, which is separate and one-time. That's enough for the FULL
layer (`ENABLED_MODULES=all` minus chatbot if you like) using the normal Shape B steps
above — nothing special needed, it's just a VPS you don't pay for. Caveats: ARM capacity
is often "out of stock" in popular regions (retry / pick another home region at signup —
the home region cannot be changed later), and torch/prophet install fine on aarch64.
Live GPS fallback still requires network access to the company MongoDB from the VM —
that's a data-access question for the company, not a hosting one.

**Render (or similar PaaS) free tier — anomaly only:** 512 MB RAM rules out the full
stack, but the anomaly module needs no torch, no prophet, and **no MongoDB at runtime**
(it serves from static artifacts baked into the image). Use the dedicated image:

```bash
docker build -f Dockerfile.render -t winicari-anomaly .   # ENABLED_MODULES=anomaly baked in
```

On Render: "New Web Service" -> connect the repo -> Docker runtime -> point it at
`Dockerfile.render`, and set `API_KEY`/`ALLOWED_ORIGINS` in the Render env settings.
Artifacts are COPYed into the image (free tier has no persistent disk), so refreshing
models = retrain locally, commit/push artifacts, redeploy — same workflow as §6.
Expect ~200-250 MB resident and a cold start of ~30-60 s after the 15-min idle spin-down.

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
docker compose up -d --build                              # shape A: just `api`
docker compose --profile standalone-tls up -d --build      # shape B: `api` + `caddy`
docker compose --profile local-mongo up -d --build          # add if you also need the bundled MongoDB
```
(profiles combine — e.g. `--profile standalone-tls --profile local-mongo` if you somehow
need both.)

Verify:
```bash
# shape A -- through the existing web server's proxy path/subdomain:
curl https://yourdomain.tn/ai-api/health
curl -H "X-API-Key: $API_KEY" https://yourdomain.tn/ai-api/api/options

# shape B -- Caddy's own domain:
curl https://api.yourdomain.tn/health
curl -H "X-API-Key: $API_KEY" https://api.yourdomain.tn/api/options
```
Either way: `/health` needs no key and should show `models_loaded: true`; the other
endpoint needs the key and returns 401 without it.

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
