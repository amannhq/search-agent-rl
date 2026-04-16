# Integrations

## OpenEnv

- The main environment subclasses `openenv.core.env_server.interfaces.Environment` in [searcharena/engine/core.py](/Users/aman/Documents/search-env/searcharena/engine/core.py:1)
- The server-facing wrapper also subclasses `Environment` and exposes metadata in [server/environment.py](/Users/aman/Documents/search-env/server/environment.py:1)
- Client support is built on `openenv.core.EnvClient` in [searcharena/client.py](/Users/aman/Documents/search-env/searcharena/client.py:1)
- Environment metadata and state use `openenv.core.env_server.types.State` across engine and server layers

## FastAPI / HTTP Surface

- HTTP service entrypoint is [server/app.py](/Users/aman/Documents/search-env/server/app.py:1)
- Core endpoints: `POST /reset`, `POST /step`, `GET /state`, `GET /tasks`, `GET /metadata`, `GET /schema`
- Runtime uses a single process-global environment instance `_env` in [server/app.py](/Users/aman/Documents/search-env/server/app.py:41)

## OpenAI

- Baseline policy/inference flow uses `AsyncOpenAI` in [inference.py](/Users/aman/Documents/search-env/inference.py:1)
- Environment variables: `API_BASE_URL`, `API_KEY`, `MODEL_NAME`, `ENV_BASE_URL`, `LOCAL_IMAGE_NAME`
- Tool calling interface mirrors environment actions: `search`, `read`, `prune`, `answer`

## Hugging Face Space / Deployment

- App metadata for Space deployment is defined in [README.md](/Users/aman/Documents/search-env/README.md:1)
- OpenEnv space manifest is [openenv.yaml](/Users/aman/Documents/search-env/openenv.yaml:1)
- The repo includes a [Dockerfile](/Users/aman/Documents/search-env/Dockerfile:1) for containerized execution

## Data Generation Tooling

- Generator code under [data/generator](/Users/aman/Documents/search-env/data/generator) integrates with:
- `anthropic` clients for agentic generation workflows in [data/generator/core/utils.py](/Users/aman/Documents/search-env/data/generator/core/utils.py:1)
- OpenAI embeddings and completion APIs in indexing and web/sec/patent domain scripts
- `chromadb` style indexing flows via [data/generator/core/indexing.py](/Users/aman/Documents/search-env/data/generator/core/indexing.py:1)
- Web and SEC domain pipelines pull external content in files like [data/generator/domains/web/utils.py](/Users/aman/Documents/search-env/data/generator/domains/web/utils.py:1) and [data/generator/domains/sec/index.py](/Users/aman/Documents/search-env/data/generator/domains/sec/index.py:1)

## Testing / Local Tooling

- Local environment setup is expected through `uv sync` per [README.md](/Users/aman/Documents/search-env/README.md:1)
- Validation script [validate-submission.sh](/Users/aman/Documents/search-env/validate-submission.sh:1) checks Hugging Face availability, Docker buildability, and OpenEnv validation

## Integration Boundaries

- Runtime environment code under `searcharena/engine` is intentionally local and synchronous
- Server adapters in `server/` should stay thin and delegate to `searcharena.engine`
- Training and inference code depend on the environment package, not the server module directly
