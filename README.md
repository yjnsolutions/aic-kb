[![Tests Status](https://github.com/yorrick-org/aic-kb/actions/workflows/tests.yml/badge.svg)](https://github.com/yorrick-org/aic-kb/actions)

 [![codecov](https://codecov.io/gh/yorrick-org/aic-kb/branch/main/graph/badge.svg)](https://codecov.io/gh/yorrick-org/aic-kb)    

# AI Coding Knowledge Base

Yorrick's tools and knowledge base for AI coding (with Aider).
Crawler cli + documentation search API aimed at feeding agentic AI coding tools. 

Data is crawled with Crawl4AI, then put in a local postgres database with embeddings.

## Installation

1. Clone this repository
2. Install dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/):
```fish
uv sync
```
3. Install [direnv](https://direnv.net/docs/installation.html) and https://direnv.net/man/direnv.toml.1.html#codeloaddotenvcode to load the `.env` file.

## Usage

First, create a `.env` file from template and adjust values if needed:

```fish
cp .env.template .env
```

Then run database container (data is persisted on host):
```fish
docker run --name aic-kb \
  -p 127.0.0.1:$POSTGRES_PORT:5432 \
  --volume $PWD/data/postgres:/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
  -e POSTGRES_USER="$POSTGRES_USER" \
  -e POSTGRES_DB="$POSTGRES_DB" \
  -d pgvector/pgvector:pg17
```

The package provides a CLI command `aic-kb`. Basic usage:

```fish
aic-kb get-package-documentation requests --version 2.31.0 --limit 3
```


## Development

To connect to DB:

```fish
docker exec -ti aic-kb psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"
```


## Development

Run tests with:

```fish
uv run pytest -v tests/
```
