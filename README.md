[![Tests Status](https://github.com/yorrick-org/aic-kb/actions/workflows/tests.yml/badge.svg)](https://github.com/yorrick-org/aic-kb/actions)

 [![codecov](https://codecov.io/gh/yorrick-org/aic-kb/branch/main/graph/badge.svg)](https://codecov.io/gh/yorrick-org/aic-kb)    

# AI Coding Knowledge Base


Web scraping tools + semantic search for AI coding agents.

Note: this can be implemented AWS Kendra for automated crawling, chunking and indexing. Kendra will be tested in the future.

Data is crawled with Crawl4AI, then put in a local postgres database with embeddings to support semantic search.

## Installation

1. Clone this repository
2. Install dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/):
```fish
uv sync
```
3. Install [direnv](https://direnv.net/docs/installation.html) and [configure this](https://direnv.net/man/direnv.toml.1.html#codeloaddotenvcode) to load the `.env` file.

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
aic-kb get-package-documentation requests --version 2.31.0
aic-kb get-package-documentation typer
aic-kb get-package-documentation rich

# then, we can query the knowledge base with the `search` command
aic-kb search 'make an POST with requests' --match-count 5
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
