# AI Coding Knowledge Base

Yorrick's tools and knowledge base for AI coding (with Aider).
Crawler cli + documentation search API aimed at feeding agentic AI coding tools. 

Data is crawled with Crawl4AI, then put in a local postgres database with embeddings.

## Installation

1. Clone this repository
2. Install dependencies using [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
uv sync
```

## Usage

First, create a `.env` file from template and adjust values if needed:

```bash
cp .env.template .env
```

Then run database container (data is persisted on host):
```bash
docker run --name aic-kb \
  -p 127.0.0.1:${POSTGRES_PORT}:5432 \
  --volume $PWD/data/postgres:/var/lib/postgresql/data \
  -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
  -e POSTGRES_USER=${POSTGRES_USER} \
  -e POSTGRES_DB=${POSTGRES_DB} \
  -d pgvector/pgvector:pg17
```

The package provides a CLI command `aic-kb`. Basic usage:

```bash
aic-kb get-package-documentation requests --version 2.31.0 --limit 3
```


## Development

To connect to DB:

```bash
docker exec -ti aic-kb psql -U postgres
```
