-- Enable the pgvector extension
create extension if not exists vector;

-- TODO add column model, model version, source

-- Create the documentation chunks table
create table openai_site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone not null,
    CONSTRAINT unique_url_chunk UNIQUE (url, chunk_number)
);

-- Create an index for better vector similarity search performance
-- WARNING: You want at least 1000 vectors for an IVFFlat index to be effective, otherwise recall will be poor
create index on openai_site_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_openai_site_pages_metadata on openai_site_pages using gin (metadata);

-- Create a function to search for documentation chunks
create function match_site_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    (1 - (openai_site_pages.embedding <=> query_embedding)) as similarity
  from openai_site_pages
  where metadata @> filter
  order by openai_site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;
