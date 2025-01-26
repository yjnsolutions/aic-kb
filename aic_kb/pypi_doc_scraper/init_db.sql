-- Enable the pgvector extension
create extension if not exists vector;

create table tool_docs(
    id bigserial primary key,
    tool_name varchar not null,
    source_type varchar not null,
    url varchar not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone not null,
    metadata jsonb not null default '{}'::jsonb,
    CONSTRAINT unique_tool_docs UNIQUE (tool_name, source_type, url)
);

create table page (
    id bigserial primary key,
    tool_id bigint not null references tool_docs(id),
    url varchar not null,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone not null,
    CONSTRAINT unique_page UNIQUE (tool_id, url)
);

create table page_chunk (
    id bigserial primary key,
    page_id bigint not null references page(id),
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,
    embedding vector(1536),
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    updated_at timestamp with time zone not null,
    CONSTRAINT unique_page_chunk UNIQUE (page_id, chunk_number)
);

-- Create an index for better vector similarity search performance
-- WARNING: You want at least 1000 vectors for an IVFFlat index to be effective, otherwise recall will be poor
create index on page_chunk using ivfflat (embedding vector_cosine_ops);
-- After re-creating the index, you need to reindex the table
-- reindex index page_chunk_embedding_idx;

-- Create an index on metadata for faster filtering
--create index idx_tool_docs_metadata on tool_docs using gin (metadata);


-- Create a function to search for documentation chunks
create function match_site_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  tool_name varchar,
  source_type varchar,
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
    tool_name,
    source_type,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    (1 - (page_chunk.embedding <=> query_embedding)) as similarity
  from page_chunk
    join page on page.id = page_chunk.page_id
    join tool_docs on tool_docs.id = page.tool_id
  where tool_docs.metadata @> filter
  order by page_chunk.embedding <=> query_embedding
  limit match_count;
end;
$$;
