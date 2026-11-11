-- Runs once after the dvdrental restore (first container start).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

-- Film description embeddings (bge-m3). Kept in sync by the Prefect flow `sync-embeddings`:
-- text_hash changes when title/description change; deleted films drop their row via ON DELETE CASCADE.
CREATE TABLE IF NOT EXISTS film_embedding (
    film_id     INT PRIMARY KEY REFERENCES film(film_id) ON DELETE CASCADE,
    text_hash   TEXT NOT NULL,
    embedding   vector(1024) NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS film_embedding_hnsw ON film_embedding USING hnsw (embedding vector_cosine_ops);

-- BM25 over film text; updates by itself on INSERT/UPDATE/DELETE
CREATE INDEX IF NOT EXISTS film_bm25 ON film USING bm25 (film_id, title, description) WITH (key_field = 'film_id');

-- The agent's SQL runs as dvd_reader: SELECT only, read-only transactions, 10 s statement timeout.
DO $$ BEGIN
    CREATE ROLE dvd_reader LOGIN PASSWORD 'dvd_reader';
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
GRANT CONNECT ON DATABASE dvdrental TO dvd_reader;
GRANT USAGE ON SCHEMA public TO dvd_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dvd_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dvd_reader;   -- tables added later
ALTER ROLE dvd_reader SET default_transaction_read_only = on;
ALTER ROLE dvd_reader SET statement_timeout = '10s';
ALTER ROLE dvd_reader SET idle_in_transaction_session_timeout = '30s';
