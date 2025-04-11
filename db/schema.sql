-- Enable pgvector extension for vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Users table to store application users
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Tags table to store document tags
CREATE TABLE IF NOT EXISTS tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

-- Documents table to store uploaded PDF documents
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    filename VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    tags JSONB DEFAULT '[]'::JSONB,
    CONSTRAINT content_not_empty CHECK (LENGTH(content) > 0)
);

-- Index on user_id for faster user document queries
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
-- Index on document tags
CREATE INDEX IF NOT EXISTS idx_documents_tags ON documents USING GIN (tags);

-- Document chunks table to store document segments with vector embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536), -- Dimensionality depends on embedding model
    chunk_order INTEGER NOT NULL,
    chunk_metadata JSONB DEFAULT '{}'::JSONB,
    CONSTRAINT content_not_empty CHECK (LENGTH(content) > 0)
);

-- Index on document_id for faster document chunk queries
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
-- Create a vector index for similarity search
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Chat sessions table to store conversations about documents
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- Index on user_id and document_id for faster chat session queries
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_document_id ON chat_sessions(document_id);

-- Messages table to store chat messages
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL, -- 'user' or 'assistant'
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL,
    CONSTRAINT role_check CHECK (role IN ('user', 'assistant')),
    CONSTRAINT content_not_empty CHECK (LENGTH(content) > 0)
);

-- Index on session_id for faster message queries
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
