# Agentic RAG starter kit

This project is generated from an optimized RAG pattern. Configure the empty
credentials in `.env` before running the application.

The indexed production collection is an input to this starter kit. Do not run
`make load-docs` against the production collection. Re-indexing belongs to the
separate documents-indexing pipeline and can overwrite or corrupt the shared
collection.

The application exposes `POST /chat/completions`, `GET /health`, and a local
playground at `GET /`.
