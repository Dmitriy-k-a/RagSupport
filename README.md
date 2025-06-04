# RagSupport

This repository provides simple helpers for working with the LangChain
framework and a local Chroma vector database.

* `start_embedding.py` – takes raw text, splits it into chunks, creates
  embeddings using a HuggingFace model and stores them in a persistent
  Chroma database.
* `searchVDB.py` – queries the persisted database for the most relevant
  chunks to a text query.
