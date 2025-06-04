from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def search_vdb(text: str, k: int = 3, persist_directory: str = "chroma_db") -> List[str]:
    """Search stored documents for the text query and return top k chunks."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    results = db.similarity_search(text, k=k)
    return [doc.page_content for doc in results]
