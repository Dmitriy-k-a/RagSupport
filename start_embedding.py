from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def start_embedding(text: str, persist_directory: str = "chroma_db") -> Chroma:
    """Split text into chunks, embed them and store in ChromaDB.

    Args:
        text: Source text to index.
        persist_directory: Directory for Chroma persistence.

    Returns:
        The created Chroma vector store.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = Chroma.from_documents(
        documents, embedding=embeddings, persist_directory=persist_directory
    )
    db.persist()
    return db
