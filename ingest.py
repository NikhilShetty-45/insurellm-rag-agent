import glob
from pathlib import Path
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv(override=True)

KNOWLEDGE_BASE = str(Path(__file__).parent / "knowledge-base")
DB_NAME = str(Path(__file__).parent / "vector_db")
embedding = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    """
    Read all the documents from the knowledge base folder
    """
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    """
    Split the documentrs in chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embedding(chunks):
    """
    Embed the chunks and store it in the vector database
    """
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embedding).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding= embedding, persist_directory= DB_NAME
    )

    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embedding(chunks)