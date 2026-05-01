import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

for env in ("PG_VECTOR_COLLECTION_NAME", "DATABASE_URL","PDF_PATH"):
    if not os.getenv(env):
        raise RuntimeError(f"Variável de ambiente {env} não encontrada")

PDF_PATH = os.getenv("PDF_PATH")

def ingest_pdf():
    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise RuntimeError("Erro ao dividir o documento em chunks")

    chunks_enriched = [
        Document(
            page_content=doc.page_content,
            metadata={key: value for key, value in doc.metadata.items() if value not in ("", None)}
        )
    for doc in chunks
    ]  
    ids = [f"id-{i}" for i in range(len(chunks_enriched))]

    embedding_model = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL"))
    db = PGVector(
        embeddings=embedding_model,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    db.add_documents(chunks_enriched, ids=ids)

if __name__ == "__main__":
    ingest_pdf()