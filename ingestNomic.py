import os
import hashlib
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, OptimizersConfigDiff

from sentence_transformers import SentenceTransformer
import tiktoken
from pypdf import PdfReader



DOCS_PATH = "Pdf"
COLLECTION_NAME = "documents"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BATCH_SIZE = 32

ERROR_PATH = "errores"
os.makedirs(ERROR_PATH, exist_ok=True)


print("Cargando modelo de embeddings...")
model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
VECTOR_SIZE = model.get_sentence_embedding_dimension()

tokenizer = tiktoken.get_encoding("cl100k_base")

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def file_hash(path):
    """Hash SHA256 para evitar PDFs duplicados"""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def count_tokens(text):
    return len(tokenizer.encode(text))


def chunk_text(text):

    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), CHUNK_SIZE - CHUNK_OVERLAP):

        chunk = tokens[i:i + CHUNK_SIZE]
        chunks.append(tokenizer.decode(chunk))

    return chunks


def embed_batch(texts):

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()


def create_collection_if_not_exists():

    collections = client.get_collections().collections
    names = [c.name for c in collections]

    if COLLECTION_NAME not in names:

        print("Creando colección en Qdrant...")

        client.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000
            )
        )


def get_existing_hashes():

    hashes = set()

    points, next_page = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True
    )

    while True:

        for p in points:
            if p.payload and "file_hash" in p.payload:
                hashes.add(p.payload["file_hash"])

        if next_page is None:
            break

        points, next_page = client.scroll(
            collection_name=COLLECTION_NAME,
            offset=next_page,
            limit=1000,
            with_payload=True
        )

    return hashes


def process_pdf(path, known_hashes):

    h = file_hash(path)

    if h in known_hashes:
        print(f"Duplicado ignorado: {path}")
        return []

    filename = os.path.basename(path)

    try:
        reader = PdfReader(path)
    except Exception as e:
        print(f"Error leyendo PDF {path}: {e}")

        error_path = os.path.join(ERROR_PATH, filename)
        try:
            os.rename(path, error_path)
        except Exception:
            pass

        return []

    text = ""

    for page_number, page in enumerate(reader.pages):

        try:
            extracted = page.extract_text()

            if extracted:
                text += extracted + "\n"

        except Exception as e:
            print(f"Error leyendo página {page_number} en {filename}: {e}")
            continue

    if not text.strip():

        print(f"PDF sin texto: {path}")

        error_path = os.path.join(ERROR_PATH, filename)

        try:
            os.rename(path, error_path)
        except Exception:
            pass

        return []

    chunks = chunk_text(text)

    payloads = []

    for i, chunk in enumerate(chunks):

        payloads.append({
            "text": chunk,
            "source": filename,
            "chunk": i,
            "tokens": count_tokens(chunk),
            "file_hash": h
        })

    return payloads


def ingest():

    create_collection_if_not_exists()

    known_hashes = get_existing_hashes()

    batch_texts = []
    batch_payloads = []

    for root, _, files in os.walk(DOCS_PATH):

        for file in files:

            if not file.lower().endswith(".pdf"):
                continue

            path = os.path.join(root, file)

            print(f"Procesando: {file}")

            payloads = process_pdf(path, known_hashes)

            for payload in payloads:

                batch_texts.append(payload["text"])
                batch_payloads.append(payload)

                if len(batch_texts) >= BATCH_SIZE:

                    embeddings = embed_batch(batch_texts)

                    points = []

                    for emb, pay in zip(embeddings, batch_payloads):

                        points.append(
                            PointStruct(
                                id=str(uuid4()),
                                vector=emb,
                                payload=pay
                            )
                        )

                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )

                    print(f"Subidos {len(points)} chunks")

                    batch_texts = []
                    batch_payloads = []

    if batch_texts:

        embeddings = embed_batch(batch_texts)

        points = []

        for emb, pay in zip(embeddings, batch_payloads):

            points.append(
                PointStruct(
                    id=str(uuid4()),
                    vector=emb,
                    payload=pay
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print(f"Subidos {len(points)} chunks finales")


if __name__ == "__main__":
    ingest()