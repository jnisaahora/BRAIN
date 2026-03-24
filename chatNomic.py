import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "documents"

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"

TOP_K = 3


print("Cargando modelo de embeddings...")
model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)


print("Conectando con Qdrant...")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def embed_query(query):

    embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    return embedding.tolist()


def search(query):

    vector = embed_query(query)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=TOP_K
    )

    return results


def print_results(results):

    print("\nResultados encontrados:\n")

    for i, r in enumerate(results, 1):

        payload = r.payload

        print("=" * 80)
        print(f"Resultado {i}")
        print(f"Score: {r.score:.4f}")
        print(f"Documento: {payload.get('source')}")
        print(f"Chunk: {payload.get('chunk')}")
        print()

        text = payload.get("text", "")

        if len(text) > 600:
            text = text[:600] + "..."

        print(text)
        print()


def chat():

    print("\nChat de prueba con Qdrant")
    print("Escribe 'exit' para salir\n")

    while True:

        query = input("Pregunta: ").strip()

        if query.lower() in ["exit", "quit"]:
            sys.exit()

        results = search(query)

        if not results:
            print("\nNo se encontraron resultados\n")
            continue

        print_results(results)


if __name__ == "__main__":
    chat()