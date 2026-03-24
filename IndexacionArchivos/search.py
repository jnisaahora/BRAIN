#!/usr/bin/env python3
"""
search.py — Utilidad de búsqueda semántica sobre el índice Qdrant.
Permite verificar la calidad de recuperación después de indexar.

Uso:
    python search.py "¿Cuál es el procedimiento de contratación?"
    python search.py "machine learning techniques" --top 10
    python search.py "informe anual" --filter filename=reporte_2024.pdf
"""

import sys
import argparse
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

# Importar configuración del indexador
from indexer import CONFIG, BGE_INDEX_PREFIX


def search(
    query: str,
    top_k: int = 5,
    filename_filter: str = None,
    score_threshold: float = 0.35,
) -> list[dict]:
    """
    Busca los chunks más relevantes para una query.

    Aplica el prefijo BGE de CONSULTA (diferente al de indexación)
    para retrieval asimétrico de alta calidad.

    Args:
        query:            Texto de búsqueda
        top_k:            Número de resultados a retornar
        filename_filter:  Filtrar por nombre de archivo específico
        score_threshold:  Umbral mínimo de similitud coseno
    """
    # Cargar modelo
    model  = SentenceTransformer(CONFIG.embedding_model)
    client = QdrantClient(host=CONFIG.qdrant_host, port=CONFIG.qdrant_port)

    # Prefijo de QUERY (distinto al de indexación en BGE)
    query_with_prefix = f"Represent this query for searching relevant passages: {query}"
    embedding = model.encode(
        [query_with_prefix],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()[0]

    # Filtro opcional por archivo
    query_filter = None
    if filename_filter:
        query_filter = qdrant_models.Filter(
            must=[qdrant_models.FieldCondition(
                key="filename",
                match=qdrant_models.MatchValue(value=filename_filter),
            )]
        )

    # Búsqueda vectorial con umbral de score
    results = client.query_points(
        collection_name=CONFIG.collection_name,
        query=embedding,
        limit=top_k,
        query_filter=query_filter,
        score_threshold=score_threshold,
        with_payload=True,
    ).points

    return results


def display_results(results: list, query: str) -> None:
    """Muestra resultados de forma legible."""
    print(f"\n{'═' * 70}")
    print(f"  🔍 Query: \"{query}\"")
    print(f"  📊 Resultados encontrados: {len(results)}")
    print(f"{'═' * 70}\n")

    if not results:
        print("  ⚠️  No se encontraron resultados por encima del umbral de similitud.")
        return

    for i, hit in enumerate(results, 1):
        p = hit.payload
        score = hit.score

        print(f"  [{i}] Score: {score:.4f}  |  {p.get('filename', 'N/A')}", end="")
        if "page" in p:
            print(f"  (pág. {p['page']}/{p.get('total_pages', '?')})", end="")
        if "section" in p:
            print(f"  [sección: {p['section'][:40]}]", end="")
        print()
        print(f"      Chunk {p.get('chunk_id', '?')}/{p.get('total_chunks', '?')} "
              f"| {p.get('char_count', 0)} chars")
        print()

        # Mostrar texto del chunk (primeros 300 chars)
        text = p.get("text", "")
        preview = text[:300].replace("\n", " ")
        if len(text) > 300:
            preview += "..."
        print(f"      {preview}")
        print()
        print(f"  {'─' * 66}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Búsqueda semántica en documentos indexados")
    parser.add_argument("query", help="Texto de búsqueda")
    parser.add_argument("--top",    type=int,   default=5,    help="Número de resultados (default: 5)")
    parser.add_argument("--filter", type=str,   default=None, help="Filtrar por filename=X")
    parser.add_argument("--threshold", type=float, default=0.35, help="Score mínimo (default: 0.35)")
    args = parser.parse_args()

    filename_filter = None
    if args.filter and "=" in args.filter:
        filename_filter = args.filter.split("=", 1)[1]

    results = search(
        query=args.query,
        top_k=args.top,
        filename_filter=filename_filter,
        score_threshold=args.threshold,
    )
    display_results(results, args.query)
