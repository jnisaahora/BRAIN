"""
AHORA Brain - Reranker Local
API de reranking usando BGE cross-encoder, corre 100% en local (sin GPU).

Uso:
    pip install -r requirements.txt
    python reranker_api.py

La API queda disponible en http://localhost:8001
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import CrossEncoder
import time
import uvicorn

app = FastAPI(title="AHORA Brain - Reranker Local")

# ---------------------------------------------------------------------------
# Modelo: BAAI/bge-reranker-base es ligero (~1GB) y funciona bien en espanol
# Alternativa mas precisa pero mas lenta: BAAI/bge-reranker-v2-m3
# Se descarga automaticamente la primera vez (~1-2 minutos)
# ---------------------------------------------------------------------------
print("Cargando modelo de reranking... (solo la primera vez tarda)")
model = CrossEncoder("BAAI/bge-reranker-base", max_length=512)
print("Modelo cargado y listo.")


# ---------------------------------------------------------------------------
# Modelos de datos
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    text: str
    source: Optional[str] = "desconocido"
    metadata: Optional[dict] = {}

class RerankRequest(BaseModel):
    query: str
    chunks: List[Chunk]
    top_n: Optional[int] = 3

class RankedChunk(BaseModel):
    text: str
    source: str
    metadata: dict
    score: float
    original_index: int

class RerankResponse(BaseModel):
    ranked_chunks: List[RankedChunk]
    query: str
    total_chunks_received: int
    top_n_returned: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Comprueba que la API esta viva. Llamalo desde n8n antes de reranquear."""
    return {"status": "ok", "model": "BAAI/bge-reranker-base"}


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    """
    Recibe una pregunta y una lista de chunks, devuelve los chunks
    reordenados por relevancia real respecto a la pregunta.

    Ejemplo de llamada desde n8n (nodo HTTP Request):
    POST http://localhost:8001/rerank
    {
        "query": "cuantas horas lleva el proyecto X",
        "chunks": [
            {"text": "Proyecto X: 142 horas imputadas", "source": "erp"},
            {"text": "El modulo de RRHH gestiona horas", "source": "manual.pdf"}
        ],
        "top_n": 3
    }
    """
    start = time.time()

    if not request.chunks:
        raise HTTPException(status_code=400, detail="La lista de chunks esta vacia")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="La query no puede estar vacia")

    # Construir pares [pregunta, chunk] para el cross-encoder
    pairs = [[request.query, chunk.text] for chunk in request.chunks]

    # Calcular scores - aqui es donde el modelo lee pregunta+chunk juntos
    scores = model.predict(pairs)

    # Ordenar por score descendente
    ranked = sorted(
        zip(scores, range(len(request.chunks)), request.chunks),
        key=lambda x: x[0],
        reverse=True
    )

    # Coger solo top_n
    top_n = min(request.top_n, len(ranked))
    result = []
    for score, original_index, chunk in ranked[:top_n]:
        result.append(RankedChunk(
            text=chunk.text,
            source=chunk.source,
            metadata=chunk.metadata or {},
            score=round(float(score), 4),
            original_index=original_index
        ))

    latency = round((time.time() - start) * 1000, 1)

    return RerankResponse(
        ranked_chunks=result,
        query=request.query,
        total_chunks_received=len(request.chunks),
        top_n_returned=top_n,
        latency_ms=latency
    )


# ---------------------------------------------------------------------------
# Arrancar servidor
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
