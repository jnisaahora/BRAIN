"""
Script de prueba del reranker local.
Ejecuta esto con la API ya corriendo para verificar que funciona.

Uso:
    python test_reranker.py
"""

import requests
import json

BASE_URL = "http://localhost:8001"


def test_health():
    print("=== Test 1: Health check ===")
    r = requests.get(f"{BASE_URL}/health")
    print(f"Status: {r.status_code}")
    print(f"Respuesta: {r.json()}\n")


def test_rerank_documentos():
    print("=== Test 2: Reranking documentos AHORA ===")

    payload = {
        "query": "cuantas horas se han imputado al proyecto X este mes",
        "chunks": [
            {
                "text": "La gestion de proyectos en AHORA ERP permite registrar horas por empleado y por tarea.",
                "source": "manual_erp_v12.pdf"
            },
            {
                "text": "El modulo de RRHH gestiona las horas de los trabajadores y sus vacaciones.",
                "source": "manual_rrhh.pdf"
            },
            {
                "text": "Proyecto X: 142 horas imputadas en marzo 2026. Presupuesto total: 200 horas.",
                "source": "informe_proyectos.pdf"
            },
            {
                "text": "Los proyectos se clasifican por cliente, area de negocio y prioridad.",
                "source": "politicas_proyectos.pdf"
            },
            {
                "text": "Para consultar el estado de un proyecto accede al modulo de gestion en flexygo.",
                "source": "guia_usuario.pdf"
            }
        ],
        "top_n": 3
    }

    r = requests.post(f"{BASE_URL}/rerank", json=payload)
    data = r.json()

    print(f"Query: '{data['query']}'")
    print(f"Chunks recibidos: {data['total_chunks_received']}")
    print(f"Latencia: {data['latency_ms']}ms\n")

    print("Chunks reordenados por relevancia:")
    for i, chunk in enumerate(data["ranked_chunks"]):
        print(f"  [{i+1}] Score: {chunk['score']} | Fuente: {chunk['source']}")
        print(f"       {chunk['text'][:80]}...")
        print()


def test_rerank_sin_contexto():
    print("=== Test 3: Pregunta sin respuesta en los chunks ===")

    payload = {
        "query": "cual es el precio de la licencia enterprise de flexygo",
        "chunks": [
            {
                "text": "Flexygo es una plataforma low-code desarrollada por AHORA Software.",
                "source": "web_ahora.md"
            },
            {
                "text": "El modulo de facturacion soporta facturas electronicas segun normativa espanola.",
                "source": "manual_facturacion.pdf"
            }
        ],
        "top_n": 2
    }

    r = requests.post(f"{BASE_URL}/rerank", json=payload)
    data = r.json()

    print("Scores (todos bajos = ningun chunk responde la pregunta):")
    for chunk in data["ranked_chunks"]:
        print(f"  Score: {chunk['score']} | {chunk['text'][:60]}...")
    print(f"  Latencia: {data['latency_ms']}ms\n")
    print("  -> Un score bajo en todos es una senal para el prompt builder")
    print("     de que debe responder 'no tengo esa informacion'\n")


if __name__ == "__main__":
    print("Iniciando pruebas del reranker local AHORA Brain")
    print("=" * 50)
    try:
        test_health()
        test_rerank_documentos()
        test_rerank_sin_contexto()
        print("Todas las pruebas completadas correctamente.")
    except requests.exceptions.ConnectionError:
        print("ERROR: No se puede conectar a la API.")
        print("Asegurate de que reranker_api.py esta corriendo antes de ejecutar este script.")
