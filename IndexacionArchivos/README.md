# Document Indexer — PDF/DOCX → Qdrant

Pipeline de indexación vectorial optimizado para **RAG (Retrieval-Augmented Generation)**.  
Procesa archivos PDF y DOCX, genera embeddings de alta calidad e indexa en Qdrant.

---

## Estructura de carpetas

```
project/
├── indexer.py              # Script principal de indexación
├── search.py               # Utilidad de búsqueda para verificar calidad
├── requirements.txt        # Dependencias Python
├── README.md
│
├── Pdf/                    # ← Aquí van tus documentos
│   ├── contrato_2024.pdf
│   ├── manual_tecnico.docx
│   └── informes/
│       └── Q3_report.pdf
│
├── errores/                # Archivos que fallaron (creada automáticamente)
│   └── archivo_corrupto__20241201_143022.pdf
│
├── indexer.log             # Log completo de ejecución
└── .indexer_state.json     # Estado de indexación (hashes MD5)
```

---

## Instalación

### 1. Requisitos previos

- Python 3.10+
- Docker instalado y corriendo

### 2. Levantar Qdrant en Docker

```bash
# Crear colecion qdrant + contenedor, ejecutar solo la primera vez
 docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant:latest

 docker run -d --name qdrant_Prueba -p 6335:6333 -p 6336:6334 -v ${PWD}/qdrant_Prueba/storage:/qdrant/storage qdrant/qdrant:latest

# Comprobar que se creo el contenedor
 docker ps
# Ya esta listo para ejecutar, ahora si quieres parar el contenedor o volverlo a arrancar
 docker stop qdrant # ponemos qdrant porque es el nombre que indicamos en el primer comando
 docker start qdrant # si apagamos el ordenador con poner este comando ya se arranca el ordenador, no hay que volver a poner el primer comando

# Para hacer un Backup copiar la carpeta qdrant_storage e indexer_state.json (se puede hacer a mano pero parando antes e iniciando despues el contenedor) 
    docker stop qdrant
    cp -r qdrant_storage backup/ #copiar indexer_state.json tambien
    docker start qdrant

# Ver contenedores
    docker ps        # activos
    docker ps -a     # todos

# Eliminar contenedores
    docker rm qdrant #nombre_contenedor
```

Verificar que está corriendo:
```bash
curl http://localhost:6333/healthz
# Respuesta esperada: {"title":"qdrant - vector search engine","version":"..."}
```

### 3. Instalar dependencias Python

```bash
# Crear entorno virtual (recomendado)
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt
```

> **⚠️ GPU (opcional pero recomendado):**  
> El modelo BGE se puede usar en CPU, pero con GPU es ~10x más rápido.  
> Para GPU CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### 4. Preparar documentos

```bash
mkdir -p Pdf
# Copiar tus PDFs y DOCXs dentro de la carpeta Pdf/
```

---

## Ejecución

### Indexar documentos

```bash
python indexer.py
```

Salida de ejemplo:
```
14:32:01  INFO      =================================================================
14:32:01  INFO        DOCUMENT INDEXER — Inicio del pipeline
14:32:01  INFO        Fecha/Hora : 2024-12-01 14:32:01
14:32:01  INFO        Carpeta    : /project/Pdf
14:32:01  INFO        Colección  : documents
14:32:01  INFO        Modelo     : BAAI/bge-large-en-v1.5
14:32:01  INFO      =================================================================
14:32:01  INFO      📁 Archivos encontrados: 12
14:32:05  INFO      ✅ Modelo cargado (768 dims)
14:32:06  INFO      ✅ Conexión a Qdrant establecida
14:32:06  INFO      🆕 Creando colección 'documents'
Indexando: 100%|████████████████| 12/12 [02:14<00:00, 11.2s/archivo]
14:34:20  INFO      ✅ contrato_2024.pdf → 47 chunks indexados
14:34:31  INFO      ✅ manual_tecnico.docx → 134 chunks indexados
...
14:34:35  INFO      =================================================================
14:34:35  INFO        RESUMEN FINAL
14:34:35  INFO        ✅ Procesados   : 11
14:34:35  INFO        ⏭️  Omitidos    : 0  (sin cambios)
14:34:35  INFO        ❌ Errores      : 1
14:34:35  INFO        🧩 Chunks total : 847
14:34:35  INFO        ⏱️  Tiempo total : 154.2s
14:34:35  INFO        📊 Puntos en Qdrant: 847
```

### Buscar en los documentos indexados

```bash
# Búsqueda básica
python search.py "¿Cuáles son las condiciones del contrato?"

# Con más resultados y umbral más bajo
python search.py "procedimiento de renovación" --top 10 --threshold 0.25

# Filtrar por archivo específico
python search.py "datos financieros" --filter filename=reporte_Q3.pdf
```

---

## Configuración avanzada

Edita la clase `Config` en `indexer.py`:

```python
@dataclass
class Config:
    # Cambiar modelo para documentos en español/multilingüe:
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim:   int = 1024   # ← Ajustar dimensión también

    # Chunks más grandes para documentos técnicos densos:
    chunk_size:    int = 768
    chunk_overlap: int = 80

    # HNSW más agresivo para máxima calidad (más lento):
    hnsw_m:            int = 32
    hnsw_ef_construct: int = 400

    # Deshabilitar skip para re-indexar todo:
    skip_already_indexed: bool = False
```

---

## Decisiones técnicas

### Extracción — PyMuPDF vs alternativas

| Librería    | Layout complejo | Encoding | Velocidad | Tablas |
|-------------|:--------------:|:--------:|:---------:|:------:|
| **PyMuPDF** | ✅ Excelente   | ✅       | ✅ Rápido | ✅     |
| pdfminer    | ⚠️ Regular     | ✅       | ❌ Lento  | ❌     |
| pypdf2      | ❌ Básico      | ⚠️       | ✅        | ❌     |

PyMuPDF preserva el orden de lectura visual con `get_text("text", sort=True)`, crítico para PDFs con múltiples columnas.

### Chunking — RecursiveCharacterTextSplitter

Se eligió sobre chunking simple por longitud porque:
1. **Jerarquía de separadores**: respeta párrafos → oraciones → palabras antes de cortar arbitrariamente
2. **Overlap del 10%**: preserva contexto semántico en fronteras sin duplicar excesivamente
3. **512 chars**: sweet spot empírico para modelos SBERT — suficiente para una idea completa, pequeño suficiente para precisión de recuperación

**Trade-off**: chunks más grandes (1024+) mejoran la comprensión contextual pero bajan el precision@k en retrieval. Para documentos muy técnicos con párrafos densos, considera subir a 768.

### Modelo de embeddings — BAAI/bge-large-en-v1.5

- **#1 en MTEB Retrieval** (benchmark estándar para RAG)
- 768 dimensiones: balance calidad/almacenamiento
- Retrieval **asimétrico**: prefijos diferentes para documentos vs queries
- Descargado automáticamente la primera vez (~1.3GB)

**Para documentos en español**: cambiar a `intfloat/multilingual-e5-large` (1024 dims, top-3 MTEB multilingual). Degradación mínima en inglés, gran mejora en otros idiomas.

### HNSW en Qdrant

- `m=16`: cada nodo conectado a 16 vecinos. Sube a 32 para +1-2% recall con +2x memoria
- `ef_construct=200`: calidad del índice en build-time. Reducir a 100 para indexación más rápida
- IDs determinísticos por hash: permite **upsert idempotente** (re-indexar = actualizar, no duplicar)

### Idempotencia

El script es completamente idempotente:
- Hash MD5 detecta si un archivo cambió desde la última indexación
- IDs de puntos generados por hash `(file_hash + chunk_id)` → mismo archivo siempre genera mismo ID
- Re-ejecutar el script actualiza solo los archivos modificados

---

## Integración con LangChain / LlamaIndex

Para usar el índice desde tu pipeline RAG:

```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
model  = SentenceTransformer("BAAI/bge-large-en-v1.5")

def retrieve(query: str, top_k: int = 5):
    q_embedding = model.encode(
        [f"Represent this query for searching relevant passages: {query}"],
        normalize_embeddings=True,
    ).tolist()[0]

    return client.search(
        collection_name="documents",
        query_vector=q_embedding,
        limit=top_k,
        score_threshold=0.35,
        with_payload=True,
    )
```

---

## Troubleshooting

**"Connection refused" en Qdrant**
```bash
docker ps | grep qdrant   # Verificar que el contenedor está corriendo
docker start qdrant        # Si está parado, iniciarlo
```

**PDF sin texto extraíble (escaneado)**  
El archivo se moverá a `errores/`. Para indexar PDFs escaneados necesitas OCR:
```bash
pip install pytesseract pillow
# Y añadir paso de OCR en extract_pdf() antes del get_text()
```

**Memoria insuficiente con GPU**  
Reducir `batch_size` en `Config`:
```python
batch_size: int = 8   # En lugar de 32
```

**Re-indexar todo desde cero**
```bash
rm .indexer_state.json
# Y si quieres vaciar Qdrant también:
python -c "from qdrant_client import QdrantClient; QdrantClient('localhost',6333).delete_collection('documents')"
```
