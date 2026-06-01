![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Chatbot con RAG sobre documentos propios

Chatbot que responde preguntas sobre las **políticas internas de TechNova S.L.** (empresa ficticia) utilizando RAG (Retrieval-Augmented Generation). El sistema recupera fragmentos relevantes de documentos reales y los usa como contexto para el modelo de lenguaje.

## Arquitectura

```
docs/                     ← documentos fuente (.txt)
  ├── politica_vacaciones.txt
  ├── politica_trabajo_remoto.txt
  ├── politica_gastos.txt
  ├── onboarding.txt
  ├── politica_equipamiento.txt
  └── codigo_conducta.txt
indexer.py                ← chunking + embeddings + ChromaDB
chatbot.py                ← lógica RAG + historial de sesión
api.py                    ← FastAPI (3 endpoints + seguridad)
chroma_db/                ← base de datos vectorial (generada al indexar)
```

```
┌─────────┐  pregunta   ┌──────────┐  embedding  ┌──────────┐
│ Usuario │ ──────────► │  api.py  │ ──────────► │ ChromaDB │
└─────────┘             └──────────┘             └──────────┘
                              │                       │ top-3 chunks
                              │      prompt + contexto│
                              ▼                       ▼
                        ┌───────────────────────────────┐
                        │         LM Studio (LLM)        │
                        └───────────────────────────────┘
                                       │ respuesta
                                       ▼
                              JSON response + fuentes
```

## Requisitos previos

- **Python 3.11+**
- **LM Studio** instalado y en ejecución con:
  - Un **modelo de chat** cargado (p. ej. `Meta-Llama-3-8B-Instruct`)
  - Un **modelo de embeddings** cargado (p. ej. `nomic-embed-text-v1.5`)
  - El servidor local habilitado en `http://localhost:1234`

## Instalación

```bash
# Clona el repositorio
git clone https://github.com/tu-usuario/lab-web-py-chatbot-rag
cd lab-web-py-chatbot-rag

# Crea y activa el entorno virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Instala dependencias
pip install fastapi uvicorn openai chromadb python-dotenv tiktoken
pip freeze > requirements.txt
```

## Configuración

```bash
# Copia el archivo de ejemplo y edítalo
cp .env.example .env
```

Edita `.env` con los nombres de los modelos que tengas cargados en LM Studio:

```env
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_API_KEY=lm-studio
CHAT_MODEL=nombre-del-modelo-en-lm-studio
EMBEDDING_MODEL=nombre-del-modelo-de-embeddings
CHROMA_DB_PATH=./chroma_db
DOCS_PATH=./docs
```

> **Cómo encontrar el nombre del modelo:** en LM Studio, el identificador del modelo aparece en la parte superior de la ventana del servidor local. Cópialo exactamente tal como aparece.

## Indexar los documentos

```bash
python indexer.py
```

Salida esperada:
```
============================================================
TechNova RAG Indexer
============================================================

[1/4] Loading documents from './docs'...
      Found 6 documents: [...]

[2/4] Chunking documents...
      politica_vacaciones.txt: 3 chunks
      ...
      Total: 18 chunks across 6 documents
      Total tokens: 9,842
      Estimated cost if using OpenAI API: $0.0002  (local LM Studio = $0.00)

[3/4] Connecting to ChromaDB at './chroma_db'...
[4/4] Creating embeddings and storing in ChromaDB...

Indexing complete!
  Documents : 6
  Chunks    : 18
  Tokens    : 9,842
```

## Arrancar la API

```bash
python api.py
# o equivalente:
uvicorn api:app --reload --port 8000
```

Swagger UI disponible en: **http://localhost:8000/docs**

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `POST` | `/chat` | Envía una pregunta, recibe respuesta + fuentes |
| `GET` | `/chat/history/{session_id}` | Historial de la sesión |
| `GET` | `/documentos` | Lista de documentos indexados |

### Ejemplo: POST /chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "¿Cuántos días de vacaciones tengo al año?", "session_id": "test-1"}'
```

Respuesta:
```json
{
  "respuesta": "Según la Política de Vacaciones, todos los empleados a tiempo completo tienen derecho a 23 días laborables de vacaciones al año.",
  "fuentes": ["politica_vacaciones.txt"],
  "session_id": "test-1",
  "fragmentos_usados": 3,
  "advertencias_privacidad": []
}
```

### Ejemplo: Pregunta fuera del contexto

```json
{
  "respuesta": "No tengo información sobre eso en los documentos disponibles.",
  "fuentes": [...],
  "fragmentos_usados": 3
}
```

## Medidas de seguridad y privacidad

| Medida | Implementación |
|--------|---------------|
| **Rate limiting** | Máx. 10 peticiones/minuto por IP — responde `429 Too Many Requests` |
| **Validación de input** | Longitud máxima 500 caracteres (Pydantic `max_length`) |
| **Detección de PII** | Detecta emails y patrones de nombre en la pregunta; devuelve `advertencias_privacidad` |
| **Logging** | Registra IP, session_id, longitud de pregunta y fuentes — sin contenido de documentos |

## Probar desde Swagger UI

1. Abre `http://localhost:8000/docs`
2. Despliega `POST /chat` → **Try it out**
3. Escribe una pregunta y un `session_id` (o déjalo vacío para autogenerar)
4. Haz clic en **Execute**

## Estructura de respuesta del chatbot

```python
{
    "respuesta": str,            # Respuesta generada por el LLM
    "fuentes": list[str],        # Archivos .txt usados como contexto
    "session_id": str,           # ID de sesión
    "fragmentos_usados": int,    # Número de chunks recuperados (siempre 3)
    "advertencias_privacidad": list[str]  # Avisos PII si los hay
}
```

## Documentos incluidos (TechNova S.L.)

| Archivo | Contenido |
|---------|-----------|
| `politica_vacaciones.txt` | Días de vacaciones, asuntos propios, festivos |
| `politica_trabajo_remoto.txt` | Modelo híbrido, herramientas, seguridad remota |
| `politica_gastos.txt` | Límites de reembolso, dietas, viajes, tarjeta corporativa |
| `onboarding.txt` | Primer día, período de prueba, beneficios, accesos |
| `politica_equipamiento.txt` | Hardware asignado, uso aceptable, seguridad del dispositivo |
| `codigo_conducta.txt` | Valores, acoso, conflictos de interés, canal de denuncias |
