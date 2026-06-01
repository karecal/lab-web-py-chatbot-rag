"""
FastAPI application exposing the TechNova RAG chatbot.

Endpoints:
  POST /chat                        — send a question, get an answer + sources
  GET  /chat/history/{session_id}   — retrieve conversation history
  GET  /documentos                  — list indexed documents

Security:
  - Rate limiting: max 10 requests/minute per IP
  - Input validation: max 500 chars per question
  - PII detection: warns if question contains email or name-like patterns
  - Logging: every request is logged (without full document content)
"""

import logging
import re
import time
import uuid
from collections import defaultdict
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import chatbot

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("technova.api")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TechNova RAG Chatbot API",
    description=(
        "Chatbot que responde preguntas sobre las políticas internas de TechNova S.L. "
        "utilizando RAG sobre documentos indexados en ChromaDB."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Stores { ip: [timestamp, ...] } — only keeps timestamps within the last 60 s
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW = 60  # seconds
_rate_limit_store: dict[str, list[float]] = defaultdict(list)


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    timestamps = [t for t in _rate_limit_store[ip] if t > window_start]
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        oldest = min(timestamps)
        retry_after = int(RATE_LIMIT_WINDOW - (now - oldest)) + 1
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Límite de {RATE_LIMIT_REQUESTS} peticiones por minuto alcanzado. "
                f"Inténtalo de nuevo en {retry_after} segundos."
            ),
            headers={"Retry-After": str(retry_after)},
        )
    timestamps.append(now)
    _rate_limit_store[ip] = timestamps


# ── PII detection ─────────────────────────────────────────────────────────────
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
# Patterns like "me llamo X", "mi nombre es X", "soy X González"
_NAME_RE = re.compile(
    r"\b(me llamo|mi nombre es|soy|llamame|llámame)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+",
    re.IGNORECASE,
)


def _detect_pii(text: str) -> list[str]:
    warnings = []
    if _EMAIL_RE.search(text):
        warnings.append("La pregunta parece contener una dirección de email.")
    if _NAME_RE.search(text):
        warnings.append("La pregunta parece contener un nombre personal.")
    return warnings


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    pregunta: str = Field(..., min_length=1, max_length=500, description="Tu pregunta (máx. 500 caracteres)")
    session_id: Optional[str] = Field(None, description="ID de sesión; se genera automáticamente si no se proporciona")

    @field_validator("pregunta")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        # Strip leading/trailing whitespace
        return v.strip()


class ChatResponse(BaseModel):
    respuesta: str
    fuentes: list[str]
    session_id: str
    fragmentos_usados: int
    advertencias_privacidad: list[str] = []


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    mensajes: list[HistoryMessage]
    total_mensajes: int


class DocumentosResponse(BaseModel):
    documentos: list[str]
    total: int


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, summary="Envía una pregunta al chatbot RAG")
async def post_chat(request: Request, body: ChatRequest):
    """
    Envía una pregunta y recibe una respuesta basada en los documentos indexados.

    - **pregunta**: Tu pregunta (máximo 500 caracteres)
    - **session_id**: Opcional. Si no lo proporcionas, se genera uno nuevo.
    """
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip)

    session_id = body.session_id or str(uuid.uuid4())
    pii_warnings = _detect_pii(body.pregunta)

    logger.info(
        "POST /chat | ip=%s | session=%s | q_len=%d | pii=%s",
        client_ip,
        session_id,
        len(body.pregunta),
        bool(pii_warnings),
    )

    try:
        result = chatbot.chat(pregunta=body.pregunta, session_id=session_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.error("Error in chatbot.chat: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno al procesar la pregunta. Comprueba que LM Studio está en ejecución.",
        )

    return ChatResponse(
        respuesta=result["respuesta"],
        fuentes=result["fuentes"],
        session_id=result["session_id"],
        fragmentos_usados=result["fragmentos_usados"],
        advertencias_privacidad=pii_warnings,
    )


@app.get(
    "/chat/history/{session_id}",
    response_model=HistoryResponse,
    summary="Obtiene el historial de una sesión",
)
async def get_chat_history(session_id: str, request: Request):
    """
    Devuelve el historial de conversación para la sesión indicada.
    Devuelve una lista vacía si la sesión no existe o ha expirado.
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info("GET /chat/history | ip=%s | session=%s", client_ip, session_id)

    history = chatbot.get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        mensajes=[HistoryMessage(role=m["role"], content=m["content"]) for m in history],
        total_mensajes=len(history),
    )


@app.get(
    "/documentos",
    response_model=DocumentosResponse,
    summary="Lista los documentos indexados",
)
async def get_documentos(request: Request):
    """
    Devuelve la lista de documentos que han sido indexados en ChromaDB
    y están disponibles para el chatbot.
    """
    client_ip = request.client.host if request.client else "unknown"
    logger.info("GET /documentos | ip=%s", client_ip)

    try:
        docs = chatbot.list_indexed_documents()
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

    return DocumentosResponse(documentos=docs, total=len(docs))


# ── Global error handlers ─────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Error interno del servidor."},
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
