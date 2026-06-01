"""
Chatbot RAG — retrieves relevant document chunks from ChromaDB,
builds a context-aware prompt, and generates answers via LM Studio.
Maintains per-session conversation history in memory.
"""

import os
import time
import logging
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
CHAT_MODEL = os.getenv("CHAT_MODEL", "local-model")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5-GGUF")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

COLLECTION_NAME = "technova_docs"
TOP_K = 3
MAX_HISTORY_TURNS = 6  # keep last N user+assistant pairs in context

logger = logging.getLogger(__name__)

openai_client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# In-memory session store: { session_id: [ {role, content}, ... ] }
_sessions: dict[str, list[dict]] = {}

SYSTEM_PROMPT = """Eres el asistente virtual de RR.HH. de TechNova S.L.
Respondes preguntas sobre las políticas internas de la empresa basándote ÚNICAMENTE en los fragmentos de documentación que se te proporcionan.

Reglas que DEBES seguir:
1. Responde solo con información que esté explícitamente presente en los fragmentos de contexto proporcionados.
2. Si el contexto no contiene información suficiente para responder la pregunta, di exactamente: "No tengo información sobre eso en los documentos disponibles."
3. NO inventes datos, fechas, cantidades ni procedimientos que no aparezcan en el contexto.
4. NO hagas suposiciones ni extrapolaciones más allá de lo que dicen los documentos.
5. Cuando cites información, menciona el documento de origen (por ejemplo: "Según la Política de Vacaciones...").
6. Sé conciso y directo. Usa listas o numeración cuando ayude a la claridad.
7. Si la pregunta es ambigua, pide aclaración antes de responder."""


def _get_chroma_collection():
    """Return the ChromaDB collection, raising a clear error if not indexed."""
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return chroma_client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "ChromaDB collection not found. Run 'python indexer.py' first."
        ) from exc


def _embed_query(text: str) -> list[float]:
    """Create an embedding for the user query."""
    response = openai_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def _retrieve_context(question: str, collection) -> tuple[list[str], list[str]]:
    """
    Query ChromaDB for TOP_K most relevant chunks.
    Returns (texts, filenames).
    """
    query_embedding = _embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    texts = results["documents"][0]
    filenames = [m["filename"] for m in results["metadatas"][0]]
    return texts, filenames


def _build_context_block(texts: list[str], filenames: list[str]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for i, (text, fname) in enumerate(zip(texts, filenames), 1):
        parts.append(f"--- Fragmento {i} (fuente: {fname}) ---\n{text}")
    return "\n\n".join(parts)


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep only the last MAX_HISTORY_TURNS * 2 messages (user + assistant pairs)."""
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        return history[-max_messages:]
    return history


def chat(pregunta: str, session_id: str) -> dict:
    """
    Main RAG chat function.

    1. Retrieves the 3 most relevant document fragments from ChromaDB.
    2. Builds a prompt combining context + conversation history.
    3. Calls the LM Studio LLM for a response.
    4. Persists the turn in the session history.

    Returns:
        {
            "respuesta": str,
            "fuentes": list[str],          # unique filenames used
            "session_id": str,
            "fragmentos_usados": int,
        }
    """
    start_time = time.time()

    # Ensure session exists
    if session_id not in _sessions:
        _sessions[session_id] = []

    # Retrieve relevant context
    collection = _get_chroma_collection()
    context_texts, context_filenames = _retrieve_context(pregunta, collection)

    context_block = _build_context_block(context_texts, context_filenames)

    # Build messages list
    # System message + trimmed history + new user turn (with injected context)
    history = _trim_history(_sessions[session_id])

    user_message_with_context = (
        f"CONTEXTO DE DOCUMENTOS:\n{context_block}\n\n"
        f"PREGUNTA DEL USUARIO:\n{pregunta}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_message_with_context},
    ]

    # Call LM Studio
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.1,  # low temperature for factual Q&A
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    elapsed = round(time.time() - start_time, 2)

    # Persist history (store the plain question, not the one with injected context)
    _sessions[session_id].append({"role": "user", "content": pregunta})
    _sessions[session_id].append({"role": "assistant", "content": answer})

    unique_sources = list(dict.fromkeys(context_filenames))  # preserves order, removes dupes

    logger.info(
        "chat | session=%s | sources=%s | fragments=%d | elapsed=%.2fs",
        session_id,
        unique_sources,
        len(context_texts),
        elapsed,
    )

    return {
        "respuesta": answer,
        "fuentes": unique_sources,
        "session_id": session_id,
        "fragmentos_usados": len(context_texts),
    }


def get_history(session_id: str) -> list[dict]:
    """Return conversation history for a session."""
    return _sessions.get(session_id, [])


def clear_session(session_id: str) -> None:
    """Remove session history."""
    _sessions.pop(session_id, None)


def list_indexed_documents() -> list[str]:
    """Return list of unique document filenames currently indexed in ChromaDB."""
    collection = _get_chroma_collection()
    results = collection.get(include=["metadatas"])
    filenames = sorted({m["filename"] for m in results["metadatas"]})
    return filenames


if __name__ == "__main__":
    # Quick smoke test from the terminal
    import uuid

    session = str(uuid.uuid4())
    questions = [
        "¿Cuántos días de vacaciones tengo al año?",
        "¿Puedo llevarme el portátil corporativo a casa?",
        "¿Qué pasa si pierdo el móvil de empresa?",
        "¿Cuál es el precio de una pizza margarita?",  # fuera del contexto
    ]
    for q in questions:
        print(f"\nPregunta: {q}")
        result = chat(q, session)
        print(f"Respuesta: {result['respuesta']}")
        print(f"Fuentes: {result['fuentes']}")
        print("-" * 40)
