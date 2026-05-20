![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Chatbot con RAG sobre documentos propios

## Objetivo

Construir un chatbot que responda preguntas sobre un conjunto de documentos usando RAG. El sistema debe ser honesto cuando no tiene información suficiente y debe proteger la privacidad del usuario.

Aunque lo construyas con Python, ya sabemos que nuestros ordenadores no son capaces de gestionar un entrenamiento. Por lo que haz la programación en Python para que aprendas cómo funciona, pero a la hora de conectarlo usa un modelo ya entrenado en LM Studio

## Setup

```bash
# fork & clone the repository
cd lab-web-py-chatbot-rag
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn openai chromadb python-dotenv tiktoken
pip freeze > requirements.txt
```

## Dataset a usar

Crea una carpeta `docs/` con al menos 5 archivos `.txt` que contengan información sobre tu tema elegido. Ejemplos:

- Políticas de una empresa ficticia (RR.HH., devoluciones, horarios)
- FAQ de un producto de software
- Reglas de un juego de mesa
- Guía de viaje de una ciudad

## Arquitectura del sistema

```
docs/              ← tus documentos fuente
  └── *.txt
indexer.py         ← lee docs, crea embeddings, guarda en ChromaDB
chatbot.py         ← RAG + LLM: responde preguntas usando el índice
api.py             ← FastAPI que expone el chatbot
```

## Parte 1: Indexador

`indexer.py` debe:
1. Leer todos los `.txt` de `docs/`
2. Fragmentarlos si son largos (función de chunking)
3. Crear embeddings con OpenAI
4. Almacenarlos en ChromaDB con metadatos (nombre del archivo, chunk_id)
5. Imprimir resumen: N documentos, N chunks, N tokens procesados, coste estimado

## Parte 2: Chatbot RAG

`chatbot.py` debe implementar:

```python
def chat(pregunta: str, session_id: str) -> dict:
    """
    1. Recupera los 3 fragmentos más relevantes
    2. Construye el prompt con contexto
    3. Mantiene historial de la conversación
    4. Indica las fuentes usadas en la respuesta
    """
    return {
        "respuesta": "...",
        "fuentes": ["archivo1.txt", "archivo2.txt"],
        "session_id": session_id,
        "fragmentos_usados": 3
    }
```

El sistema prompt debe incluir:
- Instrucción de responder solo con el contexto disponible
- Instrucción de decir "No tengo información sobre eso" si el contexto no es relevante
- Instrucción de no inventar datos

## Parte 3: API FastAPI

`api.py` debe exponer:
- `POST /chat` — pregunta + session_id → respuesta + fuentes
- `GET /chat/history/{session_id}` — historial de la sesión
- `GET /documentos` — lista de documentos indexados

## Parte 4: Medidas de privacidad y seguridad

Implementa:
- Rate limiting básico (máx 10 peticiones por minuto por IP)
- Validación de input (longitud máxima de pregunta: 500 caracteres)
- Logging de cada llamada (sin loguear el contenido completo de los documentos)
- Si la pregunta contiene información personal del usuario (nombre, email), advertir antes de enviar al LLM

## Entrega

- Repositorio en GitHub
- `README.md` con instrucciones de instalación y cómo indexar los documentos
- Capturas de conversaciones de prueba en Swagger UI o Postman
Todos nuestros ejercicios están almacenados en GitHub, así que sigue este [enlace](https://github.com/ironhack-labs/lab-web-py-chatbot-rag) para acceder.
