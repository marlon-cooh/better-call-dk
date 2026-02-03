
# Better Call DK

Un sistema de Generación Aumentada por Recuperación (RAG) para consultar sentencias legales y jurisprudencia relacionada con redes sociales, ciberacoso y derechos educativos.

## Descripción General

Better Call DK aprovecha Modelos de Lenguaje Grande (LLMs) y bases de datos vectoriales para proporcionar consultoría legal accesible. Los usuarios pueden hacer preguntas en lenguaje natural sin expertise legal y recibir respuestas informadas basadas en decisiones legales pasadas.

## Características

- **Consultas en Lenguaje Natural**: Haz preguntas legales en español o inglés conversacional
- **Búsqueda Vectorial**: Utiliza embeddings de OpenAI con ChromaDB para recuperación semántica de documentos
- **Pipeline RAG**: Combina recuperación con GPT-4 para contexto legal preciso
- **Procesamiento por Lotes**: Maneja eficientemente grandes colecciones de documentos
- **Filtrado de Metadatos**: Self-query retriever para filtrado inteligente de documentos

## Stack Tecnológico

- **LLM**: OpenAI GPT-4o
- **Embeddings**: text-embedding-ada-002
- **Vector DB**: ChromaDB
- **Framework**: LangChain
- **Procesamiento de Datos**: Pandas

## Instalación

```bash
pip install -r requirements.txt
```

Configura las variables de entorno:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Estructura del Proyecto

```
├── consultoria_legal.ipynb    # Notebook con flujo completo
├── rag.py                      # Inicialización de RAG y utilidades
├── chat.py                     # Interfaz de chat y cadena de recuperación
├── data/                       # Documentos legales (formato Excel)
└── vector_db/                  # Almacenamiento persistente de ChromaDB
```

## Uso

```python
from chat import rag_chain, get_response

response = get_response(
    user_query="¿Cuál fue la sentencia del caso de acoso escolar?",
    chain=rag_chain
)
print(response["output"])
```

## Licencia

Licencia MIT.

