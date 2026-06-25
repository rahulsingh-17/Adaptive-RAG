# Adaptive RAG Chatbot

An agentic Retrieval-Augmented Generation (RAG) system that decides, per query, whether to answer from indexed documents, general LLM knowledge, or a live web search — instead of always retrieving the same way.

## Why "adaptive"?

Most basic RAG setups always retrieve, even for questions that don't need it. This project routes each incoming question through a classifier first, then sends it down one of three paths:

| Query type | What handles it |
|---|---|
| Answerable from uploaded docs | Vector retriever (Qdrant) |
| General knowledge | Direct LLM call |
| Needs current/real-time info | Web search (Tavily) |

The routing, document-relevance grading, and query rewriting are all orchestrated as a graph using LangGraph, rather than a single linear chain.

## Tech stack

- **Backend:** FastAPI + Uvicorn
- **Orchestration:** LangGraph (ReAct-style agent)
- **LLM:** OpenAI (GPT-4o)
- **Vector store:** Qdrant
- **Web search:** Tavily
- **Chat memory:** MongoDB (with in-memory fallback)
- **Frontend:** Streamlit

## How it works

1. A user query comes in through the FastAPI `/rag/query` endpoint
2. A classification step labels it as **index**, **general**, or **search**
3. Based on the label, the graph routes to the retriever, a plain LLM call, or a Tavily search
4. If documents were retrieved, a grading step checks they're actually relevant — if not, the query gets rewritten and retried
5. A final node generates the response and returns it, with the conversation saved to MongoDB under a session ID

Documents are uploaded and chunked/embedded via a separate `/rag/documents/upload` endpoint, then indexed into Qdrant for retrieval.

## Project layout

```
.
├── src/
│   ├── main.py              # FastAPI app entry point
│   ├── api/routes.py        # Query + upload endpoints
│   ├── rag/                 # Graph construction, nodes, retriever, ReAct agent
│   ├── models/              # Pydantic schemas (state, grading, routing)
│   ├── llms/                # OpenAI client setup
│   ├── memory/              # MongoDB / in-memory chat history
│   ├── db/                  # Mongo client
│   └── config/              # Settings + prompt templates
└── streamlit_app/           # Chat UI + document upload page
```

## Running it locally

**Requirements:** Python 3.9+, MongoDB, a Qdrant instance, an OpenAI API key, a Tavily API key.

```bash
git clone <your-repo-url>
cd Adaptive-Rag
python -m venv venv
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_key
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=adaptive_rag
```

Then run both services:

```bash
# Terminal 1
python -m uvicorn src.main:app --reload --port 8000

# Terminal 2
streamlit run streamlit_app/home.py
```

- App: http://localhost:8501
- API docs: http://localhost:8000/docs

## What I'd improve next

- [ ] Add automated tests for the routing/grading logic
- [ ] Swap hardcoded prompts for a config-driven prompt versioning setup
- [ ] Add a Dockerfile for one-command setup
- [ ] Support multi-document sessions instead of one collection per upload
