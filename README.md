
this is not a production based RAG system, its just a basic implementation of RAG concept, gonna make it production-based letter.

## Features

- **PDF Upload** — Upload any PDF document and build a vector database
- **Mistral AI Embeddings** — Uses `MistralAIEmbeddings` for document embedding
- **Mistral AI LLM** — Uses `ChatMistralAI` (mistral-small-2506) for question answering
- **Chroma Vector Store** — Persisted locally for fast retrieval
- **Streamlit UI** — Simple web interface for interaction

## Requirements

- Python 3.10+
- `MISTRAL_API_KEY` environment variable (set in `.env`)

## Setup

1. **Install dependencies:**
  
   pip install -r requirements.txt

2. **Set environment variables:**
   Create a `.env` file:
  
   MISTRAL_API_KEY="your-mistral-api-key"


## How to Run

### Streamlit Web App (app.py)


.venv/bin/streamlit run app.py


Then open `http://localhost:8501` in your browser:
1. Upload a PDF using the file uploader
2. Click **"Create Vector Database"** — the previous `chroma_db` is automatically deleted and rebuilt
3. Enter your question(qs must be related to the pdf,though its a basic chat rag system cant recognize everything) and click submit



if the stramlit doesnt work must try cli version
### CLI Version (main.py)
# First, create the vector database
.venv/bin/python db.py ------>it will take time
(if chroma-db exist delete it firstly)

# Then run the interactive CLI
.venv/bin/python main.py


Type your question and press Enter. Type `0` to exit.


## Notes

- Each PDF upload deletes the previous `chroma_db` folder to avoid stale data
- Embeddings are generated via Mistral API with automatic token-based batching (up to 16k tokens per request)
- The Mistral tokenizer is downloaded from Hugging Face on first run