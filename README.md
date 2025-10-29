## Overview

DocuMind is an intelligent document question-answering system that allows users to upload documents and ask questions in natural language. The system uses RAG (Retrieval Augmented Generation) to provide accurate, context-aware answers with source citations.

## Features

### Core Features

- **File Upload & Storage**: Upload multiple documents (PDF, DOCX, TXT)
- **Intelligent Indexing**: Automatic text extraction and semantic embedding generation
- **Vector Search**: Fast similarity search using ChromaDB
- **Natural Language Queries**: Ask questions in plain English
- **AI-Powered Responses**: Context-aware answers using GPT models
- **Source Citations**: View exact document sources for each answer

### Technical Stack

- **Frontend**: Streamlit
- **Document Processing**: PyPDF2, python-docx, LangChain
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: ChromaDB
- **LLM**: Local via Ollama (e.g., llama3.1:8b)
- **Language**: Python 3.9+

## Project Structure

```
documind/
│
├── app.py                    # Main Streamlit application
├── document_processor.py     # Document loading and chunking
├── vector_store.py          # Vector database management
├── llm_handler.py           # LLM integration
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create from .env.example)
├── .env.example            # Example environment file
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.9 or higher
- PowerShell (on Windows)
- Ollama installed and running (see `https://ollama.ai`)

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd documind
```

### Step 2: Create virtual environments (two venvs)

Create the main app venv (Chroma + Streamlit lives here):

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
deactivate
```

Create a separate venv ONLY for Ollama’s Python client to avoid Pydantic version conflicts:

```powershell
py -m venv .venv_ollama
.\.venv_ollama\Scripts\Activate.ps1
pip install --upgrade pip
pip install ollama pydantic>=2
deactivate
```

### Step 3: Pull an Ollama model

Start the Ollama service (Ollama app/daemon) and pull a model, e.g.:

```powershell
ollama pull llama3.1:8b
```

### Step 4: Configure environment for the app

Set `OLLAMA_PY` to point to the Python inside `.venv_ollama` so the app can call it:

```powershell
$env:OLLAMA_PY="C:\GIT\DocuMind\.venv_ollama\Scripts\python.exe"

# Optional: persist for future shells
[System.Environment]::SetEnvironmentVariable("OLLAMA_PY","C:\GIT\DocuMind\.venv_ollama\Scripts\python.exe","User")
```

Verify the worker bridge:

```powershell
& $env:OLLAMA_PY .\ollama_worker.py --check
# Expected output: {"ok": true}
```

### Step 5: Run the application

```powershell
cd C:\GIT\DocuMind
.\.venv\Scripts\Activate.ps1
$env:OLLAMA_PY="C:\GIT\DocuMind\.venv_ollama\Scripts\python.exe"
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

### 1. Upload Documents

- Click the "Upload Documents" button in the sidebar
- Select one or more files (PDF, DOCX, or TXT)
- Click "Process Documents" to index them

### 2. Ask Questions

- Once documents are processed, type your question in the chat input
- The system will:
  - Search for relevant document chunks
  - Generate a context-aware answer
  - Show source citations

### 3. View Sources

- Expand the "View Sources" section to see exact document excerpts used
- Each source shows the document name and page number

### 4. Clear Data

- Use the "Clear All Data" button to reset the system

## Architecture

### Workflow

1. **Document Upload**: User uploads files through Streamlit interface
2. **Text Extraction**: Extract text from PDF, DOCX, or TXT files
3. **Chunking**: Split text into manageable chunks with overlap
4. **Embedding Generation**: Convert chunks to vector embeddings (local model)
5. **Vector Storage**: Store embeddings in ChromaDB
6. **Query Processing**: Convert user question to embedding
7. **Similarity Search**: Find most relevant document chunks
8. **Answer Generation**: Use GPT to generate answer from context
9. **Response Display**: Show answer with source citations

### Components

#### Document Processor (`document_processor.py`)

- Supports PDF, DOCX, and TXT files
- Extracts text with page numbers
- Chunks text using RecursiveCharacterTextSplitter
- Preserves metadata for citations

#### Vector Store Manager (`vector_store.py`)

- Uses ChromaDB for vector storage
- Sentence Transformers for embeddings (no API required)
- Semantic search with cosine similarity
- Persistent storage

#### LLM Handler (`llm_handler.py`)

- Integrates with OpenAI API
- Creates context from retrieved documents
- Generates accurate, source-based answers
- Includes source citations

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: Model to use (default: gpt-3.5-turbo)

### Customization

You can customize the following parameters:

**Document Processing**:

- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

**Vector Search**:

- `k`: Number of similar documents to retrieve (default: 4)

**LLM Generation**:

- `temperature`: Response randomness (default: 0.7)
- `max_tokens`: Maximum response length (default: 500)

## Cost Considerations

### Embeddings

- Uses **local Sentence Transformers** model (FREE)
- No API costs for embeddings

### LLM

- Uses local models via Ollama (no per-token API charges)

## Troubleshooting

### Common Issues

**1. ChromaDB migration hash error**

If you see an error like `InconsistentHashError` regarding `metadb/00001-...sqlite.sql`, your local Chroma store is out of sync.

- The app now auto-recovers by deleting `./chroma_db` and reinitializing.
- If it persists, manually delete the `chroma_db` folder in the project root while the app is stopped, then restart.

### Is `chroma_db` necessary?

- Yes, for persistence. The folder stores the local ChromaDB SQLite database so your indexed embeddings remain across restarts.
- It is created automatically on first run. You can safely delete it to reset the index; the app will recreate it.
- Recommend adding it to `.gitignore` (already included) so large DB files aren’t committed.

**2. Ollama not available / model not found**

- Ensure the Ollama daemon/app is running.
- Run: `ollama pull llama3.1:8b`
- Verify the worker: `& $env:OLLAMA_PY .\ollama_worker.py --check`

**3. PDF extraction issues**

- Some PDFs (scanned images) require OCR
- Try converting PDF to text first

**4. Memory issues**

- Reduce chunk_size
- Process fewer documents at once

## Future Enhancements

### Stretch Goals (from requirements)

- [ ] Document Summarization
- [ ] Conversational Memory
- [ ] Role-based Access Control
- [ ] Multilingual Support
- [ ] Analytics Dashboard

### Additional Ideas

- [ ] Google Drive integration
- [ ] Support for more file types (Excel, CSV)
- [ ] Advanced filters for search
- [ ] Export chat history
- [ ] Fine-tuning on domain-specific data

## License

MIT License

## Support

For issues or questions, please create an issue in the repository.

## Acknowledgments

- Built for Better Hackathon Challenge
- Local LLM via Ollama
- ChromaDB for vector storage
- Sentence Transformers for embeddings
