# 🦙 CustomLLM - Local RAG System

A fully private, local Retrieval-Augmented Generation (RAG) system that runs entirely on your machine. Chat with your documents using Qwen2.5:3B, with zero cloud dependencies and complete data privacy.


## 🏗️ Architecture

```
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Document Processor  │  (Intelligent chunking)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   BGE-M3 Embeddings  │  (Semantic understanding)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   QdrantDB Store    │  (Vector storage)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   Qwen2.5:3B LLM    │  (Answer generation)
└─────────────────────┘
```


## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Zaharah/CustomLLM.git
cd CustomLLM
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Pull required models**
```bash
ollama pull qwen2.5:3b
ollama pull bge-m3
```

### Usage

#### 1) Start the backend API
```bash
uvicorn app:app --reload --port 8000
```

#### 2) Start the React UI
```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173

## 📁 Project Structure

```
CustomLLM/
│
├── app.py                  # FastAPI backend
├── frontend/               # React UI (Vite)
├── rag_engine.py           # Core RAG orchestration
├── document_processor.py   # Document loading & chunking
├── config.py               # Configuration settings
├── evaluation.py           # Retrieval benchmarking & telemetry
├── ingestion/              # Tiered document parsing (Docling/PyMuPDF/RapidOCR)
│   ├── __init__.py
│   └── parser.py
├── benchmark_examples/     # Example benchmark JSONL files
├── requirements.txt
└── qdrant_db/              # Persistent vector storage (auto-created)
```

## ⚙️ Configuration

Edit `config.py` to customize

## 🧠 How It Works

1. **Document Ingestion**: Documents are loaded and split into semantic chunks with overlap
2. **Embedding**: Each chunk is converted to a vector using BGE-M3 embeddings
3. **Storage**: Vectors are stored in QdrantDB for fast similarity search
4. **Retrieval**: Questions are embedded and matched against stored chunks
5. **Generation**: Qwen2.5:3B generates answers based on retrieved context

## 🔧 Troubleshooting

**"Failed to initialize RAG engine"**
- Ensure Ollama is running: `ollama serve`
- Verify models are pulled: `ollama list`

**Slow responses**
- Reduce `RETRIEVAL_TOP_K` in config.py
- Use a smaller model (though Qwen2.5 3B is already optimized for speed)

**Out of memory**
- Reduce `CHUNK_SIZE` to create smaller chunks
- Close other applications to free up RAM

## 📏 Retrieval Benchmark

Use the benchmark helper to measure `recall@k` and latency on labeled examples:

```bash
python evaluation.py --benchmark benchmark_examples/retrieval_benchmark.example.jsonl --top-k 5
```

Optional fields in each JSONL row:
- `question`: required
- `expected_source`: source file name used for recall@k
- `expected_chunk_id`: chunk id used for stricter recall@k
- `metadata_filters`: optional retrieval filters such as `type`, `source`, `section`, or `date`

Add `--with-generation` if you also want end-to-end answer latency, not just retrieval latency.

Build a labeled benchmark directly from the current knowledge base:

```bash
python evaluation.py --build-benchmark benchmark_examples/generated_benchmark.jsonl --max-cases 100
```

The generated file includes `question`, `expected_source`, `expected_chunk_id`, `expected_contains`, and `metadata_filters` so you can run recall@k against chunks that really exist in your uploaded documents.
