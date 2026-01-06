# Birbal — Personal Knowledge Base Search Engine

Birbal is a hybrid semantic + lexical search engine for your personal notes.
It uses Ollama for local LLM inference and PostgreSQL + pgvector for high-quality retrieval.

---

## 1. Setup Ollama (Required)

Birbal uses Ollama for all LLM and embedding operations.
It expects Ollama to already be running with the required models installed on the host machine.

### Install Ollama

**macOS (Homebrew)**

```bash
brew install ollama
```

### Start Ollama

```bash
ollama serve
```

### Download required models

You can configure the models you would like to use in the `.env` file.
The following models are recommended for Macbook M4 Pro as of 1 Jan, 2026.

```bash
ollama pull qwen3-embedding:8b
ollama pull llama3:instruct
```

---

## 2. Run using Docker (Recommended)

### Create .env file

Copy `example.env` to `.env`
Replace `FILE_DIR` with the folder containing your notes.

---

### Start the server

```bash
docker compose up --build
```

The API will be available at:

```
http://localhost:8080
```

---

## 3. Run without Docker

### Install dependencies

```bash
uv pip install -e .
```

---

### Run the API server

```bash
birbal-server
```

Server runs at:

```
http://localhost:8080
```

---

## Example Query

```bash
curl "http://localhost:8080/query?q=how%20does%20rust%20ownership%20work"
```
