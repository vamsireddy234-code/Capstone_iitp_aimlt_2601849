# Zepto GenAI RAG Service

This project is a small **GenAI RAG (Retrieval-Augmented Generation)** application for answering questions about Zepto policies.

The application takes a question, checks whether it is about a Zepto policy, searches the local Zepto documents when needed, and returns a JSON answer.

**The required version works completely offline.** No LLM API key or signup is needed.

---

## How it works

```text
Zepto Policy Documents
        ↓
Load + Split Documents
        ↓
Create Embeddings
(all-MiniLM-L6-v2)
        ↓
Store in ChromaDB
        ↓
User Question
        ↓
LangGraph
        ↓
Is it a policy question?
     ↙           ↘
   YES            NO
    ↓              ↓
Search ChromaDB   Direct Answer
    ↓
Top 3 Documents
    ↓
Final JSON Response
```

### 1. Ingestion

The 8 Zepto policy files are stored in the `docs/` folder:

```text
docs/
├── doc_01.txt
├── doc_02.txt
├── doc_03.txt
├── doc_04.txt
├── doc_05.txt
├── doc_06.txt
├── doc_07.txt
└── doc_08.txt
```

The application loads these files and treats each short document as a chunk.

### 2. Embedding

`sentence-transformers` with `all-MiniLM-L6-v2` converts each document into numbers called **embeddings**.

These embeddings are stored locally in **ChromaDB**.

No API key is required.

### 3. Retrieval

When the user asks a policy question, the `retrieve_and_answer` LangGraph node searches ChromaDB.

The question is converted into an embedding and compared with the stored embeddings using **cosine similarity**.

The **top 3 most similar chunks** are retrieved.

### 4. Generation

The LangGraph has 3 main nodes:

```text
classify_intent
       ↓
   ┌───┴───┐
   ↓       ↓
retrieve  direct
_and      _answer
answer
```

`classify_intent` decides between:

* `policy_question`
* `general_question`

In the required mock mode, these keywords mean it is a policy question:

`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`

For a policy question, `retrieve_and_answer` retrieves the top 3 chunks.

For a general question, `direct_answer` returns:

```text
I can only answer questions about Zepto policies right now.
```

---

## MOCK_LLM

The application uses `MOCK_LLM` to control LLM usage.

By default:

```text
MOCK_LLM=1
```

or when the variable is not set, the application uses deterministic Python logic.

There is:

* no LLM API call
* no API key
* no signup
* no network LLM call

This is the **required graded mode**.

When `MOCK_LLM=0`, a real LLM can optionally be used for classification and answer generation.

**Retrieval always uses the real local embedding model and ChromaDB in both modes.**

---

## Structured Response

The final response is validated using Pydantic:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

`answer` contains the response.

`sources` contains the retrieved document IDs. It is empty for general questions.

`confidence` is between `0` and `1`. Mock mode uses `1.0`.

The prompt used for the optional real LLM follows:

**Role → Context → Task → Format → Length**

It also includes a negative rule such as:

> Do not answer using information that is not present in the provided context.

and includes a few-shot example.

---

## Run the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the document database:

```bash
python ingest.py
```

Run the API:

```bash
uvicorn main:app --reload
```

The API runs at:

```text
http://127.0.0.1:8000
```

Send a policy question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
-H "Content-Type: application/json" \
-d "{\"query\":\"How much is the delivery fee?\"}"
```

Send a general question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
-H "Content-Type: application/json" \
-d "{\"query\":\"What is the capital of India?\"}"
```

The first question uses retrieval. The second does not.

---

## Docker

Build:

```bash
docker build -t zepto-rag .
```

Run:

```bash
docker run -p 7860:7860 zepto-rag
```

The API is then available on port `7860`.

The complete pipeline is:

**Ingestion → Embedding → ChromaDB → Retrieval → LangGraph → Generation → Pydantic JSON → FastAPI**

The required implementation works locally and offline using `MOCK_LLM` mode.
