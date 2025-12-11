
# 🎥 Project Demo Video Here 👉 https://youtu.be/NCUYpHODJmE
# ACRR FOSS – Automated Compliance & Risk Reporter
````markdown
ACRR FOSS is an open-source **financial document analysis tool** that turns raw PDFs  
(like lender fee worksheets, loan estimates, or other mortgage docs) into:

- ✅ Searchable **vector indexes** (FAISS)
- ✅ **RAG-style Q&A** over a single document
- ✅ **Mortgage key-field extraction** (loan amount, rate, fees, etc.)
- ✅ **Multi-agent analysis** with CrewAI (overview, numeric checks, risk notes)
- ✅ A clean **Streamlit UI** to demo the whole pipeline end-to-end

All logic runs with **free, local tools**: Python, FastAPI, FAISS, Ollama, Streamlit.

---

## 🔧 Tech Stack

### **Backend**
- Python 3.11  
- FastAPI + Uvicorn  
- FAISS (vector store)  
- Local LLM via **Ollama** (Qwen, Mistral, LLaMA models)  
- CrewAI for multi-agent workflows  

### **Frontend**
- Streamlit (single-page UI)
- Guided pipeline tabs:
  1. Extract & Inspect  
  2. Index & Search  
  3. Q&A & Key Fields  
  4. Multi-Agent Analysis  
  5. Chat (Experimental)

### **PDF / Text Processing**
- PyMuPDF (`fitz`) for text extraction  
- Character-based text chunking  
- Layout analysis with bounding boxes (`layout_blocks`)  
- Optional layout-based key/value discovery  

---

## 📂 Repository Structure (simplified)

```text
acrr-foss/
├── app/
│   ├── api/                # FastAPI routes (RAG, key fields, multi-agent analysis)
│   ├── rag/                # Embeddings + FAISS wrapper + RAG pipeline
│   └── agents/             # CrewAI agents & orchestration
├── src/
│   └── extract/
│       ├── pdf_text.py     # Text extraction
│       ├── chunking.py     # Chunk generation
│       └── layout_blocks.py# Layout block extraction + KV finder
├── data/
│   ├── vector_stores/      # Stored FAISS indices (per doc_id)
│   └── layout_blocks/      # Stored layout JSON (per doc_id)
├── streamlit_app.py        # Main Streamlit UI
├── requirements.txt
└── README.md
````

---

## 🚀 Getting Started (Local Development)

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create & activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and run Ollama

```bash
ollama pull qwen2.5:latest
ollama serve
```

---

## 🧠 Running the Backend (FastAPI)

```bash
uvicorn app.api.main:app --reload --port 8000
```

FastAPI docs → [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🖥 Running the Streamlit Frontend

```bash
streamlit run streamlit_app.py --server.port 8502
```

App opens at → [http://localhost:8502](http://localhost:8502)

---

## 🧭 How the App Works (Step-by-Step)

### 1️⃣ Extract & Inspect

* Upload a PDF
* View:

  * Text per page
  * Combined text
  * Chunks (debug)
  * Layout blocks
* Optional: find values using spatial layout (“Total Loan Amount”, etc.)

---

### 2️⃣ Index & Search

* Build embeddings for all chunks
* Create a FAISS index
* Save vector store + layout JSON
* Test with semantic search (“loan amount”, “interest rate fees”, etc.)

---

### 3️⃣ Q&A & Mortgage Key Fields

Choose mode:

* **Native RAG**
* **CrewAI agent answering**

Ask questions such as:

* “What is the total loan amount?”
* “Are there discount points?”

Also provides a **key-fields JSON** (loan amount, rate, fees, escrows).

---

### 4️⃣ Multi-Agent Analysis (CrewAI)

Runs multiple agents:

* Overview
* Numeric checks
* Checklist checks
* Risk analysis
* Cross-validation

Each agent shows:

* Markdown output
* Contexts used
* Exportable combined report

---

### 5️⃣ Chat with the Document (Experimental)

* Ask follow-up questions
* Uses chosen mode (RAG or multi-agent)
* Maintains simple chat history

---


## ✅ Status & Future Ideas

### Completed

* [x] PDF → text & layout extraction
* [x] Chunking + FAISS indexing
* [x] Local-LLM RAG
* [x] Key-field extraction
* [x] Multi-agent financial analysis
* [x] Streamlit UI




