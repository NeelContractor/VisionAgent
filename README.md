# 🧿 Vision Agent (Local Image Analysis)

A lightweight **local AI vision agent** built with **Streamlit + LangGraph + Ollama** that analyzes images and answers user questions using a multi-step pipeline.

<img src="https://github.com/NeelContractor/VisionAgent/blob/main/images/demo.png" width="75%" height="75%">

## Features

* Upload any image (JPG, PNG, WebP, GIF)
* Multi-step reasoning pipeline:

  * **Vision** → image description (LLaVA)
  * **Research** → reasoning over description
  * **Writer** → clean final answer
* Fully **local (no API costs)**
* Clean modern UI with Streamlit

---

## Architecture

```
User Input
   │
   ▼
[ Vision Model (llava-phi3) ]
   │
   ▼
[ Research Agent (llama3.2) ]
   │
   ▼
[ Writer Agent (llama3.2) ]
   │
   ▼
Final Answer
```

Built using **LangGraph state machine**.

---

## Requirements

* Python 3.9+
* Ollama installed → [https://ollama.com](https://ollama.com)
* 8GB RAM recommended

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Ollama
ollama serve

# Pull models
ollama pull llava-phi3
ollama pull llama3.2:1b
```

---

## Run App

```bash
streamlit run frontend.py
```

---

## Project Structure

```
.
├── frontend.py   # Streamlit frontend
├── backend.py    # LangGraph pipeline
└── README.md
```

---

## How It Works

1. Image is converted to base64
2. Sent to **LLaVA (vision model)** via Ollama
3. Output is analyzed by **LLM (llama3.2)**
4. Final answer is generated and displayed

---
