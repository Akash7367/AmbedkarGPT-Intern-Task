📘 AmbedkarGPT — Question Answering System (Python + LangChain + LLM)

AmbedkarGPT is an intelligent Question–Answering AI system built using LangChain, Python, and Chroma vector database.
It processes Dr. B.R. Ambedkar’s text content and generates meaningful and context-aware answers to user queries using LLMs.

This project was built as part of my internship assignment to demonstrate skills in:

LLM integration

Text embeddings

Vector databases

Retrieval-based QA

Python backend development

🚀 Features
✔ 1. Intelligent QA System

Uses embedding-based search to retrieve the most relevant content and answer user questions.

✔ 2. ChromaDB for Vector Storage

All text data (speech.txt) is converted into embeddings and stored in a Chroma vector DB for fast retrieval.

✔ 3. LangChain Pipeline

End-to-end pipeline for:

Document loading

Chunking

Embedding

Retrieval

LLM question answering

✔ 4. Modular Python Code

Clean project structure with separate files for:

Embedding

Query handling

Database management

🗂 Project Structure
AmbedkarGPT/
│── main.py                # Main program
│── requirements.txt       # All Python dependencies
│── speech.txt             # Ambedkar text data
│── .gitignore             # venv & DB ignored
│── chroma_db/ (ignored)   # Vector DB (auto created)
│── venv/ (ignored)        # Python virtual environment

💻 How to Run Locally
1️⃣ Create Virtual Environment
python -m venv venv

2️⃣ Activate Environment

Windows:

.\venv\Scripts\activate


Linux/Mac:

source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
python main.py

🧠 Tech Stack
Component	Technology
Language	Python
LLM Framework	LangChain
Vector DB	ChromaDB
Embeddings	Sentence Embeddings / OpenAI / Ollama
Backend	Python Scripts
Data Source	Ambedkar's speech.txt
📄 What This Project Demonstrates

This project showcases my skills in:

Working with large language models

Prompt engineering basics

Text processing & chunking

Building retrieval-augmented generation (RAG) systems

Git & GitHub workflow

Managing Python environments

Internship-level project delivery