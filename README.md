# AI Companion with GraphRAG

A Streamlit-based AI assistant powered by Ollama, LangChain, and Neo4j. It supports document uploads (PDF, DOCX, CSV, TXT) for building knowledge graphs, debugging, code analysis, and more.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Setup Python Environment](#setup-python-environment)
5. [Install Requirements](#install-requirements)
6. [Configure Environment Variables](#configure-environment-variables)
7. [Setting up Neo4j Credentials](#setting-up-neo4j-credentials)
8. [Running the Application](#running-the-application)
9. [Usage Tips](#usage-tips)
10. [Project File Structure](#project-file-structure)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)
13. [License](#license)

---

## Project Overview

This project is an **AI Companion** built with Streamlit. It uses a local LLM (Ollama) together with LangChain and Neo4j to provide a developer assistant that can:

- Ingest documents (PDF, DOCX, CSV, TXT).
- Build a knowledge graph (Neo4j) from uploaded documents.
- Answer coding questions and help with debugging using GraphRAG.
- Analyze code, suggest improvements, and fetch relevant document context.

The goal is to run everything locally for privacy and speed.

---

## Features

- Upload documents in the sidebar to build a knowledge graph.
- GraphRAG-enabled search for better answers using Neo4j context.
- Chat-style interface to ask coding and debugging questions.
- Options to clear the graph or change the LLM model.

---

## Prerequisites

- **Python 3.10+**
- **Ollama** (local LLM server)
- **Neo4j Desktop** (local graph database)
- A terminal (Command Prompt / PowerShell on Windows or Terminal on macOS/Linux)

---

## Setup Python Environment

Follow these steps to set up a clean Python environment and run the app.

1. **Install Python**

   - Download and install Python 3.10 or newer: https://www.python.org/downloads/
   - Verify in a terminal:

   ```bash
   python --version
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   # from your project folder
   python -m venv venv

   # activate (Windows)
   venv\Scripts\activate

   # activate (macOS / Linux)
   source venv/bin/activate
   ```

   You will see `(venv)` in your prompt when the environment is active.

3. **Install Ollama** (local LLM server):

   - Download and install from: https://ollama.com/download
   - Start the Ollama server in a separate terminal:

   ```bash
   ollama serve
   ```

   - Pull models to the local Ollama server, for example:
   #### Make sure models are capable to call tools, funtionCall.
   ```bash
   ollama pull llama3.2:latest
   ```

   Use whichever model you prefer that is supported by your Ollama installation.

4. **Install Neo4j Desktop** (for the knowledge graph database):

   - Download Neo4j Desktop: https://neo4j.com/download-center/#desktop
   - Install and launch Neo4j Desktop.
   - Create a DBMS instance as described in the **Setting up Neo4j Credentials** section below.

---

## Install Requirements

1. Make sure your virtual environment is active.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

- If you encounter issues, update pip first:

```bash
pip install --upgrade pip
```

---

## Configure Environment Variables

Create a `.env` file in the project root to store Neo4j credentials. Ollama runs locally so no external API keys are required for it.

Add the following to `.env` (replace `your_password` with the password you set in Neo4j Desktop):

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

Keep this file private and do not commit it to public repositories.

---

## Setting up Neo4j Credentials

To obtain the values for the `.env` file, create a new DBMS instance in Neo4j Desktop and set a password.

**Create a New DBMS Instance**

- Open Neo4j Desktop.
- Click **Create instance** (center or top-right).
- Fill in the details:
  - **Instance name**: e.g., `MyLocalDB`
  - **Neo4j version**: select the recommended/latest (5.x series is common)
  - **Database user**: `neo4j`
  - **Password**: set a strong password and save it somewhere secure
- Click **Create** and wait for the instance to download and prepare.

**Start the Database Instance**

- In the instances list, find your new instance card.
- Click the **Play** button (▶) to start it.
- Wait until the status shows **Running**.
- Confirm the **URI** is `bolt://localhost:7687` on the Overview panel.

**Test Connection**

- Launch Neo4j Browser from Desktop and connect with:
  - URI: `bolt://localhost:7687`
  - User: `neo4j`
  - Password: the password you set

Update your `.env` file with the password used here.

---

## Running the Application

1. Ensure the following are running:
   - Your Python virtual environment is activated.
   - Ollama server is running (`ollama serve`).
   - Your Neo4j instance is started in Neo4j Desktop.

2. Place the application files in your project folder. Required files:

```
frontend.py
graph_rag.py

# plus any modules, assets, or helper scripts your app uses
```

3. Run the Streamlit app:

```bash
streamlit run frontend.py
```

4. Open your browser to:

```
http://localhost:8501
```

Streamlit will display the AI Code Companion UI.

---

## Usage Tips

- Upload documents via the sidebar to build the knowledge graph.
- Enable **GraphRAG** in the sidebar for enhanced, graph-aware responses.
- Ask coding questions in the chat input (for example: "Debug this Python error...").
- Use the options in the sidebar to clear the graph or switch models.
- Keep large files split if ingestion is slow; small chunks help embeddings and search.

---

## Project File Structure (example)

```
ai-code-companion/
├─ frontend.py
├─ graph_rag.py
├─ requirements.txt
├─ .env
├─ data/            # uploaded documents (optional)
├─ notebooks/       # experiments
└─ README.md
```

---

## Troubleshooting

- **Ollama server not running**: Make sure you started it with `ollama serve` in a separate terminal.
- **Neo4j connection error**: Check `.env` has correct `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD`. Make sure the DB is Running in Neo4j Desktop.
- **Python packages fail to install**: Update pip and retry: `pip install --upgrade pip` then `pip install -r requirements.txt`.
- **Streamlit not opening**: Check the terminal where you ran `streamlit run frontend.py` for errors. Visit `http://localhost:8501` manually.

If you see specific error messages, paste them into the chat for help.

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repo.
2. Create a feature branch.
3. Make changes and test locally.
4. Submit a pull request with a clear description of your changes.

---


If you want me to export this as a `README.md` file or adjust wording, tell me what to change.




### Use Custom System prompts to get better results.

