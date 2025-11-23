# AI Research Agent

A **RAG-style AI Web Researcher** built with **Streamlit**, **DuckDuckGo search**, and **Sentence Transformers**. The agent automates research by retrieving relevant passages from the web, ranking them with embeddings, and generating concise summaries in real-time.

---

## Features

* **Natural Language Query:** Ask questions in plain English.
* **Web Search Automation:** Searches multiple sources using DuckDuckGo.
* **Contextual Summarization:** Embeddings rank and summarize top passages.
* **Top Passage Retrieval:** Returns top 5 most relevant passages with URLs.
* **Streamlit Interface:** User-friendly web app to interact with the AI.

---

## Tech Stack

* **Python**
* **Streamlit** for web UI
* **DuckDuckGo Search API (`ddgs`)** for web scraping
* **BeautifulSoup** for HTML parsing
* **Sentence Transformers (`all-MiniLM-L6-v2`)** for embeddings
* **NumPy** for cosine similarity calculations

---

## Architecture Overview

```
          +---------------------+
          |   User Interface    |
          |     (Streamlit)     |
          +---------+-----------+
                    |
                    v
          +---------------------+
          |   Query Processing  |
          | - Input sanitization|
          | - Text preprocessing|
          +---------+-----------+
                    |
                    v
          +---------------------+
          |   Web Search Agent  |
          | - DuckDuckGo Search |
          | - URL extraction   |
          | - HTML text fetch  |
          +---------+-----------+
                    |
                    v
          +---------------------+
          |  Passage Processing |
          | - Chunking texts    |
          | - Sentence splitting|
          +---------+-----------+
                    |
                    v
          +---------------------+
          | Embedding & Ranking |
          | - Sentence embeddings|
          | - Cosine similarity |
          | - Top-N ranking     |
          +---------+-----------+
                    |
                    v
          +---------------------+
          |   Summarization     |
          | - Top sentences     |
          | - Concise summary   |
          +---------+-----------+
                    |
                    v
          +---------------------+
          |   Output to UI      |
          | - Summary           |
          | - Top passages      |
          | - Scores & URLs     |
          +---------------------+
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/AI-Research-Agent.git
cd AI-Research-Agent
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Usage

1. Enter your research question in the input box.
2. Click **Run Research**.
3. View the **extractive summary** and **top relevant passages** along with their sources.



