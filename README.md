# AI Research Assistant

A **RAG-style AI Web Researcher** built with **Streamlit**, **DuckDuckGo Search**, and **Sentence Transformers**.
This project automates online research by retrieving high-quality passages from the web, ranking them using semantic similarity, and generating an **extractive summary** from the most relevant source.

---

# ⭐ Features

* **Natural Language Query** : Ask any research question — no special formatting required.

* **Automated Web Search** : Uses DuckDuckGo Search (ddgs) to gather multiple sources.

* **Extractive RAG Summaries** : Generates a clean summary using only the top-ranked passage.

* **Top Passage Retrieval** : Shows a single best passage determined by similarity score.

* **Minimal & Professional Streamlit UI** : Clean interface suitable for showcasing to recruiters.

---

# 🧠 Tech Stack

* Python
* Streamlit
* DuckDuckGo Search (ddgs)
* BeautifulSoup
* SentenceTransformers
* NumPy
* OpenAI / LLM for summarization

---

# 🏗️ Architecture Overview

```
        ┌────────────────────────┐
        │     Streamlit UI       │
        └───────────┬────────────┘
                    │ User Query
                    ▼
        ┌────────────────────────┐
        │   Query Preprocessor    │
        └───────────┬────────────┘
                    │
                    ▼
        ┌────────────────────────┐
        │   Web Search Module     │
        │  (DuckDuckGo + BS4)     │
        └───────────┬────────────┘
                    │ Raw Web Data
                    ▼
        ┌────────────────────────┐
        │  Passage Chunking       │
        │  & Sentence Splitting   │
        └───────────┬────────────┘
                    │ Passages
                    ▼
        ┌────────────────────────┐
        │  Embedding Generator    │
        │ (SentenceTransformers)  │
        └───────────┬────────────┘
                    │ Vectors
                    ▼
        ┌────────────────────────┐
        │ Similarity Ranker       │
        │ (Cosine Similarity)     │
        └───────────┬────────────┘
                    │ Best Passage
                    ▼
        ┌────────────────────────┐
        │ Extractive Summarizer   │
        │ (LLM or Rule-based)     │
        └───────────┬────────────┘
                    │ Final Output
                    ▼
        ┌────────────────────────┐
        │     UI Output Panel     │
        └────────────────────────┘
```

---

# 🖼️ Streamlit App 

```
![App Screenshot](path_to_your_screenshot.png)
```



# 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/AI-Research-Assistant.git
cd AI-Research-Assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```
