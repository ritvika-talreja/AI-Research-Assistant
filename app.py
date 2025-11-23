import streamlit as st
import re
import urllib.parse
from ddgs import DDGS
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np
import time

# -----------------------------
# CONFIG
# -----------------------------
SEARCH_RESULTS = 6
PASSAGES_PER_PAGE = 4
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_PASSAGES = 5
SUMMARY_SENTENCES = 3
TIMEOUT = 8

# -----------------------------
# CACHE MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

# -----------------------------
# HELPERS
# -----------------------------
def unwrap_ddg(url):
    try:
        parsed = urllib.parse.urlparse(url)
        if "duckduckgo.com" in parsed.netloc:
            qs = urllib.parse.parse_qs(parsed.query)
            uddg = qs.get("uddg")
            if uddg:
                return urllib.parse.unquote(uddg[0])
    except Exception:
        pass
    return url

def search_web(query, max_results=SEARCH_RESULTS):
    urls = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("url")
            if not url:
                continue
            url = unwrap_ddg(url)
            urls.append(url)
    return urls

def fetch_text(url, timeout=TIMEOUT):
    headers = {"User-Agent": "Mozilla/5.0 (research-agent)"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers)
        if r.status_code != 200:
            return ""

        if "html" not in r.headers.get("content-type", "").lower():
            return ""

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer",
                         "svg", "iframe", "nav", "aside"]):
            tag.extract()

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join([p for p in paragraphs if p])

        if text.strip():
            return re.sub(r"\s+", " ", text).strip()

        meta = soup.find("meta", attrs={"name": "description"}) or \
               soup.find("meta", attrs={"property": "og:description"})

        if meta and meta.get("content"):
            return meta["content"].strip()

        if soup.title and soup.title.string:
            return soup.title.string.strip()

    except Exception:
        return ""

    return ""

def chunk_passages(text, max_words=120):
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def split_sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

# -----------------------------
# AGENT
# -----------------------------
class ShortResearchAgent:
    def __init__(self):
        self.embedder = load_model()

    def run(self, query):
        start = time.time()

        urls = search_web(query)

        docs = []
        for u in urls:
            txt = fetch_text(u)
            if not txt:
                continue

            chunks = chunk_passages(txt, max_words=120)
            for c in chunks[:PASSAGES_PER_PAGE]:
                docs.append({"url": u, "passage": c})

        if not docs:
            return {"query": query, "passages": [], "summary": "", "time": 0}

        texts = [d["passage"] for d in docs]
        emb_texts = self.embedder.encode(texts, convert_to_numpy=True)
        q_emb = self.embedder.encode([query], convert_to_numpy=True)[0]

        def cosine(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

        sims = [cosine(e, q_emb) for e in emb_texts]
        top_idx = np.argsort(sims)[::-1][:TOP_PASSAGES]

        top_passages = [
            {"url": docs[i]["url"], "passage": docs[i]["passage"], "score": float(sims[i])}
            for i in top_idx
        ]

        # -----------------------------
        # CLEAN EXTRACTIVE SUMMARY (1 passage, 1 source)
        # -----------------------------
        best_passage = top_passages[0]
        best_text = best_passage["passage"]
        best_url = best_passage["url"]

        sent_list = split_sentences(best_text)

        if len(sent_list) == 0:
            summary = "No summary could be generated."
        else:
            # Top 2 readable sentences
            summary_sentences = sent_list[:2]
            summary = " ".join(summary_sentences)

        summary = summary + f"\n\nSource: {best_url}"

        elapsed = time.time() - start

        return {
            "query": query,
            "passages": top_passages,
            "summary": summary,
            "time": elapsed
        }

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.markdown("""
<div style='text-align:center; margin-bottom:10px;'>
    <h1>🔍 AI Research Assistant</h1>
</div>
""", unsafe_allow_html=True)

query = st.text_input(
    "",
    placeholder="Enter your research question...",
    key="query_input"
)

if st.button("💡 Generate Insights", use_container_width=True):
    if not query.strip():
        st.error("Please enter a question.")
    else:
        agent = ShortResearchAgent()
        with st.spinner("Running research..."):
            result = agent.run(query)

        st.subheader("📌 Extractive Summary")
        st.success(result["summary"])

        st.subheader("📄 Top Passages")
        for p in result["passages"]:
            with st.expander(f"{p['url']}  |  Score: {p['score']:.3f}"):
                st.write(p["passage"])
