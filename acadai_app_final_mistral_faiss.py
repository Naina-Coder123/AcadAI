import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional FAISS support. The app still runs without these packages and falls back to TF-IDF.
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
except Exception:
    SentenceTransformer = None
    CrossEncoder = None

APP_TITLE = "AcadAI"
DEFAULT_TOP_K = 8  # Lightweight default for local/laptop use
DEFAULT_FAISS_DIR = os.getenv("FAISS_STORE_DIR", "./AcadAI_FAISS_STORE")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
DEFAULT_CANDIDATE_K = int(os.getenv("FAISS_CANDIDATE_K", "100"))
DEFAULT_MIN_HYBRID_SCORE = float(os.getenv("MIN_HYBRID_SCORE", "0.25"))
DEFAULT_CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
SUBJECT_KEYWORDS = {
    # Whole-B.Tech subject map. These are used only as soft signals when index.pkl has no true metadata.
    "CN": [
        "qos", "quality of service", "bandwidth", "latency", "jitter", "packet", "routing", "tcp", "udp",
        "ip", "network", "congestion", "subnet", "subnetting", "subnet mask", "cidr", "vlsm",
        "classless", "classful", "network address", "broadcast address", "host address", "default gateway",
        "ipv4", "ip address", "prefix", "slash notation", "supernetting", "lan", "wan", "osi", "dns", "http"
    ],
    "OS": ["operating system", "deadlock", "paging", "segmentation", "process", "thread", "scheduler", "memory management", "semaphore", "mutex", "cpu scheduling", "virtual memory"],
    "DBMS": ["dbms", "database", "sql", "normalization", "transaction", "acid", "primary key", "foreign key", "join", "relational", "schema", "er diagram", "indexing"],
    "DSA": ["algorithm", "data structure", "tree", "graph", "recursion", "dynamic programming", "stack", "queue", "array", "linked list", "sorting", "searching"],
    "PYTHON": ["python", "function", "def ", "list", "tuple", "dictionary", "pandas", "numpy", "regex", "regular expression"],
    "WEB": ["web technology", "html", "css", "javascript", "react", "node", "http", "dom", "browser", "frontend", "backend", "servlet", "php"],
    "SE": ["software engineering", "sdlc", "waterfall", "agile", "scrum", "requirement", "testing", "uml", "use case", "software design", "maintenance"],
    "ML": ["machine learning", "ml", "classification", "regression", "clustering", "neural", "svm", "decision tree", "reinforcement learning", "training", "feature"],
    "DWM": ["data warehousing", "data mining", "olap", "etl", "star schema", "snowflake", "association rule", "apriori", "classification", "clustering"],
    "DA": ["data analytics", "analytics", "visualization", "statistics", "mean", "median", "variance", "correlation", "dashboard", "power bi", "tableau"],
    "HV": ["human values", "ethics", "value education", "professional ethics", "harmony", "society", "human conduct"],
}

SUBJECT_QUERY_HINTS = {
    "CN": [
        "subnet", "subnetting", "cidr", "vlsm", "ip address", "ipv4", "subnet mask",
        "network address", "broadcast address", "host address", "qos", "routing", "tcp", "udp", "osi", "dns"
    ],
    "OS": ["operating system", "deadlock", "paging", "segmentation", "process", "thread", "semaphore", "scheduling", "virtual memory"],
    "DBMS": ["dbms", "database", "sql", "normalization", "transaction", "acid", "primary key", "foreign key", "join", "er diagram"],
    "DSA": ["algorithm", "data structure", "recursion", "tree", "graph", "dynamic programming", "array", "linked list", "sorting"],
    "PYTHON": ["python", "regex", "pandas", "numpy", "dictionary", "tuple", "list comprehension"],
    "WEB": ["web technology", "html", "css", "javascript", "react", "node", "dom", "http", "frontend", "backend"],
    "SE": ["software engineering", "sdlc", "agile", "waterfall", "uml", "testing", "requirement", "scrum"],
    "ML": ["machine learning", "classification", "regression", "clustering", "neural network", "svm", "decision tree"],
    "DWM": ["data warehouse", "data warehousing", "data mining", "olap", "etl", "star schema", "apriori"],
    "DA": ["data analytics", "visualization", "statistics", "correlation", "dashboard"],
    "HV": ["human values", "ethics", "harmony", "professional ethics"],
}

SUBJECT_EXPANSIONS = {
    "CN": "computer networks subnetting numericals ip addressing ipv4 cidr vlsm subnet mask network address broadcast address usable hosts host range prefix length class a class b class c classful classless host bits borrowed bits number of subnets hosts per subnet routing qos bandwidth latency jitter packet loss osi dns tcp udp",
    "OS": "operating systems memory management process scheduling deadlock paging segmentation threads synchronization semaphores virtual memory cpu scheduling",
    "DBMS": "database management system sql normalization transaction acid primary key foreign key joins relational schema er diagram indexing",
    "DSA": "data structures algorithms complexity recursion graph tree dynamic programming stack queue array linked list sorting searching",
    "PYTHON": "python programming functions data types regex list tuple dictionary pandas numpy",
    "WEB": "web technology html css javascript dom browser http client server frontend backend react node servlet php",
    "SE": "software engineering sdlc waterfall agile scrum requirements testing uml use case software design maintenance",
    "ML": "machine learning classification regression clustering neural network svm decision tree training feature model evaluation",
    "DWM": "data warehousing data mining etl olap star schema snowflake schema association rule apriori classification clustering",
    "DA": "data analytics statistics visualization correlation dashboard business intelligence power bi tableau",
    "HV": "human values ethics harmony society professional ethics value education",
}



@dataclass
class Chunk:
    doc_id: str
    source: str
    page: int
    text: str


@dataclass
class AgentTrace:
    agent: str
    action: str
    result: str
    latency: float = 0.0


# ── Demo corpus ────────────────────────────────────────────────────────────────

DEMO_CHUNKS = [
    Chunk("dbms_notes.pdf::12", "dbms_notes.pdf", 12,
          "Normalization in DBMS is a design technique that organizes relations to reduce "
          "redundancy and avoid update, insertion, and deletion anomalies. First normal form "
          "removes repeating groups. Second normal form removes partial dependency. Third normal "
          "form removes transitive dependency."),
    Chunk("dbms_notes.pdf::13", "dbms_notes.pdf", 13,
          "A primary key uniquely identifies each row in a relation. A foreign key is an "
          "attribute in one relation that references the primary key of another relation and "
          "helps maintain referential integrity."),
    Chunk("os_notes.pdf::8", "os_notes.pdf", 8,
          "Deadlock is a condition where a set of processes are blocked because each process "
          "is holding a resource and waiting for another resource held by another process. "
          "The necessary conditions are mutual exclusion, hold and wait, no preemption, and "
          "circular wait."),
    Chunk("os_notes.pdf::18", "os_notes.pdf", 18,
          "Paging is a memory management scheme that divides logical memory into fixed-size "
          "pages and physical memory into frames. It removes external fragmentation and uses "
          "a page table to translate logical addresses into physical addresses."),
    Chunk("dsa_notes.pdf::21", "dsa_notes.pdf", 21,
          "Recursion is a programming technique where a function calls itself to solve smaller "
          "instances of the same problem. A recursive solution needs a base case and a recursive "
          "case, such as factorial n = n times factorial n minus 1."),
    Chunk("python_notes.pdf::5", "python_notes.pdf", 5,
          "Python functions are defined using the def keyword. A recursive function calls itself. "
          "Example: def factorial(n): return 1 if n<=1 else n*factorial(n-1). "
          "Base case prevents infinite recursion."),
    Chunk("os_notes.pdf::30", "os_notes.pdf", 30,
          "Process scheduling algorithms include FCFS (First Come First Served), SJF (Shortest "
          "Job First), Round Robin with time quantum, and Priority Scheduling. Round Robin is "
          "widely used in time-sharing systems."),
    Chunk("dbms_notes.pdf::40", "dbms_notes.pdf", 40,
          "SQL joins: INNER JOIN returns matching rows. LEFT JOIN returns all from left plus "
          "matches. RIGHT JOIN returns all from right plus matches. FULL OUTER JOIN returns all "
          "rows from both tables."),
]


# ── Utility ────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def quote(text: str, limit: int = 260) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + "..."


def keyword_overlap(query: str, text: str) -> float:
    query_terms = {t for t in tokenize(query) if len(t) > 2}
    if not query_terms:
        return 0.0
    return len(query_terms & set(tokenize(text))) / len(query_terms)


# ── Document Ingestion Pipeline ────────────────────────────────────────────────

def split_text(text: str, source: str, page: int,
               chunk_size: int = 512, overlap: int = 64) -> List[Chunk]:
    text = clean_text(text)
    if not text:
        return []
    chunks, start, idx = [], 0, 1
    while start < len(text):
        part = text[start: start + chunk_size]
        if len(part) > 60:
            chunks.append(Chunk(f"{source}::{page}.{idx}", source, page, part))
        start += max(1, chunk_size - overlap)
        idx += 1
    return chunks


def read_pdf_uploads(files) -> Tuple[List[Chunk], List[str]]:
    chunks, skipped = [], []
    if not files:
        return chunks, skipped
    try:
        from pypdf import PdfReader
    except Exception:
        return [], ["PDF upload needs pypdf. Run: pip install pypdf"]
    for file in files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                path = tmp.name
            reader = PdfReader(path)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                chunks.extend(split_text(text, file.name, page_num))
        except Exception as exc:
            skipped.append(f"{file.name}: {type(exc).__name__}: {exc}")
    return chunks, skipped



# ── FAISS Store Loading ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str):
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_cross_encoder(model_name: str):
    if CrossEncoder is None:
        return None
    try:
        return CrossEncoder(model_name)
    except Exception:
        return None


def _doc_to_chunk(doc, fallback_id: str, source: str = "faiss_store") -> Chunk:
    """Convert common LangChain Document-like objects into our Chunk dataclass."""
    text = getattr(doc, "page_content", None) or getattr(doc, "text", None) or str(doc)
    meta = getattr(doc, "metadata", {}) or {}
    return Chunk(
        str(meta.get("doc_id", fallback_id)),
        str(meta.get("source", source)),
        int(meta.get("page", 0) or 0),
        clean_text(text),
    )


def _extract_chunks_from_pickle(obj) -> List[Chunk]:
    """Supports common FAISS pickle formats, including LangChain's index.pkl."""
    chunks: List[Chunk] = []

    # LangChain FAISS usually stores: (docstore, index_to_docstore_id)
    if isinstance(obj, tuple) and len(obj) >= 2:
        docstore, index_to_docstore_id = obj[0], obj[1]
        docs = getattr(docstore, "_dict", None)
        if isinstance(docs, dict) and isinstance(index_to_docstore_id, dict):
            for i in sorted(index_to_docstore_id):
                doc_id = index_to_docstore_id[i]
                doc = docs.get(doc_id)
                if doc is not None:
                    chunks.append(_doc_to_chunk(doc, str(doc_id)))
            return chunks

    # A direct list of Document/Chunk/dict items.
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, Chunk):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("page_content") or item.get("text") or item.get("content") or ""
                chunks.append(Chunk(
                    str(item.get("doc_id", f"faiss::{i}")),
                    str(item.get("source", "faiss_store")),
                    int(item.get("page", 0) or 0),
                    clean_text(text),
                ))
            else:
                chunks.append(_doc_to_chunk(item, f"faiss::{i}"))
        return [c for c in chunks if c.text]

    # A dict containing texts/chunks/documents.
    if isinstance(obj, dict):
        items = obj.get("chunks") or obj.get("documents") or obj.get("docs") or obj.get("texts")
        if isinstance(items, list):
            return _extract_chunks_from_pickle(items)

    return chunks


@st.cache_resource(show_spinner=False)
def load_faiss_store(store_dir: str):
    if faiss is None:
        return None, [], "FAISS package missing. Install: pip install faiss-cpu"

    index_path = os.path.join(store_dir, "index.faiss")
    pkl_path = os.path.join(store_dir, "index.pkl")
    if not os.path.exists(index_path) or not os.path.exists(pkl_path):
        return None, [], f"Could not find index.faiss and index.pkl inside: {store_dir}"

    try:
        index = faiss.read_index(index_path)
        import pickle
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        chunks = _extract_chunks_from_pickle(obj)
        if not chunks:
            return index, [], "FAISS index loaded, but index.pkl text metadata format was not recognized."
        return index, chunks, ""
    except Exception as exc:
        return None, [], f"FAISS load failed: {type(exc).__name__}: {exc}"


# ── Improved Hybrid Retrieval helpers ─────────────────────────────────────────


def infer_subject_from_text(text: str, source: str = "") -> str:
    """Infer broad academic subject from chunk source + text when metadata lacks subject."""
    hay = f"{source} {text}".lower()
    best_subject, best_score = "GENERAL", 0
    for subject, kws in SUBJECT_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in hay)
        if score > best_score:
            best_subject, best_score = subject, score
    return best_subject if best_score > 0 else "GENERAL"


def detect_query_subjects(query: str) -> List[str]:
    """Detect likely subject(s) from user query for lightweight metadata filtering."""
    q = query.lower()
    subjects = []
    for subject, kws in SUBJECT_QUERY_HINTS.items():
        if any(kw in q for kw in kws):
            subjects.append(subject)
    # Subnetting/IP numericals are always Computer Networks, even if the word "network" is absent.
    if any(x in q for x in ["subnet", "subnetting", "cidr", "vlsm", "subnet mask", "broadcast address"]):
        if "CN" not in subjects:
            subjects.insert(0, "CN")
    return subjects


def is_subnetting_query(query: str) -> bool:
    q = query.lower()
    return any(x in q for x in [
        "subnet", "subnetting", "cidr", "vlsm", "subnet mask", "broadcast address",
        "network address", "host bits", "hosts per subnet", "borrow bits", "prefix length",
        "class a", "class b", "class c", "ip addressing"
    ])


def cn_keyword_score(text: str) -> float:
    tokens = [
        "subnet", "subnetting", "cidr", "vlsm", "subnet mask", "network address",
        "broadcast address", "host", "hosts", "ip address", "ipv4", "prefix",
        "classful", "classless", "routing", "network", "bits", "borrow"
    ]
    hay = text.lower()
    return sum(1 for t in tokens if t in hay) / max(1, len(tokens))


def source_subject_boost(source: str, wanted_subjects: List[str]) -> float:
    """Boost candidate when filename/folder name hints match the query subject."""
    s = (source or "").lower()
    source_aliases = {
        "CN": ["cn", "network", "computer network", "data communication", "communication"],
        "OS": ["os", "operating", "operating system"],
        "DBMS": ["dbms", "database", "sql"],
        "DSA": ["dsa", "data structure", "algorithm"],
        "PYTHON": ["python"],
        "WEB": ["web", "html", "css", "javascript"],
        "SE": ["software engineering", "se", "sdlc"],
        "ML": ["machine learning", "ml", "techniques"],
        "DWM": ["warehouse", "mining", "dwm"],
        "DA": ["analytics"],
        "HV": ["human values", "ethics"],
    }
    score = 0.0
    for subj in wanted_subjects:
        if any(alias in s for alias in source_aliases.get(subj, [])):
            score += 0.12
    return min(score, 0.24)


def keyword_fallback_search(query: str, chunks: List[Chunk], wanted_subjects: List[str], top_k: int = 30) -> List[Dict]:
    """No-rebuild fallback: exact/TF-IDF search over all chunks when semantic FAISS misses the topic."""
    expanded = expand_query(query)
    q_terms = [t for t in tokenize(expanded) if len(t) > 2]
    q_set = set(q_terms)
    rows = []

    for idx, c in enumerate(chunks):
        subject = infer_subject_from_text(c.text, c.source)
        text_l = c.text.lower()
        source_boost = source_subject_boost(c.source, wanted_subjects)
        subject_boost = 0.20 if wanted_subjects and subject in wanted_subjects else 0.0

        exact_hits = sum(1 for t in q_set if t in text_l)
        phrase_hits = 0
        for phrase in ["subnet mask", "network address", "broadcast address", "ip address",
                       "primary key", "foreign key", "operating system", "machine learning",
                       "software engineering", "data warehouse"]:
            if phrase in expanded.lower() and phrase in text_l:
                phrase_hits += 3

        if exact_hits == 0 and phrase_hits == 0 and source_boost == 0:
            continue

        overlap = keyword_overlap(expanded, c.text)
        subject_quality = 0.0
        if subject in wanted_subjects:
            subject_quality += 0.20
        if is_subnetting_query(query):
            subject_quality += 0.35 * cn_keyword_score(c.text)

        score = exact_hits + phrase_hits + (10 * overlap) + (10 * source_boost) + (10 * subject_quality)
        rows.append((score, idx, c, subject, overlap))

    rows.sort(reverse=True, key=lambda x: x[0])
    results = []
    for rank, (score, idx, c, subject, overlap) in enumerate(rows[:top_k], 1):
        results.append({
            "rank": rank,
            "doc_id": c.doc_id,
            "source": c.source,
            "page": c.page,
            "subject": subject,
            "score": float(score),
            "dense_norm": 0.0,
            "lexical": round(float(score), 4),
            "hybrid_score": round(float(score / 10.0), 4),
            "overlap": round(float(overlap), 4),
            "chunk_index": idx,
            "evidence": quote(c.text, 420),
            "text": c.text,
            "retrieval_mode": "keyword_fallback",
        })
    return results


def filter_candidates_by_subject(candidate_rows, query: str, mode: str = "auto", manual_subject: str = "Any"):
    """Soft metadata filtering. Auto keeps detected subjects; manual can force one subject."""
    if mode == "off":
        return candidate_rows, "Subject filter off"
    wanted = []
    if manual_subject and manual_subject != "Any":
        wanted = [manual_subject]
    elif mode == "auto":
        wanted = detect_query_subjects(query)
    if not wanted:
        return candidate_rows, "No subject detected"
    filtered = []
    for row in candidate_rows:
        _, _, c = row
        subject = infer_subject_from_text(c.text, c.source)
        if subject in wanted:
            filtered.append(row)
    # Safety: if filter is too strict, keep original so app never returns empty context.
    # For CN subnetting/IP queries, prefer even a small CN-only set over random cross-subject chunks.
    q = query.lower()
    is_strong_cn_query = "CN" in wanted and is_subnetting_query(query)
    if len(filtered) >= 1 and is_strong_cn_query:
        return filtered, f"Strong CN filter applied: {', '.join(wanted)}"
    if len(filtered) >= 3:
        return filtered, f"Filtered to subject(s): {', '.join(wanted)}"
    return candidate_rows, f"Subject filter too strict; kept all candidates. Detected: {', '.join(wanted)}"


ACADEMIC_SYNONYMS = {
    "qos": "quality of service bandwidth latency jitter packet loss traffic prioritization",
    "subnet": "subnetting ip addressing cidr vlsm subnet mask network address broadcast address usable hosts host range prefix length",
    "subnetting": "ip addressing cidr vlsm subnet mask network address broadcast address usable hosts host range prefix length classful classless",
    "cidr": "classless inter domain routing prefix length subnet mask network address broadcast address host range",
    "vlsm": "variable length subnet mask subnetting cidr host requirement network allocation",
    "os": "operating system memory management process scheduling deadlock paging segmentation",
    "dbms": "database management system sql normalization transaction acid keys joins",
    "cn": "computer networks tcp ip routing congestion control quality of service subnetting cidr vlsm",
    "dsa": "data structures algorithms complexity recursion graph tree dynamic programming",
}


def expand_query(query: str) -> str:
    """Lightweight query expansion for academic acronyms and common CS terms."""
    terms = tokenize(query)
    extras = []
    for t in terms:
        if t in ACADEMIC_SYNONYMS:
            extras.append(ACADEMIC_SYNONYMS[t])
    # Also expand explicit acronym patterns and detected subject-specific concepts.
    q = query.lower()
    if "quality of service" in q:
        extras.append(ACADEMIC_SYNONYMS["qos"])
    for subject in detect_query_subjects(query):
        extras.append(SUBJECT_EXPANSIONS.get(subject, ""))
    return clean_text(query + " " + " ".join(extras))


def llm_query_expansion(query: str) -> str:
    """Optional Mistral query expansion. Falls back silently when API is unavailable."""
    system = (
        "Expand this academic search query for retrieval. Return only a compact keyword-rich query, "
        "no explanation. Include synonyms, full forms, and related CS terms."
    )
    expanded = call_llm(query, system)
    expanded = clean_text(expanded)
    if not expanded or len(expanded) > 500:
        return query
    return clean_text(query + " " + expanded)


def model_query_text(query: str, model_name: str) -> str:
    """E5 models perform best with a 'query:' prefix."""
    if "e5" in model_name.lower() and not query.lower().startswith("query:"):
        return "query: " + query
    return query


def normalize_dense_scores(raw_scores: np.ndarray, index) -> np.ndarray:
    """Convert FAISS scores/distances to 0..1 where higher means better."""
    scores = raw_scores.astype("float32")
    if len(scores) == 0:
        return scores
    metric_type = getattr(index, "metric_type", None)
    # Inner product/cosine-like indexes: larger is better. L2 indexes: smaller is better.
    if faiss is not None and metric_type == faiss.METRIC_INNER_PRODUCT:
        mn, mx = float(scores.min()), float(scores.max())
        return (scores - mn) / (mx - mn + 1e-9)
    inv = 1.0 / (1.0 + np.maximum(scores, 0))
    mn, mx = float(inv.min()), float(inv.max())
    return (inv - mn) / (mx - mn + 1e-9)


def lexical_scores(query: str, texts: List[str]) -> np.ndarray:
    """TF-IDF score over only FAISS candidates; helps reject semantically-close but wrong chunks."""
    if not texts:
        return np.array([], dtype="float32")
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
        matrix = vectorizer.fit_transform([query] + texts)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel().astype("float32")
        return sims
    except Exception:
        return np.zeros(len(texts), dtype="float32")


def lexical_subject_scan(query: str, chunks: List[Chunk], wanted_subjects: List[str], limit: int = 80):
    """Fallback retrieval: scan all chunks lexically when dense FAISS misses subject-specific terms."""
    pool = []
    for idx, c in enumerate(chunks):
        subject = infer_subject_from_text(c.text, c.source)
        if wanted_subjects and subject not in wanted_subjects:
            continue
        if is_subnetting_query(query) and subject == "CN" and cn_keyword_score(c.text) <= 0:
            continue
        pool.append((idx, c))
    if not pool:
        return []
    texts = [c.text for _, c in pool]
    scores = lexical_scores(expand_query(query), texts)
    order = np.argsort(scores)[::-1][:limit]
    rows = []
    for rank, oi in enumerate(order, 1):
        idx, c = pool[int(oi)]
        score = float(scores[int(oi)])
        if score <= 0:
            continue
        rows.append({
            "rank": rank, "doc_id": c.doc_id, "source": c.source, "page": c.page,
            "subject": infer_subject_from_text(c.text, c.source), "score": score,
            "dense_norm": 0.0, "lexical": round(score, 4), "hybrid_score": round(score, 4),
            "overlap": round(keyword_overlap(expand_query(query), c.text), 4),
            "chunk_index": idx, "evidence": quote(c.text, 420), "text": c.text,
            "retrieval_mode": "lexical_subject_fallback",
        })
    return rows


def expand_neighbor_context(results: List[Dict], chunks: List[Chunk], max_extra_chars: int = 900) -> List[Dict]:
    """Append adjacent chunks from the same source/page when available for better answer context."""
    improved = []
    for r in results:
        idx = int(r.get("chunk_index", -1))
        texts = [r["text"]]
        base = chunks[idx] if 0 <= idx < len(chunks) else None
        for nidx in (idx - 1, idx + 1):
            if base and 0 <= nidx < len(chunks):
                n = chunks[nidx]
                same_source = n.source == base.source
                close_page = abs((n.page or 0) - (base.page or 0)) <= 1
                if same_source and close_page and n.text:
                    texts.append(n.text[:max_extra_chars])
        r = dict(r)
        r["text"] = clean_text(" ".join(texts))
        r["evidence"] = quote(r["text"], 420)
        improved.append(r)
    return improved


def cross_encoder_rerank(query: str, rows: List[Dict], model_name: str) -> List[Dict]:
    """Re-rank selected candidates using a cross encoder. Higher score = better."""
    reranker = get_cross_encoder(model_name)
    if reranker is None or not rows:
        return rows
    try:
        pairs = [[query, r["text"]] for r in rows]
        ce_scores = reranker.predict(pairs)
        for r, s in zip(rows, ce_scores):
            r["cross_score"] = round(float(s), 4)
        rows.sort(key=lambda x: x.get("cross_score", -999), reverse=True)
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        return rows
    except Exception:
        return rows


def retrieve_faiss(query: str, index, chunks: List[Chunk], model_name: str,
                   top_k: int = DEFAULT_TOP_K, candidate_k: int = DEFAULT_CANDIDATE_K,
                   min_hybrid_score: float = DEFAULT_MIN_HYBRID_SCORE,
                   use_hybrid_rerank: bool = True,
                   use_cross_encoder: bool = False,
                   cross_encoder_model: str = DEFAULT_CROSS_ENCODER_MODEL,
                   subject_filter_mode: str = "auto",
                   manual_subject: str = "Any",
                   use_llm_expansion: bool = False,
                   parent_context_chars: int = 900):
    model = get_embedding_model(model_name)
    if model is None:
        return [], {"match": False, "reason": "SentenceTransformer missing or model failed to load", "overlap": 0.0}
    try:
        expanded = expand_query(query)
        if use_llm_expansion:
            expanded = llm_query_expansion(expanded)

        q_emb = model.encode([model_query_text(expanded, model_name)], normalize_embeddings=True).astype("float32")
        if q_emb.shape[1] != index.d:
            return [], {
                "match": False,
                "reason": f"Embedding dimension mismatch: model gives {q_emb.shape[1]}, FAISS index needs {index.d}. Use the same embedding model used while creating the index.",
                "overlap": 0.0,
            }

        # Search a larger pool because subject filtering + reranking need room.
        search_k = max(top_k, min(max(candidate_k * 2, candidate_k), len(chunks)))
        raw_scores, raw_ids = index.search(q_emb, search_k)
        candidate_rows = []
        candidate_texts = []
        valid_scores = []

        for pos, idx in enumerate(raw_ids[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            c = chunks[int(idx)]
            if not c.text:
                continue
            candidate_rows.append((pos, int(idx), c))
            candidate_texts.append(c.text)
            valid_scores.append(float(raw_scores[0][pos]))

        if not candidate_rows:
            return [], {"match": False, "reason": "FAISS returned no valid candidates", "overlap": 0.0}

        candidate_rows, subject_note = filter_candidates_by_subject(
            candidate_rows, query, mode=subject_filter_mode, manual_subject=manual_subject
        )
        candidate_rows = candidate_rows[:candidate_k]
        candidate_texts = [c.text for _, _, c in candidate_rows]
        valid_scores = [float(raw_scores[0][pos]) for pos, _, _ in candidate_rows]

        dense_norm = normalize_dense_scores(np.array(valid_scores, dtype="float32"), index)
        lex = lexical_scores(expanded, candidate_texts)
        overlaps = np.array([keyword_overlap(expanded, t) for t in candidate_texts], dtype="float32")

        required_subjects_for_boost = detect_query_subjects(query)
        hybrid_scores = []
        for i in range(len(candidate_rows)):
            _, _, c_for_boost = candidate_rows[i]
            if use_hybrid_rerank:
                hybrid = 0.45 * float(dense_norm[i]) + 0.40 * float(lex[i]) + 0.15 * float(overlaps[i])
            else:
                hybrid = float(dense_norm[i])

            # No-rebuild improvements: source/folder boost + subject-specific keyword boost.
            hybrid += source_subject_boost(c_for_boost.source, required_subjects_for_boost)
            if is_subnetting_query(query):
                hybrid += 0.25 * cn_keyword_score(candidate_texts[i])
            hybrid_scores.append(hybrid)

        order = np.argsort(np.array(hybrid_scores))[::-1]
        pre_selected = []
        seen_keys = set()
        for cand_i in order:
            pos, idx, c = candidate_rows[int(cand_i)]
            subject = infer_subject_from_text(c.text, c.source)
            key = (c.source, c.page, c.text[:90])
            if key in seen_keys:
                continue
            # Reject weak chunks when the query has clear subject-specific keywords.
            # This prevents subnetting queries from returning Python/ML/math chunks.
            required_subjects = detect_query_subjects(query)
            is_subject_specific = bool(required_subjects)
            current_overlap = float(overlaps[int(cand_i)])
            current_lexical = float(lex[int(cand_i)])
            if is_subject_specific and subject not in required_subjects and current_overlap < 0.15 and current_lexical < 0.05:
                continue
            if "CN" in required_subjects and any(x in query.lower() for x in ["subnet", "subnetting", "cidr", "vlsm", "ip address"]):
                if subject != "CN" and current_overlap < 0.18:
                    continue
            seen_keys.add(key)
            pre_selected.append({
                "rank": len(pre_selected) + 1,
                "doc_id": c.doc_id,
                "source": c.source,
                "page": c.page,
                "subject": subject,
                "score": float(valid_scores[int(cand_i)]),
                "dense_norm": round(float(dense_norm[int(cand_i)]), 4),
                "lexical": round(float(lex[int(cand_i)]), 4),
                "hybrid_score": round(float(hybrid_scores[int(cand_i)]), 4),
                "overlap": round(float(overlaps[int(cand_i)]), 4),
                "chunk_index": idx,
                "evidence": quote(c.text, 420),
                "text": c.text,
            })
            if len(pre_selected) >= max(top_k * 2, top_k):
                break

        # If dense FAISS + filters miss, do an exact keyword fallback over the existing index.pkl chunks.
        if not pre_selected:
            fallback_rows = lexical_subject_scan(query, chunks, detect_query_subjects(query), limit=max(candidate_k, 80))
            keyword_rows = keyword_fallback_search(query, chunks, detect_query_subjects(query), top_k=max(candidate_k, 80))
            merged = fallback_rows + keyword_rows
            if merged:
                # Deduplicate and keep strongest fallback rows.
                seen_fb = set()
                deduped = []
                for row in sorted(merged, key=lambda r: float(r.get("hybrid_score", 0)), reverse=True):
                    k = (row.get("source"), row.get("page"), row.get("text", "")[:90])
                    if k in seen_fb:
                        continue
                    seen_fb.add(k)
                    deduped.append(row)
                pre_selected = deduped[:max(top_k * 2, top_k)]
                subject_note += " | keyword/lexical fallback used"
            else:
                return [], {"match": False, "reason": f"No sufficiently relevant chunks after subject/keyword filtering. {subject_note}", "overlap": 0.0}

        # If the best dense result is weak, mix in keyword fallback before reranking.
        if pre_selected:
            best_overlap_now = float(pre_selected[0].get("overlap", 0))
            best_hybrid_now = float(pre_selected[0].get("hybrid_score", 0))
            if best_overlap_now < 0.08 or best_hybrid_now < min_hybrid_score:
                keyword_rows = keyword_fallback_search(query, chunks, detect_query_subjects(query), top_k=40)
                if keyword_rows:
                    pre_selected = pre_selected + keyword_rows
                    seen_mix = set()
                    mixed = []
                    for row in sorted(pre_selected, key=lambda r: float(r.get("hybrid_score", 0)), reverse=True):
                        k = (row.get("source"), row.get("page"), row.get("text", "")[:90])
                        if k in seen_mix:
                            continue
                        seen_mix.add(k)
                        mixed.append(row)
                    pre_selected = mixed[:max(top_k * 3, top_k)]
                    subject_note += " | weak-result keyword fallback mixed"

        if use_cross_encoder:
            pre_selected = cross_encoder_rerank(expanded, pre_selected, cross_encoder_model)

        if is_subnetting_query(query):
            cn_rows = [r for r in pre_selected if r.get("subject") == "CN" or cn_keyword_score(r.get("text", "")) > 0]
            if len(cn_rows) >= 1:
                pre_selected = cn_rows + [r for r in pre_selected if r not in cn_rows]

        selected = pre_selected[:top_k]
        selected = expand_neighbor_context(selected, chunks, max_extra_chars=parent_context_chars)

        combined = " ".join(r["text"] for r in selected[:min(5, len(selected))])
        overlap = keyword_overlap(expanded, combined)
        best_hybrid = float(selected[0].get("hybrid_score", 0.0)) if selected else 0.0
        best_raw = float(selected[0].get("score", 0.0)) if selected else 0.0
        best_cross = selected[0].get("cross_score", "off") if selected else "off"
        match = bool(selected) and (best_hybrid >= min_hybrid_score or overlap >= 0.08)
        reason = (
            f"Advanced FAISS: raw={best_raw:.3f}, hybrid={best_hybrid:.3f}, "
            f"cross={best_cross}, overlap={overlap:.0%}, candidates={len(candidate_rows)}. {subject_note}"
        )
        return selected, {
            "match": match,
            "reason": reason,
            "overlap": overlap,
            "best_score": best_raw,
            "best_hybrid": best_hybrid,
            "expanded_query": expanded,
        }
    except Exception as exc:
        return [], {"match": False, "reason": f"FAISS retrieval failed: {type(exc).__name__}: {exc}", "overlap": 0.0}

# ── Retriever ──────────────────────────────────────────────────────────────────

def build_index(chunks: List[Chunk]):
    corpus = [c.text for c in chunks]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=12000)
    mat = vec.fit_transform(corpus)
    return vec, mat


def retrieve(query: str, chunks: List[Chunk], top_k: int = DEFAULT_TOP_K):
    if not chunks:
        return [], {"match": False, "reason": "No chunks available", "overlap": 0.0}
    vec, mat = build_index(chunks)
    q_vec = vec.transform([query])
    sims = cosine_similarity(q_vec, mat).ravel()
    ranked = np.argsort(sims)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(ranked, 1):
        c = chunks[int(idx)]
        results.append({
            "rank": rank, "doc_id": c.doc_id, "source": c.source, "page": c.page,
            "score": float(sims[int(idx)]),
            "overlap": keyword_overlap(query, c.text),
            "evidence": quote(c.text),
            "text": c.text,
        })
    best = float(sims[int(ranked[0])]) if ranked.size else 0.0
    combined = " ".join(r["text"] for r in results[:3])
    overlap = keyword_overlap(query, combined)
    match = best >= 0.18 or overlap >= 0.34
    return results, {"match": match,
                     "reason": f"best score {best:.2f}, overlap {overlap:.0%}",
                     "overlap": overlap, "best_score": best}


# ── Web Search ─────────────────────────────────────────────────────────────────

def web_search(query: str, top_k: int = 5) -> List[Dict]:
    rows = []
    try:
        r = requests.post("https://duckduckgo.com/html/", data={"q": query},
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for res in soup.select(".result"):
            ta = res.select_one(".result__a")
            ts = res.select_one(".result__snippet")
            if not ta:
                continue
            rows.append({"doc_id": f"web::{len(rows)+1}",
                         "title": clean_text(ta.get_text(" ")),
                         "url": ta.get("href") or "",
                         "snippet": clean_text(ts.get_text(" ") if ts else "")})
            if len(rows) >= top_k:
                break
    except Exception as exc:
        rows.append({"doc_id": "web::err", "title": "DuckDuckGo unavailable",
                     "url": "", "snippet": str(exc)})
    useful = [r for r in rows if r["doc_id"] not in {"web::err"}]
    if len(useful) >= top_k:
        return useful[:top_k]
    try:
        wiki = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "utf8": 1},
            headers={"User-Agent": "AcadAI/1.0"}, timeout=10)
        wiki.raise_for_status()
        for item in wiki.json().get("query", {}).get("search", []):
            title = clean_text(item.get("title", ""))
            snippet = clean_text(
                BeautifulSoup(item.get("snippet", ""), "html.parser").get_text(" "))
            if not title:
                continue
            rows.append({"doc_id": f"web::{len(rows)+1}", "title": title,
                         "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                         "snippet": snippet})
            if len(rows) >= top_k:
                break
    except Exception:
        pass
    useful = [r for r in rows if r["doc_id"] not in {"web::err"}]
    return useful[:top_k] if useful else rows[:top_k]


# ── LLM caller ─────────────────────────────────────────────────────────────────

def call_llm(prompt: str, system: str = "") -> str:
    """Call only Mistral AI. Returns empty string if MISTRAL_API_KEY is missing or the API fails."""
    mistral_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_key:
        return ""

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        r = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {mistral_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": os.getenv("MISTRAL_MODEL", "mistral-large-latest"),
                "temperature": 0.1,
                "messages": messages,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return ""


# ── Agent 1: Router ────────────────────────────────────────────────────────────

def router_agent(query: str, db_match: bool,
                 use_web: bool) -> Tuple[str, AgentTrace]:
    t0 = time.time()
    q_lower = query.lower()
    realtime_kw = ["today", "latest", "current", "2024", "2025", "recent", "news", "price"]
    general_kw = ["who is", "what year", "define ", "meaning of", "capital of",
                  "how many", "convert "]
    if any(p in q_lower for p in realtime_kw) and use_web:
        route = "Web Search"
    elif db_match:
        route = "RAG"
    elif any(p in q_lower for p in general_kw):
        route = "Direct LLM"
    elif use_web:
        route = "Web Search"
    else:
        route = "RAG"
    trace = AgentTrace("Router Agent",
                       "Classified query intent → Direct LLM / Web Search / RAG",
                       f"Route → {route}",
                       time.time() - t0)
    return route, trace


# ── Agent 2: Reasoning ─────────────────────────────────────────────────────────

def reasoning_agent(query: str) -> Tuple[Dict, AgentTrace]:
    t0 = time.time()
    system = (
        "You are a Reasoning Agent in an intelligent tutoring system. "
        "Analyse the student query and return ONLY valid JSON with keys: "
        "key_concepts (list), solution_plan (list of steps), tools_needed (list), "
        "difficulty_estimate (beginner/intermediate/advanced). No markdown fences."
    )
    raw = call_llm(query, system)
    plan = {}
    if raw:
        try:
            plan = json.loads(re.sub(r"```json|```", "", raw).strip())
        except Exception:
            plan = {"key_concepts": [], "solution_plan": [raw[:200]],
                    "tools_needed": [], "difficulty_estimate": "intermediate"}
    if not plan:
        words = [w for w in tokenize(query) if len(w) > 3]
        plan = {
            "key_concepts": words[:4],
            "solution_plan": ["Retrieve relevant context",
                              "Synthesise step-by-step explanation",
                              "Add worked examples",
                              "Cite source documents"],
            "tools_needed": ["RAG retriever", "Tutor Agent"],
            "difficulty_estimate": "intermediate",
        }
    trace = AgentTrace(
        "Reasoning Agent",
        "Identified key concepts, built solution plan, detected required tools",
        f"Concepts: {plan.get('key_concepts', [])} | "
        f"{len(plan.get('solution_plan', []))} plan steps",
        time.time() - t0)
    return plan, trace


# ── Agent 3: Tutor ─────────────────────────────────────────────────────────────

def tutor_agent(query: str, difficulty: str,
                context_rows: List[Dict], web_rows: List[Dict],
                route: str, plan: Dict) -> Tuple[str, AgentTrace]:
    t0 = time.time()
    if route == "RAG":
        evidence = "\n\n".join(f"[{r['doc_id']}] {r['evidence']}" for r in context_rows)
    elif route == "Web Search":
        evidence = "\n\n".join(
            f"[{r['doc_id']}] {r['title']}. {r['snippet']}" for r in web_rows)
    else:
        evidence = "Use your general knowledge."

    system = (
        "You are AcadAI's Tutor Agent — a pedagogically expert AI tutor. "
        "Generate a well-structured academic answer using ONLY the evidence provided. "
        "If evidence is weak or partially relevant, clearly say what is missing instead of guessing. "
        "Structure: (1) Concept explanation, (2) Step-by-step breakdown with worked "
        "examples, (3) Exam-oriented tips, (4) Explicit source citations. "
        "Adapt depth to the requested difficulty level."
    )
    concepts = ", ".join(plan.get("key_concepts", []))
    prompt = (
        f"Difficulty: {difficulty}\n"
        f"Key concepts: {concepts}\n"
        f"Student query: {query}\n\n"
        f"Evidence:\n{evidence}"
    )
    answer = call_llm(prompt, system)
    if not answer:
        if route == "RAG" and context_rows:
            lines = [f"**Answer ({difficulty}) — Grounded from course materials**\n"]
            for r in context_rows[:3]:
                lines.append(f"- **[{r['doc_id']}]** {r['evidence']}\n")
            lines.append("\n**Key concepts:** " + concepts)
            lines.append("\n**Cited:** " + ", ".join(
                f"`{r['doc_id']}`" for r in context_rows))
            answer = "\n".join(lines)
        elif route == "Web Search" and web_rows:
            lines = [f"**Answer ({difficulty}) — Web fallback**\n",
                     "_Verify before submitting academic work._\n"]
            for r in web_rows[:3]:
                lines.append(
                    f"- **{r['title']}**: {r['snippet']}  \n  {r['url']}\n")
            answer = "\n".join(lines)
        else:
            answer = ("NOT_FOUND: No reliable evidence. "
                      "Upload relevant PDFs or enable web fallback.")
    trace = AgentTrace(
        "Tutor Agent",
        "Generated pedagogical answer (step-by-step, examples, citations)",
        f"Route={route} | {len(context_rows or web_rows)} evidence chunks",
        time.time() - t0)
    return answer, trace


# ── Agent 4: Critic ────────────────────────────────────────────────────────────

def critic_agent(query: str, answer: str) -> Tuple[Dict, AgentTrace]:
    t0 = time.time()
    system = (
        "You are AcadAI's Critic Agent. Evaluate strictly on four dimensions. "
        "Return ONLY valid JSON with keys: relevance (0-10), completeness (0-10), "
        "accuracy (0-10), clarity (0-10), overall (0-10), "
        "satisfactory (true if overall>=7 else false), "
        "feedback (improvement note if not satisfactory, else empty string). No markdown."
    )
    raw = call_llm(f"Query: {query}\n\nAnswer:\n{answer[:1500]}", system)
    scores = {}
    if raw:
        try:
            scores = json.loads(re.sub(r"```json|```", "", raw).strip())
        except Exception:
            pass
    if not scores:
        words = len(answer.split())
        has_example = any(kw in answer.lower()
                          for kw in ["example", "e.g.", "for instance", "such as"])
        has_cite = "[" in answer and "]" in answer
        rel = min(10.0, 5 + keyword_overlap(query, answer) * 10)
        comp = min(10.0, 4 + words / 80)
        acc = 7.5
        cla = 7.0 + (1.0 if has_example else 0) + (0.5 if has_cite else 0)
        overall = round((rel + comp + acc + cla) / 4, 1)
        scores = {"relevance": round(rel, 1), "completeness": round(comp, 1),
                  "accuracy": round(acc, 1), "clarity": round(cla, 1),
                  "overall": overall, "satisfactory": overall >= 7.0,
                  "feedback": "" if overall >= 7.0
                  else "Add more examples and explicit source citations."}
    trace = AgentTrace(
        "Critic Agent",
        "Evaluated Relevance, Completeness, Accuracy, Clarity",
        f"Overall: {scores.get('overall','?')}/10 | "
        f"Satisfactory: {scores.get('satisfactory','?')}",
        time.time() - t0)
    return scores, trace


def refine_answer(query: str, answer: str,
                  feedback: str, difficulty: str) -> str:
    system = (
        "You are AcadAI's Tutor Agent in a refinement pass. "
        "Improve the answer based on the Critic Agent's feedback. "
        "Keep all correct content; address only the stated weaknesses."
    )
    refined = call_llm(
        f"Original answer:\n{answer}\n\nCritic feedback:\n{feedback}\n\n"
        f"Query: {query}\nDifficulty: {difficulty}",
        system)
    return refined if refined else answer + f"\n\n_Refinement note: {feedback}_"


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(route: str, evidence_count: int, latency: float,
                    scores: Dict) -> Dict:
    return {
        "Answer route": route,
        "Relevance": f"{scores.get('relevance', '—')}/10",
        "Completeness": f"{scores.get('completeness', '—')}/10",
        "Accuracy": f"{scores.get('accuracy', '—')}/10",
        "Clarity": f"{scores.get('clarity', '—')}/10",
        "Overall quality": f"{scores.get('overall', '—')}/10",
        "Evidence used": evidence_count,
        "Needs review": "No" if scores.get("satisfactory") else "Yes",
        "Response time": f"{latency:.2f}s",
    }


def save_history(event: Dict):
    st.session_state.setdefault("history", []).append(
        {"time": datetime.now().strftime("%H:%M:%S"), **event})


# ── Conversation Memory + Grounding + Viva Helpers ───────────────────────────

def init_learning_state():
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("quiz_questions", "")
    st.session_state.setdefault("quiz_topic", "")
    st.session_state.setdefault("quiz_rows", [])
    st.session_state.setdefault("student_profile", {
        "name": "Student",
        "semester": "B.Tech",
        "branch": "CSE / AIML",
        "preferred_level": "intermediate",
        "goal": "exam + interview preparation",
    })
    st.session_state.setdefault("weak_topics", {})
    st.session_state.setdefault("quiz_attempts", [])
    st.session_state.setdefault("saved_flashcards", [])
    st.session_state.setdefault("saved_roadmaps", [])


def build_memory_context(max_turns: int = 4) -> str:
    history = st.session_state.get("chat_history", [])[-max_turns:]
    blocks = []
    for i, item in enumerate(history, 1):
        blocks.append(
            f"Turn {i}: Student asked: {item.get('query','')}\n"
            f"AcadAI answered: {quote(item.get('answer',''), 450)}\n"
            f"Subject: {item.get('subject','GENERAL')} | Grounding: {item.get('grounding','—')}%"
        )
    return "\n\n".join(blocks)


def store_conversation_turn(query: str, answer: str, route: str, db_rows: List[Dict], grounding_score: float):
    subject = "GENERAL"
    if db_rows:
        subject = str(db_rows[0].get("subject", "GENERAL"))
    st.session_state.setdefault("chat_history", []).append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "query": query,
        "answer": answer,
        "route": route,
        "subject": subject,
        "grounding": round(float(grounding_score), 1),
    })
    st.session_state["chat_history"] = st.session_state["chat_history"][-30:]


def answer_sentences(answer: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", answer or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [p.strip() for p in parts if len(p.strip().split()) >= 5]


def calculate_grounding_report(answer: str, evidence_rows: List[Dict]) -> Dict:
    evidence_text = " ".join(str(r.get("text") or r.get("evidence") or "") for r in evidence_rows)
    evidence_text = clean_text(evidence_text)
    sents = answer_sentences(answer)
    if not sents or not evidence_text:
        return {"score": 0.0, "supported": 0, "total": len(sents), "unsupported": sents[:5], "status": "No evidence"}

    supported, unsupported = 0, []
    for sent in sents:
        overlap = keyword_overlap(sent, evidence_text)
        sent_terms = [t for t in tokenize(sent) if len(t) > 3]
        evidence_hits = sum(1 for t in set(sent_terms) if t in evidence_text.lower())
        support_ratio = evidence_hits / max(1, len(set(sent_terms)))
        if overlap >= 0.18 or support_ratio >= 0.28:
            supported += 1
        else:
            unsupported.append(sent)

    score = round((supported / max(1, len(sents))) * 100, 1)
    status = "Strongly grounded" if score >= 75 else "Partially grounded" if score >= 45 else "Weakly grounded"
    return {"score": score, "supported": supported, "total": len(sents), "unsupported": unsupported[:6], "status": status}


def generate_quiz(topic: str, difficulty: str, evidence_rows: List[Dict]) -> str:
    evidence = "\n\n".join(f"[{r.get('doc_id')}] {r.get('evidence','')}" for r in evidence_rows[:8])
    system = (
        "You are AcadAI's Viva/Quiz Agent. Generate exactly 5 viva-style questions for a B.Tech student. "
        "Mix conceptual, applied, and one tricky follow-up. Return numbered questions only."
    )
    prompt = f"Topic: {topic}\nDifficulty: {difficulty}\nEvidence:\n{evidence}"
    out = call_llm(prompt, system)
    if out:
        return out
    return "\n".join([
        f"1. Define {topic} in your own words.",
        f"2. Explain the main steps or components involved in {topic}.",
        f"3. Give one real-world or exam-oriented example of {topic}.",
        f"4. What is one common mistake students make in {topic}?",
        f"5. How would you compare {topic} with a related concept?",
    ])


def evaluate_quiz_answer(topic: str, questions: str, student_answer: str, evidence_rows: List[Dict]) -> str:
    evidence = "\n\n".join(f"[{r.get('doc_id')}] {r.get('evidence','')}" for r in evidence_rows[:8])
    system = (
        "You are AcadAI's Viva Critic Agent. Evaluate the student's answer fairly. "
        "Give: score out of 10, strengths, missing points, corrected answer, and next practice suggestion. "
        "Use the provided evidence when possible and avoid unsupported claims."
    )
    prompt = f"Topic: {topic}\nQuestions:\n{questions}\n\nStudent answer:\n{student_answer}\n\nEvidence:\n{evidence}"
    out = call_llm(prompt, system)
    if out:
        return out
    return "**Score:** Not available without Mistral.\n\nAdd your Mistral API key to get detailed viva evaluation."


def profile_summary() -> str:
    profile = st.session_state.get("student_profile", {})
    weak = st.session_state.get("weak_topics", {})
    weak_sorted = sorted(weak.items(), key=lambda x: x[1], reverse=True)[:6]
    weak_text = ", ".join(f"{k} ({v})" for k, v in weak_sorted) if weak_sorted else "None tracked yet"
    return (
        f"Student: {profile.get('name','Student')} | Semester: {profile.get('semester','B.Tech')} | "
        f"Branch: {profile.get('branch','CSE')} | Level: {profile.get('preferred_level','intermediate')} | "
        f"Goal: {profile.get('goal','exam preparation')} | Weak topics: {weak_text}"
    )


def update_weak_topic(topic: str, amount: int = 1):
    topic = clean_text(topic or "General")[:80]
    weak = st.session_state.setdefault("weak_topics", {})
    weak[topic] = int(weak.get(topic, 0)) + amount


def parse_score_out_of_10(text: str) -> Optional[float]:
    patterns = [r"score\s*[:=]\s*(\d+(?:\.\d+)?)\s*/\s*10", r"(\d+(?:\.\d+)?)\s*/\s*10"]
    for pat in patterns:
        m = re.search(pat, text or "", flags=re.I)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                return None
    return None


def adaptive_difficulty_from_attempts(default_level: str) -> str:
    attempts = st.session_state.get("quiz_attempts", [])[-5:]
    scores = [a.get("score") for a in attempts if isinstance(a.get("score"), (int, float))]
    if len(scores) < 2:
        return default_level
    avg = sum(scores) / len(scores)
    if avg >= 8:
        return "advanced"
    if avg <= 5:
        return "beginner"
    return "intermediate"


def generate_learning_roadmap(topic: str, days: int, difficulty: str, evidence_rows: List[Dict]) -> str:
    evidence = "\n\n".join(f"[{r.get('doc_id')}] {r.get('evidence','')}" for r in evidence_rows[:10])
    system = (
        "You are AcadAI's Learning Roadmap Agent. Create a practical day-wise B.Tech study roadmap. "
        "Use the student profile and evidence. Include daily topics, practice, revision, and mini-tests."
    )
    prompt = f"Student profile: {profile_summary()}\nTopic: {topic}\nDays: {days}\nDifficulty: {difficulty}\nEvidence:\n{evidence}"
    out = call_llm(prompt, system)
    if out:
        return out
    return "\n".join([f"Day {i}: Study {topic} subtopic {i}, make notes, solve 5 questions, revise mistakes." for i in range(1, days + 1)])


def generate_flashcards(topic: str, evidence_rows: List[Dict], count: int = 10) -> str:
    evidence = "\n\n".join(f"[{r.get('doc_id')}] {r.get('evidence','')}" for r in evidence_rows[:10])
    system = (
        "You are AcadAI's Flashcard Agent. Create concise exam flashcards. "
        "Return in Q/A format. Keep answers short and accurate."
    )
    prompt = f"Topic: {topic}\nNumber of flashcards: {count}\nEvidence:\n{evidence}"
    out = call_llm(prompt, system)
    if out:
        return out
    return "\n".join([f"Q{i}. What is an important point about {topic}?\nA{i}. Review the retrieved evidence and write the definition/example." for i in range(1, count + 1)])


def generate_revision_notes(topic: str, mode: str, evidence_rows: List[Dict]) -> str:
    evidence = "\n\n".join(f"[{r.get('doc_id')}] {r.get('evidence','')}" for r in evidence_rows[:12])
    system = (
        "You are AcadAI's Exam Revision Agent. Create high-yield B.Tech revision material. "
        "Use only the evidence when possible. Include formulas, definitions, key points, and likely questions."
    )
    prompt = f"Topic: {topic}\nRevision mode: {mode}\nStudent profile: {profile_summary()}\nEvidence:\n{evidence}"
    out = call_llm(prompt, system)
    if out:
        return out
    return f"**{mode} for {topic}**\n\n- Key definitions\n- Important points\n- Common exam questions\n- Practice examples"


def generate_exam_questions(topic: str, evidence_rows: List[Dict]) -> str:
    evidence = "\n\n".join(f"[{r.get('doc_id')}] {r.get('evidence','')}" for r in evidence_rows[:10])
    system = (
        "You are AcadAI's PYQ/Exam Question Agent. Generate likely B.Tech exam questions. "
        "Group as 2-mark, 5-mark, 10-mark, and viva questions."
    )
    prompt = f"Topic: {topic}\nEvidence:\n{evidence}"
    out = call_llm(prompt, system)
    if out:
        return out
    return f"**Likely questions for {topic}:**\n\n2-mark: Define key terms.\n5-mark: Explain with example.\n10-mark: Discuss complete concept with diagram/steps."


def retrieve_for_tool(topic: str, k: int = 8):
    if use_faiss and faiss_index is not None and faiss_chunks:
        return retrieve_faiss(
            topic, faiss_index, chunks, embedding_model_name,
            top_k=k, candidate_k=min(candidate_k, 120),
            min_hybrid_score=min_hybrid_score,
            use_hybrid_rerank=use_hybrid_rerank,
            use_cross_encoder=False,
            cross_encoder_model=cross_encoder_model_name,
            subject_filter_mode=subject_filter_mode,
            manual_subject=manual_subject,
            use_llm_expansion=use_llm_expansion,
            parent_context_chars=parent_context_chars,
        )
    return retrieve(topic, chunks, top_k=k)


# ── UI helpers ─────────────────────────────────────────────────────────────────

def metric_card(label: str, value: str, tone: str = "neutral"):
    st.markdown(
        f'<div class="metric-card metric-{tone}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True)


def status_chip(label: str, value: str, tone: str = "neutral"):
    st.markdown(
        f"<span class='status-chip chip-{tone}'>"
        f"<span>{label}</span><strong>{value}</strong></span>",
        unsafe_allow_html=True)


def section_header(title: str, caption: str = ""):
    st.markdown(
        f'<div class="section-heading"><h3>{title}</h3><p>{caption}</p></div>',
        unsafe_allow_html=True)


def agent_badge(name: str, action: str, result: str,
                latency: float, tone: str = "info"):
    st.markdown(
        f'<div class="agent-card agent-{tone}">'
        f'<div class="agent-name">{name}</div>'
        f'<div class="agent-action">{action}</div>'
        f'<div class="agent-result">{result}</div>'
        f'<div class="agent-latency">{latency:.2f}s</div></div>',
        unsafe_allow_html=True)


# ── Page config & CSS ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AcadAI · AI Learning Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AcadAI is a multi-agent AI learning platform with RAG, Mistral, FAISS retrieval, viva mode, grounding checks, roadmap generation, and revision tools."
    },
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
  --app-bg:#f6f8fb;
  --panel:#ffffff;
  --panel-2:#f9fbfd;
  --ink:#111827;
  --ink-2:#1f2937;
  --muted:#6b7280;
  --muted-2:#94a3b8;
  --line:#e5e7eb;
  --line-2:#dbe3ee;
  --primary:#0f766e;
  --primary-2:#0ea5e9;
  --navy:#0f172a;
  --success:#059669;
  --warning:#d97706;
  --danger:#dc2626;
  --blue:#2563eb;
  --violet:#7c3aed;
  --shadow-xs:0 1px 2px rgba(15,23,42,.05);
  --shadow-sm:0 8px 24px rgba(15,23,42,.06);
  --shadow-md:0 18px 50px rgba(15,23,42,.10);
}

html, body, [class*="css"]{
  font-family:'Inter',system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  color:var(--ink);
}
[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(circle at 100% 0%, rgba(14,165,233,.08), transparent 28%),
    radial-gradient(circle at 0% 4%, rgba(15,118,110,.06), transparent 30%),
    linear-gradient(180deg,#fbfdff 0%,var(--app-bg) 45%,#eef3f8 100%);
}
.main .block-container{padding-top:1.05rem;max-width:1240px;padding-bottom:3.5rem;}
#MainMenu, footer, header {visibility:hidden;}

/* Sidebar: enterprise control panel */
div[data-testid="stSidebar"]{
  background:#ffffff;
  border-right:1px solid var(--line);
  box-shadow:8px 0 28px rgba(15,23,42,.035);
}
div[data-testid="stSidebar"] [data-testid="stSidebarContent"]{padding-top:1rem;}
div[data-testid="stSidebar"] h1,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3{
  color:var(--ink);font-weight:850;letter-spacing:-.03em;
}
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] p{color:#475569;font-size:13px;}
div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{line-height:1.55;}
div[data-testid="stSidebar"] hr{margin:1.05rem 0;border-color:#eef2f7;}
[data-testid="stFileUploader"] section{
  border-radius:16px;border:1px dashed #a7c7f5;background:#f8fbff;
  padding:14px;transition:border .2s ease, background .2s ease;
}
[data-testid="stFileUploader"] section:hover{border-color:#2563eb;background:#f3f8ff;}

/* Top product shell */
.product-shell{margin-bottom:16px;}
.product-topbar{
  display:flex;align-items:center;justify-content:space-between;gap:18px;
  margin-bottom:14px;padding:12px 16px;border:1px solid var(--line);
  background:rgba(255,255,255,.86);backdrop-filter:blur(14px);
  border-radius:22px;box-shadow:var(--shadow-sm);
}
.brand-lockup{display:flex;align-items:center;gap:12px;}
.logo-mark{
  width:42px;height:42px;border-radius:14px;
  background:linear-gradient(135deg,#0f766e,#2563eb 70%,#7c3aed);
  display:flex;align-items:center;justify-content:center;color:white;font-weight:900;
  box-shadow:0 10px 26px rgba(37,99,235,.18);
}
.brand-title{font-size:18px;font-weight:900;letter-spacing:-.035em;color:var(--ink);line-height:1.1;}
.brand-subtitle{font-size:12px;color:var(--muted);font-weight:650;margin-top:2px;}
.topbar-right{display:flex;gap:8px;align-items:center;flex-wrap:wrap;justify-content:flex-end;}
.topbar-chip{
  display:inline-flex;gap:7px;align-items:center;border:1px solid #dbeafe;background:#eff6ff;
  color:#1d4ed8;border-radius:999px;padding:8px 11px;font-size:12px;font-weight:800;
}
.topbar-chip.success{border-color:#bbf7d0;background:#ecfdf5;color:#047857;}
.topbar-chip.dark{border-color:#1e293b;background:#0f172a;color:#e2e8f0;}

/* Executive hero */
.hero{
  position:relative;overflow:hidden;border-radius:28px;padding:30px 32px;
  border:1px solid rgba(15,23,42,.10);background:
    linear-gradient(135deg,#0f172a 0%,#13223e 46%,#0f766e 100%);
  color:white;box-shadow:var(--shadow-md);
}
.hero:after{
  content:"";position:absolute;right:-90px;top:-110px;width:360px;height:360px;
  background:radial-gradient(circle,rgba(14,165,233,.45),transparent 68%);
}
.hero:before{
  content:"";position:absolute;left:35%;bottom:-170px;width:420px;height:300px;
  background:radial-gradient(circle,rgba(124,58,237,.30),transparent 70%);
}
.hero-grid{position:relative;z-index:1;display:grid;grid-template-columns:1.25fr .75fr;gap:28px;align-items:center;}
.eyebrow{
  display:inline-flex;align-items:center;gap:9px;font-size:11px;letter-spacing:.13em;text-transform:uppercase;
  color:#99f6e4;font-weight:900;margin-bottom:12px;
}
.eyebrow:before{content:"";width:8px;height:8px;border-radius:50%;background:#34d399;box-shadow:0 0 0 6px rgba(52,211,153,.18);}
.hero h1{font-size:44px;line-height:1.02;margin:0 0 13px;letter-spacing:-.055em;font-weight:900;color:white;}
.hero p{font-size:15px;line-height:1.7;color:rgba(255,255,255,.82);max-width:740px;margin:0;}
.hero-actions{display:flex;gap:10px;flex-wrap:wrap;margin-top:22px;}
.hero-pill{
  display:inline-flex;align-items:center;gap:8px;border:1px solid rgba(255,255,255,.14);
  background:rgba(255,255,255,.10);backdrop-filter:blur(12px);color:#f8fafc;border-radius:999px;
  padding:9px 13px;font-size:12px;font-weight:800;
}
.pipeline{
  border:1px solid rgba(255,255,255,.14);border-radius:22px;background:rgba(255,255,255,.08);
  backdrop-filter:blur(16px);padding:14px;display:grid;gap:10px;
}
.pipeline-step{
  display:grid;grid-template-columns:auto 1fr;gap:11px;align-items:center;
  padding:12px;border-radius:16px;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.10);
}
.pipeline-index{
  width:28px;height:28px;border-radius:10px;background:rgba(52,211,153,.16);color:#a7f3d0;
  display:flex;align-items:center;justify-content:center;font-weight:900;font-size:12px;
}
.pipeline-main{font-weight:900;font-size:13px;color:#fff;}
.pipeline-sub{font-weight:650;font-size:11px;color:rgba(226,232,240,.74);margin-top:2px;}

/* Professional feature cards */
.feature-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:16px 0 18px;}
.feature-card{
  border:1px solid var(--line);background:rgba(255,255,255,.92);border-radius:22px;padding:18px;
  box-shadow:var(--shadow-sm);min-height:128px;transition:transform .18s ease, box-shadow .18s ease, border .18s ease;
}
.feature-card:hover{transform:translateY(-3px);box-shadow:var(--shadow-md);border-color:#cbd5e1;}
.feature-icon{
  width:38px;height:38px;border-radius:14px;display:flex;align-items:center;justify-content:center;
  background:#f1f5f9;font-size:19px;margin-bottom:14px;
}
.feature-card h4{font-size:14px;font-weight:900;letter-spacing:-.025em;margin:0 0 7px;color:var(--ink);}
.feature-card p{font-size:12px;line-height:1.55;color:var(--muted);margin:0;}

/* Status chips */
.status-chip{
  display:inline-flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 12px;border-radius:999px;
  border:1px solid var(--line);background:rgba(255,255,255,.9);backdrop-filter:blur(10px);color:var(--muted);
  font-size:12px;margin:0 7px 8px 0;box-shadow:var(--shadow-xs);
}
.status-chip strong{color:var(--ink);font-size:12px;font-weight:900;}
.chip-ok{background:#ecfdf5;border-color:#bbf7d0;color:#047857;}
.chip-warn{background:#fff7ed;border-color:#fed7aa;color:#b45309;}
.chip-info{background:#eff6ff;border-color:#bfdbfe;color:#1d4ed8;}

/* Cards and metrics */
.metric-card{
  border:1px solid var(--line);border-radius:18px;padding:16px;background:rgba(255,255,255,.94);
  box-shadow:var(--shadow-sm);min-height:94px;transition:transform .16s ease, box-shadow .16s ease;
}
.metric-card:hover{transform:translateY(-2px);box-shadow:var(--shadow-md);}
.metric-ok{border-color:#bbf7d0;background:linear-gradient(180deg,#fff,#ecfdf5);}
.metric-warn{border-color:#fed7aa;background:linear-gradient(180deg,#fff,#fff7ed);}
.metric-info{border-color:#bfdbfe;background:linear-gradient(180deg,#fff,#eff6ff);}
.metric-purple{border-color:#ddd6fe;background:linear-gradient(180deg,#fff,#f5f3ff);}
.metric-label{color:var(--muted);font-size:10px;text-transform:uppercase;font-weight:900;letter-spacing:.08em;}
.metric-value{color:var(--ink);font-size:21px;font-weight:900;margin-top:8px;overflow-wrap:anywhere;letter-spacing:-.03em;}

.agent-card{border:1px solid var(--line);border-left:5px solid var(--blue);border-radius:18px;background:rgba(255,255,255,.94);padding:14px 15px;margin:7px 0;box-shadow:var(--shadow-sm);}
.agent-ok{border-left-color:var(--success);background:linear-gradient(180deg,#fff,#ecfdf5);}
.agent-warn{border-left-color:var(--warning);background:linear-gradient(180deg,#fff,#fff7ed);}
.agent-purple{border-left-color:var(--violet);background:linear-gradient(180deg,#fff,#f5f3ff);}
.agent-name{font-weight:900;font-size:11px;color:var(--ink);text-transform:uppercase;letter-spacing:.07em;}
.agent-action{font-size:13px;color:var(--muted);margin:5px 0;line-height:1.45;}
.agent-result{font-size:13px;color:var(--ink);font-weight:750;line-height:1.45;}
.agent-latency{font-size:11px;color:var(--muted);margin-top:6px;font-weight:800;}

.section-heading{margin:20px 0 11px;}
.section-heading h3{font-size:20px;margin:0 0 3px;font-weight:900;letter-spacing:-.035em;color:var(--ink);}
.section-heading p{color:var(--muted);font-size:12px;margin:0;}

/* Streamlit widgets */
.stTabs [data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid var(--line);padding-bottom:0;}
.stTabs [data-baseweb="tab"]{
  height:44px;border:1px solid var(--line);border-bottom:none;border-radius:14px 14px 0 0;
  background:rgba(255,255,255,.75);padding:10px 16px;font-weight:800;color:#475569;
}
.stTabs [aria-selected="true"]{background:#fff;color:var(--primary);box-shadow:0 -4px 16px rgba(15,23,42,.04);}
.stButton>button{
  border-radius:14px;font-weight:900;border:0;background:linear-gradient(135deg,#0f766e,#2563eb)!important;
  color:white!important;box-shadow:0 13px 28px rgba(37,99,235,.22);transition:transform .16s ease, box-shadow .16s ease;
}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 18px 38px rgba(37,99,235,.28);}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea{
  border-radius:14px!important;border:1px solid var(--line)!important;background:#fff!important;
  box-shadow:var(--shadow-xs)!important;
}
[data-testid="stDataFrame"]{border-radius:16px;overflow:hidden;box-shadow:var(--shadow-sm);}
div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:20px!important;box-shadow:var(--shadow-sm);background:rgba(255,255,255,.92);}
.streamlit-expanderHeader{font-weight:900;color:var(--ink);background:#fff;border-radius:12px;}

/* Answer readability */
[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3{letter-spacing:-.03em;}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{line-height:1.65;}

@media(max-width:980px){.hero-grid{grid-template-columns:1fr}.feature-grid{grid-template-columns:1fr 1fr}.hero h1{font-size:36px}.product-topbar{align-items:flex-start;flex-direction:column}}
@media(max-width:640px){.feature-grid{grid-template-columns:1fr}.hero{padding:24px}.hero h1{font-size:32px}.topbar-right{justify-content:flex-start}}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="product-shell">
  <div class="product-topbar">
    <div class="brand-lockup">
      <div class="logo-mark">A</div>
      <div>
        <div class="brand-title">AcadAI</div>
        <div class="brand-subtitle">AI learning platform for B.Tech knowledge workflows</div>
      </div>
    </div>
    <div class="topbar-right">
      <span class="topbar-chip success">● Local RAG Ready</span>
      <span class="topbar-chip">FAISS + Mistral</span>
      <span class="topbar-chip dark">Academic Copilot</span>
    </div>
  </div>

  <div class="hero">
    <div class="hero-grid">
      <div>
        <div class="eyebrow">AI-LMS · Multi-Agent RAG · B.Tech Knowledge Assistant</div>
        <h1>Study, revise, test, and verify answers from your own notes.</h1>
        <p>AcadAI combines retrieval, memory, viva practice, revision tooling, weak-topic tracking, and grounding checks into a clean academic workspace built for serious student learning and project demonstrations.</p>
        <div class="hero-actions">
          <span class="hero-pill">⚡ Lightweight retrieval</span>
          <span class="hero-pill">🧠 Memory-aware answers</span>
          <span class="hero-pill">🎯 Adaptive viva</span>
          <span class="hero-pill">🛡 Evidence-grounded output</span>
        </div>
      </div>
      <div class="pipeline">
        <div class="pipeline-step"><div class="pipeline-index">1</div><div><div class="pipeline-main">Router Agent</div><div class="pipeline-sub">selects RAG, web, or LLM path</div></div></div>
        <div class="pipeline-step"><div class="pipeline-index">2</div><div><div class="pipeline-main">Reasoning Agent</div><div class="pipeline-sub">plans learning and retrieval strategy</div></div></div>
        <div class="pipeline-step"><div class="pipeline-index">3</div><div><div class="pipeline-main">Tutor Agent</div><div class="pipeline-sub">teaches with examples and citations</div></div></div>
        <div class="pipeline-step"><div class="pipeline-index">4</div><div><div class="pipeline-main">Critic Agent</div><div class="pipeline-sub">checks quality, grounding, and gaps</div></div></div>
      </div>
    </div>
  </div>

  <div class="feature-grid">
    <div class="feature-card"><div class="feature-icon">💬</div><h4>Contextual Ask</h4><p>Follow-up questions reuse memory and recent learning context.</p></div>
    <div class="feature-card"><div class="feature-icon">🎓</div><h4>Viva Studio</h4><p>Generate viva questions, evaluate answers, and adapt difficulty.</p></div>
    <div class="feature-card"><div class="feature-icon">🗂️</div><h4>Revision Suite</h4><p>Roadmaps, flashcards, likely questions, and export-ready notes.</p></div>
    <div class="feature-card"><div class="feature-icon">✅</div><h4>Trust Console</h4><p>Inspect retrieved evidence, grounding score, and hallucination risk.</p></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:14px 14px 12px;border:1px solid #e5e7eb;border-radius:18px;background:linear-gradient(135deg,#ffffff,#f8fbff);box-shadow:0 8px 24px rgba(15,23,42,.05);margin-bottom:14px;">
      <div style="font-size:20px;font-weight:900;letter-spacing:-.04em;color:#0f172a;">🎓 AcadAI</div>
      <div style="font-size:12px;color:#64748b;font-weight:650;margin-top:4px;">Control center for your AI learning workspace</div>
    </div>
    """, unsafe_allow_html=True)
    st.header("Workspace")
    st.caption("Upload course PDFs or use the built-in CS demo corpus (DBMS, OS, DSA, Python).")
    uploads = st.file_uploader("Upload academic PDFs", type=["pdf"],
                               accept_multiple_files=True)
    difficulty = st.segmented_control(
        "Tutor level", ["beginner", "intermediate", "advanced"], default="intermediate")
    st.divider()
    st.subheader("Learning Platform Features")
    use_memory = st.toggle("Conversation memory", value=True)
    memory_turns = st.slider("Memory turns used", 1, 8, 4)
    show_grounding = st.toggle("Hallucination detector + grounding score", value=True)
    st.divider()
    use_web = st.toggle("Allow web fallback", value=True)
    max_refine = st.slider("Max Critic refinement loops", 0, 3, 1)
    retrieval_top_k = st.slider("Final evidence chunks (top k)", 4, 12, DEFAULT_TOP_K)
    candidate_k = st.slider("FAISS candidates to rerank", 10, 200, DEFAULT_CANDIDATE_K, step=5)
    use_hybrid_rerank = st.toggle("Hybrid rerank: semantic + keyword", value=True)
    use_cross_encoder = st.toggle("Cross-encoder rerank (optional; OFF saves data/RAM)", value=False)
    cross_encoder_model_name = st.text_input("Cross-encoder model", value=DEFAULT_CROSS_ENCODER_MODEL)
    st.caption("Lightweight reranker default is MiniLM (~80MB). Avoid BAAI/bge-reranker-v2-m3 unless you can download ~2.27GB.")
    subject_filter_mode = st.selectbox("Metadata/subject filter", ["auto", "manual", "off"], index=0)
    manual_subject = st.selectbox("Manual subject", ["Any", "CN", "OS", "DBMS", "DSA", "PYTHON", "GENERAL"], index=0)
    use_llm_expansion = st.toggle("Mistral query expansion", value=False)
    parent_context_chars = st.slider("Parent/adjacent context chars", 0, 2500, 1200, step=100)
    min_hybrid_score = st.slider("Minimum hybrid confidence", 0.00, 0.60, DEFAULT_MIN_HYBRID_SCORE, step=0.01)
    st.divider()
    st.subheader("FAISS Vector Store")
    use_faiss = st.toggle("Use existing FAISS store", value=False)
    faiss_dir = st.text_input("FAISS folder path", value=DEFAULT_FAISS_DIR)
    embedding_model_name = st.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)
    st.caption("Place `index.faiss` and `index.pkl` inside this folder. Use the same embedding model used while creating the FAISS index. For lightweight mode, keep Cross-Encoder OFF. If your FAISS index dimension is 1024, keep BAAI/bge-large-en-v1.5; changing embedding model can cause dimension mismatch.")
    st.divider()
    llm_label = "Mistral" if os.getenv("MISTRAL_API_KEY") else "Fallback mode"
    st.caption(
        f"LLM: **{llm_label}**. "
        "Add `MISTRAL_API_KEY` for full agent reasoning. "
        "Optional: set `MISTRAL_MODEL`, default is `mistral-large-latest`."
    )

uploaded_chunks, skipped = read_pdf_uploads(uploads)
faiss_index, faiss_chunks, faiss_error = (None, [], "")
if use_faiss:
    faiss_index, faiss_chunks, faiss_error = load_faiss_store(faiss_dir)

if use_faiss and faiss_index is not None and faiss_chunks:
    chunks = faiss_chunks
    corpus_label = "FAISS store"
elif uploaded_chunks:
    chunks = uploaded_chunks
    corpus_label = "Uploaded PDFs"
else:
    chunks = DEMO_CHUNKS
    corpus_label = "Demo CS corpus"

top_cols = st.columns(5)
with top_cols[0]: status_chip("Corpus", corpus_label, "ok" if uploaded_chunks else "info")
with top_cols[1]: status_chip("Chunks", str(len(chunks)), "info")
with top_cols[2]: status_chip("k (retrieval)", str(retrieval_top_k), "info")
with top_cols[3]: status_chip("LLM", llm_label.split(" ")[0], "ok"
                               if os.getenv("MISTRAL_API_KEY")
                               else "warn")
with top_cols[4]: status_chip("Web fallback", "On" if use_web else "Off",
                               "ok" if use_web else "warn")

if skipped:
    st.warning("Some PDFs were skipped: " + "; ".join(skipped))
if use_faiss and faiss_error:
    st.warning(faiss_error)
if use_faiss and faiss_index is not None and faiss_chunks:
    st.success(f"FAISS store loaded: {len(faiss_chunks)} chunks, index dimension {faiss_index.d}")

init_learning_state()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_ask, tab_viva, tab_roadmap, tab_revision, tab_eval, tab_memory = st.tabs(["Ask", "Viva Studio", "Roadmap", "Revision Suite", "Evaluation", "Memory"])

# ── Tab: Ask ───────────────────────────────────────────────────────────────────

with tab_ask:
    ask_left, ask_right = st.columns([0.74, 0.26], vertical_alignment="bottom")
    with ask_left:
        query = st.text_input("Ask AcadAI",
                              value="Explain DBMS normalization with an example")
    with ask_right:
        run = st.button("Generate Answer", type="primary", use_container_width=True)

    if run and query.strip():
        started = time.time()
        traces: List[AgentTrace] = []

        # Retriever
        if use_faiss and faiss_index is not None and faiss_chunks:
            db_rows, match = retrieve_faiss(
                query, faiss_index, chunks, embedding_model_name,
                top_k=retrieval_top_k, candidate_k=candidate_k,
                min_hybrid_score=min_hybrid_score,
                use_hybrid_rerank=use_hybrid_rerank,
                use_cross_encoder=use_cross_encoder,
                cross_encoder_model=cross_encoder_model_name,
                subject_filter_mode=subject_filter_mode,
                manual_subject=manual_subject,
                use_llm_expansion=use_llm_expansion,
                parent_context_chars=parent_context_chars,
            )
        else:
            db_rows, match = retrieve(query, chunks, top_k=retrieval_top_k)

        # Agent 1: Router
        route, tr_router = router_agent(query, match["match"], use_web)
        traces.append(tr_router)

        # Agent 2: Reasoning
        plan, tr_reasoning = reasoning_agent(query)
        traces.append(tr_reasoning)

        # Web search if needed
        web_rows = []
        if route == "Web Search":
            web_rows = web_search(query)

        # Evidence guard: if RAG evidence is weak, pass the warning into the answer context.
        if route == "RAG" and db_rows:
            best_overlap_guard = float(db_rows[0].get("overlap", 0))
            best_hybrid_guard = float(db_rows[0].get("hybrid_score", 0))
            if best_overlap_guard < 0.08 and best_hybrid_guard < min_hybrid_score:
                db_rows = [{
                    "doc_id": "retrieval_warning",
                    "source": "AcadAI retrieval guard",
                    "page": 0,
                    "evidence": "WARNING: Retrieved evidence appears weak or partially unrelated. Answer only what is supported and clearly mention missing evidence.",
                    "text": "WARNING: Retrieved evidence appears weak or partially unrelated. Answer only what is supported and clearly mention missing evidence.",
                    "rank": 0,
                    "subject": "GENERAL",
                    "hybrid_score": 0,
                    "overlap": 0,
                }] + db_rows

        # Conversation Memory Agent: add recent context only to generation, not retrieval.
        memory_context = build_memory_context(memory_turns) if use_memory else ""
        query_for_generation = query
        if memory_context:
            query_for_generation = (
                f"Current student question: {query}\n\n"
                f"Recent conversation memory for context:\n{memory_context}"
            )

        # Agent 3: Tutor
        answer, tr_tutor = tutor_agent(
            query_for_generation, difficulty,
            db_rows if route == "RAG" else [],
            web_rows, route, plan)
        traces.append(tr_tutor)

        # Agent 4: Critic + refinement loop
        scores, tr_critic = critic_agent(query, answer)
        traces.append(tr_critic)

        refine_count = 0
        while (not scores.get("satisfactory")
               and refine_count < max_refine
               and scores.get("feedback")):
            answer = refine_answer(query, answer, scores["feedback"], difficulty)
            scores, tr_critic2 = critic_agent(query, answer)
            traces.append(tr_critic2)
            refine_count += 1

        grounding_report = calculate_grounding_report(
            answer, db_rows if route == "RAG" else web_rows
        ) if show_grounding else {"score": 0.0, "supported": 0, "total": 0, "unsupported": [], "status": "Off"}

        store_conversation_turn(
            query=query, answer=answer, route=route,
            db_rows=db_rows if route == "RAG" else [],
            grounding_score=float(grounding_report.get("score", 0.0))
        )
        if show_grounding and float(grounding_report.get("score", 0.0)) < 55:
            detected = detect_query_subjects(query)
            update_weak_topic(query if not detected else detected[0], 1)

        total_latency = time.time() - started
        evidence_count = len(db_rows) if route == "RAG" else len(web_rows)
        metrics = compute_metrics(route, evidence_count, total_latency, scores)
        save_history({
            "query": query, "route": route,
            "overall": metrics["Overall quality"],
            "satisfactory": scores.get("satisfactory", "?"),
            "refinements": refine_count,
        })

        answer_col, insight_col = st.columns([0.62, 0.38], gap="large")

        with answer_col:
            section_header("Answer",
                           "Pedagogically structured response from the Tutor Agent.")
            with st.container(border=True):
                st.markdown(answer)

        with insight_col:
            section_header("Critic Scores",
                           "Quality evaluation across four dimensions.")
            mc = st.columns(2)
            tone_map = {
                "Answer route": "info",
                "Needs review": "warn",
                "Response time": "info",
                "Evidence used": "info",
            }
            for i, (lbl, val) in enumerate(metrics.items()):
                tone = tone_map.get(
                    lbl,
                    "warn" if str(val) in ("Yes",) else
                    "ok" if "/" in str(val) else "info")
                with mc[i % 2]:
                    metric_card(lbl, str(val), tone)

            if show_grounding:
                st.markdown("**Hallucination Detector**")
                g_score = float(grounding_report.get("score", 0.0))
                g_tone = "ok" if g_score >= 75 else "warn" if g_score >= 45 else "warn"
                metric_card("Grounding Score", f"{g_score}%", g_tone)
                st.caption(f"{grounding_report.get('status')} · Supported {grounding_report.get('supported')}/{grounding_report.get('total')} answer sentences")
                if grounding_report.get("unsupported"):
                    with st.expander("Possibly unsupported sentences"):
                        for s in grounding_report.get("unsupported", []):
                            st.write("- " + s)

        section_header("Agent Pipeline Trace",
                       "What each agent did and how long it took.")
        tone_seq = ["info", "purple", "ok", "warn"]
        a_cols = st.columns(min(len(traces), 4))
        for i, tr in enumerate(traces[:4]):
            with a_cols[i]:
                agent_badge(tr.agent, tr.action, tr.result, tr.latency,
                            tone_seq[i % len(tone_seq)])
        if len(traces) > 4:
            st.caption(f"+ {len(traces)-4} additional Critic refinement pass(es)")

        section_header("Reasoning Plan",
                       "Key concepts and solution plan from the Reasoning Agent.")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("**Key concepts**")
            for c in plan.get("key_concepts", []):
                st.markdown(f"- {c}")
        with r2:
            st.markdown("**Solution plan**")
            for i, step in enumerate(plan.get("solution_plan", []), 1):
                st.markdown(f"{i}. {step}")

        section_header("Evidence Path",
                       f"Route decision: {match['reason']} → **{route}**")
        st.info(f"Router selected: **{route}** | "
                f"DB match: {match['match']} | "
                f"Refinement loops: {refine_count}")
        if route == "RAG":
            st.dataframe(
                pd.DataFrame(db_rows).drop(columns=["text"]),
                use_container_width=True)
            with st.expander("Retrieved Evidence Debug"):
                for r in db_rows:
                    st.markdown(f"**Rank {r.get('rank')} | Subject: {r.get('subject','')} | Source: {r.get('source')} | Page: {r.get('page')} | Hybrid: {r.get('hybrid_score','—')} | Cross: {r.get('cross_score','—')}**")
                    st.write(r.get("evidence", ""))
        elif web_rows:
            st.dataframe(pd.DataFrame(web_rows), use_container_width=True)
        else:
            st.error("No reliable source found. "
                     "Upload relevant PDFs or enable web fallback.")

# ── Tab: Viva/Quiz ─────────────────────────────────────────────────────────────

with tab_viva:
    section_header("Viva / Quiz Mode",
                   "Generate oral-exam style questions, answer them, and get critic feedback.")
    viva_topic = st.text_input("Topic for viva", value="DBMS normalization")
    vc1, vc2, vc3 = st.columns([0.35, 0.35, 0.30])
    with vc1:
        quiz_k = st.slider("Quiz evidence chunks", 3, 10, 5)
    with vc2:
        quiz_candidate_k = st.slider("Quiz candidate search", 20, 200, 80, step=10)
    with vc3:
        make_quiz = st.button("Generate Quiz", type="primary", use_container_width=True)

    if make_quiz and viva_topic.strip():
        if use_faiss and faiss_index is not None and faiss_chunks:
            quiz_rows, quiz_match = retrieve_faiss(
                viva_topic, faiss_index, chunks, embedding_model_name,
                top_k=quiz_k, candidate_k=quiz_candidate_k,
                min_hybrid_score=min_hybrid_score,
                use_hybrid_rerank=use_hybrid_rerank,
                use_cross_encoder=use_cross_encoder,
                cross_encoder_model=cross_encoder_model_name,
                subject_filter_mode=subject_filter_mode,
                manual_subject=manual_subject,
                use_llm_expansion=use_llm_expansion,
                parent_context_chars=parent_context_chars,
            )
        else:
            quiz_rows, quiz_match = retrieve(viva_topic, chunks, top_k=quiz_k)
        st.session_state["quiz_topic"] = viva_topic
        st.session_state["quiz_rows"] = quiz_rows
        st.session_state["quiz_questions"] = generate_quiz(viva_topic, difficulty, quiz_rows)

    if st.session_state.get("quiz_questions"):
        st.markdown("### Questions")
        st.markdown(st.session_state["quiz_questions"])
        student_quiz_answer = st.text_area("Write your viva answer here", height=180)
        if st.button("Evaluate My Answer", use_container_width=True):
            feedback = evaluate_quiz_answer(
                st.session_state.get("quiz_topic", viva_topic),
                st.session_state.get("quiz_questions", ""),
                student_quiz_answer,
                st.session_state.get("quiz_rows", []),
            )
            st.markdown("### Viva Critic Feedback")
            st.markdown(feedback)
            parsed_score = parse_score_out_of_10(feedback)
            if parsed_score is not None:
                st.session_state.setdefault("quiz_attempts", []).append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "topic": st.session_state.get("quiz_topic", viva_topic),
                    "score": parsed_score,
                    "level": difficulty,
                })
                if parsed_score < 7:
                    update_weak_topic(st.session_state.get("quiz_topic", viva_topic), 2)
                metric_card("Adaptive Quiz Score", f"{parsed_score}/10", "ok" if parsed_score >= 7 else "warn")
                st.caption(f"Next recommended quiz level: {adaptive_difficulty_from_attempts(difficulty)}")
            g = calculate_grounding_report(feedback, st.session_state.get("quiz_rows", []))
            metric_card("Feedback Grounding", f"{g['score']}%", "ok" if g["score"] >= 70 else "warn")

        with st.expander("Quiz Evidence Used"):
            for r in st.session_state.get("quiz_rows", []):
                st.markdown(f"**{r.get('source')} · page {r.get('page')} · subject {r.get('subject','')}**")
                st.write(r.get("evidence", ""))

# ── Tab: Roadmap ───────────────────────────────────────────────────────────────

with tab_roadmap:
    section_header("Personalized Learning Roadmap",
                   "Generate a day-wise study plan using your B.Tech notes, profile, and weak-topic tracker.")
    profile = st.session_state.get("student_profile", {})
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        profile["name"] = st.text_input("Student name", value=profile.get("name", "Student"))
    with pc2:
        profile["semester"] = st.text_input("Semester", value=profile.get("semester", "B.Tech"))
    with pc3:
        profile["branch"] = st.text_input("Branch", value=profile.get("branch", "CSE / AIML"))
    profile["preferred_level"] = st.selectbox("Preferred explanation level", ["beginner", "intermediate", "advanced"], index=["beginner", "intermediate", "advanced"].index(profile.get("preferred_level", "intermediate")) if profile.get("preferred_level", "intermediate") in ["beginner", "intermediate", "advanced"] else 1)
    profile["goal"] = st.text_input("Learning goal", value=profile.get("goal", "exam + interview preparation"))
    st.session_state["student_profile"] = profile

    rc1, rc2, rc3 = st.columns([0.55, 0.20, 0.25])
    with rc1:
        roadmap_topic = st.text_input("Roadmap topic", value="DBMS complete revision")
    with rc2:
        roadmap_days = st.slider("Days", 3, 30, 10)
    with rc3:
        build_roadmap = st.button("Generate Roadmap", type="primary", use_container_width=True)

    if build_roadmap and roadmap_topic.strip():
        rows, _ = retrieve_for_tool(roadmap_topic, k=8)
        roadmap = generate_learning_roadmap(roadmap_topic, roadmap_days, profile.get("preferred_level", difficulty), rows)
        st.session_state.setdefault("saved_roadmaps", []).append({"topic": roadmap_topic, "days": roadmap_days, "roadmap": roadmap})
        st.markdown(roadmap)
        with st.expander("Roadmap Evidence"):
            for r in rows:
                st.markdown(f"**{r.get('source')} · page {r.get('page')} · subject {r.get('subject','')}**")
                st.write(r.get("evidence", ""))

    st.divider()
    section_header("Weak Topic Tracker", "Automatically updated from low grounding scores and low viva scores.")
    weak = st.session_state.get("weak_topics", {})
    if weak:
        df_weak = pd.DataFrame([{"Topic": k, "Weakness Count": v} for k, v in sorted(weak.items(), key=lambda x: x[1], reverse=True)])
        st.dataframe(df_weak, use_container_width=True)
    else:
        st.info("No weak topics tracked yet. Ask questions and attempt quizzes to build this profile.")

# ── Tab: Revision Tools ────────────────────────────────────────────────────────

with tab_revision:
    section_header("Revision Tools",
                   "Generate exam notes, likely questions, and flashcards from the same retrieved evidence.")
    tool_topic = st.text_input("Topic", value="Operating System deadlock")
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        mode = st.selectbox("Notes mode", ["Night-before exam notes", "One-page summary", "Detailed revision notes", "Interview/viva notes"])
    with tc2:
        flashcard_count = st.slider("Flashcards", 5, 25, 10)
    with tc3:
        run_tool = st.button("Generate Study Material", type="primary", use_container_width=True)

    if run_tool and tool_topic.strip():
        rows, _ = retrieve_for_tool(tool_topic, k=10)
        notes = generate_revision_notes(tool_topic, mode, rows)
        questions = generate_exam_questions(tool_topic, rows)
        cards = generate_flashcards(tool_topic, rows, count=flashcard_count)
        st.session_state.setdefault("saved_flashcards", []).append({"topic": tool_topic, "cards": cards})

        st.markdown("### Revision Notes")
        st.markdown(notes)
        st.markdown("### Likely Exam Questions")
        st.markdown(questions)
        st.markdown("### Flashcards")
        st.markdown(cards)
        with st.expander("Copy/export as Markdown"):
            export_md = f"# AcadAI Study Material: {tool_topic}\n\n## Revision Notes\n{notes}\n\n## Likely Questions\n{questions}\n\n## Flashcards\n{cards}"
            st.code(export_md, language="markdown")
        with st.expander("Study Material Evidence"):
            for r in rows:
                st.markdown(f"**{r.get('source')} · page {r.get('page')} · subject {r.get('subject','')}**")
                st.write(r.get("evidence", ""))

# ── Tab: Evaluation ────────────────────────────────────────────────────────────

with tab_eval:
    section_header("Retrieval Evaluation Dashboard",
                   "Tests retrieval quality before generation using current FAISS, rerank, filter, and top-k settings.")
    default_eval = "\n".join([
        "Give me most important subnetting numericals, include all patterns | CN",
        "What is Quality of Service (QoS)? | CN",
        "What is Segmentation in Operating Systems? | OS",
        "Explain normalization in DBMS | DBMS",
        "What is deadlock in operating systems? | OS",
        "Explain recursion with an example | DSA",
        "What is paging in memory management? | OS",
        "Explain primary key and foreign key | DBMS",
        "Explain SDLC models in Software Engineering | SE",
        "Explain HTML CSS and JavaScript in Web Technology | WEB",
        "Explain clustering in machine learning | ML",
        "Explain star schema in data warehousing | DWM",
    ])
    eval_text = st.text_area(
        "Evaluation queries. Format: query | expected subject",
        value=default_eval,
        height=160,
    )
    rows = []
    parsed = []
    for line in eval_text.splitlines():
        line = clean_text(line)
        if not line:
            continue
        if "|" in line:
            q, expected = [x.strip() for x in line.split("|", 1)]
        else:
            q, expected = line, ""
        parsed.append((q, expected))

    for q, expected in parsed:
        if use_faiss and faiss_index is not None and faiss_chunks:
            found, m = retrieve_faiss(
                q, faiss_index, chunks, embedding_model_name,
                top_k=retrieval_top_k, candidate_k=candidate_k,
                min_hybrid_score=min_hybrid_score,
                use_hybrid_rerank=use_hybrid_rerank,
                use_cross_encoder=use_cross_encoder,
                cross_encoder_model=cross_encoder_model_name,
                subject_filter_mode=subject_filter_mode,
                manual_subject=manual_subject,
                use_llm_expansion=use_llm_expansion,
                parent_context_chars=parent_context_chars,
            )
        else:
            found, m = retrieve(q, chunks, top_k=retrieval_top_k)
        top_subject = found[0].get("subject", "") if found else ""
        hit = bool(found) and (not expected or top_subject == expected)
        rows.append({
            "Query": q,
            "Expected": expected,
            "Top subject": top_subject,
            "Hit": "Yes" if hit else "No",
            "Top source": found[0]["source"] if found else "none",
            "Top page": found[0]["page"] if found else "—",
            "Hybrid@1": found[0].get("hybrid_score", "—") if found else "—",
            "Cross@1": found[0].get("cross_score", "—") if found else "—",
            "Overlap@1": found[0].get("overlap", "—") if found else "—",
            "Evidence preview": found[0]["evidence"] if found else "—",
            "Reason": m["reason"],
        })

    if rows:
        df_eval = pd.DataFrame(rows)
        hit_rate = round((df_eval["Hit"] == "Yes").mean() * 100, 1)
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("Hit Rate", f"{hit_rate}%", "ok" if hit_rate >= 70 else "warn")
        with c2: metric_card("Queries", str(len(rows)), "info")
        with c3: metric_card("Reranker", "CrossEncoder" if use_cross_encoder else "Hybrid", "purple")
        st.dataframe(df_eval, use_container_width=True)
        chart_df = df_eval.assign(HitValue=lambda d: (d["Hit"] == "Yes").astype(int))
        st.bar_chart(chart_df, x="Query", y="HitValue")
    else:
        st.info("Add evaluation queries to test retrieval quality.")

# ── Tab: Memory ────────────────────────────────────────────────────────────────

with tab_memory:
    section_header("Conversation Memory Agent",
                   "Recent learning turns used to make follow-up answers contextual.")

    c1, c2 = st.columns([0.7, 0.3])
    with c1:
        st.markdown("### Chat Memory")
    with c2:
        if st.button("Clear Conversation Memory", use_container_width=True):
            st.session_state["chat_history"] = []
            st.session_state["history"] = []
            st.rerun()

    chat_history = st.session_state.get("chat_history", [])
    if chat_history:
        for item in reversed(chat_history[-10:]):
            with st.container(border=True):
                st.markdown(f"**{item.get('time')} · {item.get('subject')} · {item.get('route')} · Grounding {item.get('grounding')}%**")
                st.markdown(f"**You:** {item.get('query')}")
                st.markdown(f"**AcadAI:** {quote(item.get('answer',''), 700)}")
    else:
        st.info("Ask a question to start building conversation memory.")

    st.divider()
    section_header("Student Profile + Adaptive Learning", "Personalization data used by roadmap and quiz agents.")
    st.json(st.session_state.get("student_profile", {}))
    attempts = st.session_state.get("quiz_attempts", [])
    if attempts:
        st.markdown("### Quiz Attempts")
        st.dataframe(pd.DataFrame(attempts), use_container_width=True)
    if st.session_state.get("weak_topics"):
        st.markdown("### Weak Topics")
        st.dataframe(pd.DataFrame([{"Topic": k, "Count": v} for k, v in st.session_state.get("weak_topics", {}).items()]), use_container_width=True)
    if st.session_state.get("saved_flashcards"):
        st.markdown("### Saved Flashcard Sets")
        for item in st.session_state.get("saved_flashcards", [])[-5:]:
            with st.expander(item.get("topic", "Flashcards")):
                st.markdown(item.get("cards", ""))

    st.divider()
    section_header("Session Metrics", "Classic history table for debugging and evaluation.")
    history = st.session_state.get("history", [])
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True)
    else:
        st.info("No metrics yet.")
