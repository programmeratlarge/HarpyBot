#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Knowledge-Base Chat with Retrieval-Augmented Generation (RAG)

This script launches a Gradio chatbot that answers user questions strictly from a local
knowledge base on disk. It recursively indexes files under ./knowledge_base (including
subfolders), applies token-aware chunking, computes embeddings, and performs RAG with
source citations. It also optionally ingests URLs listed in ./knowledge_base/url_list.txt.

It remembers chat context during a session, and will NOT fabricate facts: if an answer
is not supported by retrieved sources, it clearly says it doesn’t know.

A header “hero” section can be provided via a Markdown file named header.md placed in
the same directory as this script. Any images referenced by header.md should also be
located in the same directory. If header.md is not present, the UI displays a default
title ("Chatbot") and description ("Ask questions specific to the knowledge base").

Usage example:
    python chatbot.py --kb ./knowledge_base --host 0.0.0.0 --port 7860 --rebuild

Author: Paul Munn, Genomics Innovation Hub, Cornell University

Version history:
- 09/11/2025: (Version 1.2.0) Added header.md “hero” support and generic branding
- 09/11/2025: (Version 1.1.0) Added URL ingestion from knowledge_base/url_list.txt
- 09/10/2025: (Version 1.0.0) Original version
"""

# --- Imports
import os
import sys
import json
import time
import math
import glob
import hashlib
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import requests
from io import BytesIO
import re, base64, mimetypes

# Optional parsers (we'll handle missing ones gracefully)
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx  # python-docx
except Exception:
    docx = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import tiktoken
except Exception:
    tiktoken = None

import gradio as gr
from dotenv import load_dotenv

# OpenAI SDK (>=1.x)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --- Prompts: optional YAML support
try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

# --- Constants and defaults
VERSION = "1.2.0"
SUPPORTED_EXTS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm", ".xlsx", ".xls", ".csv"}
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
INDEX_DIRNAME = ".index_cache"  # stored inside the knowledge_base directory
CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 150
TOP_K = 5
SIM_THRESHOLD = 0.25  # below this cosine similarity we treat as "unknown"
MAX_CTX_MESSAGES = 12  # recent memory passed into the LLM
# --- Default generic system prompt (works for any KB chatbot)
DEFAULT_SYSTEM_PROMPT = (
    "You are a cautious, grounded knowledge-base assistant.\n"
    "Answer ONLY using the provided Knowledge Base Context.\n"
    "If the answer is not clearly supported by the context, reply exactly:\n"
    "I don't know - I can only answer questions about information in my knowledge base.\n"
    "Be concise. Always include source numbers at the end if any are used."
)

# --- Load system prompt override from prompts.yaml if present
def load_system_prompt(candidate_dirs: List[Path]) -> str:
    """
    Look for prompts.yaml in any of candidate_dirs (in order) and return the
    value of 'system_prompt'. If not found or invalid, return DEFAULT_SYSTEM_PROMPT.
    """
    # Gather candidate paths
    paths = []
    for d in candidate_dirs:
        try:
            paths.append((d / "prompts.yaml").resolve())
        except Exception:
            continue

    for p in paths:
        if not p.exists() or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
            # Prefer PyYAML if available (handles quoting/multiline properly)
            if yaml is not None:
                data = yaml.safe_load(text) or {}
                val = data.get("system_prompt")
                if isinstance(val, str) and val.strip():
                    print(f"[prompts.yaml] system_prompt loaded from: {p}")
                    return val.strip()
            else:
                # Minimal fallback parser: look for 'system_prompt:' first, grab remainder or indented lines
                import re
                m = re.search(r'^\s*system_prompt\s*:\s*(.*)$', text, flags=re.MULTILINE)
                if m:
                    first = m.group(1).strip()
                    if first and not first.startswith(('|', '>')):
                        # Same line value
                        # Strip surrounding quotes if present
                        if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
                            first = first[1:-1]
                        if first.strip():
                            print(f"[prompts.yaml] system_prompt (inline) loaded from: {p}")
                            return first.strip()
                    else:
                        # Multiline block: collect subsequent indented lines
                        lines = text.splitlines()
                        start_idx = None
                        for i, line in enumerate(lines):
                            if re.match(r'^\s*system_prompt\s*:', line):
                                start_idx = i + 1
                                break
                        if start_idx is not None:
                            collected = []
                            for j in range(start_idx, len(lines)):
                                line = lines[j]
                                if re.match(r'^\s+\S', line):  # indented
                                    collected.append(line.strip())
                                elif line.strip() == "":       # allow blank lines
                                    collected.append("")
                                else:
                                    break
                            block = "\n".join(collected).strip()
                            if block:
                                print(f"[prompts.yaml] system_prompt (block) loaded from: {p}")
                                return block
        except Exception as e:
            print(f"[prompts.yaml] error reading {p}: {e}")

    # Fallback
    print("[prompts.yaml] not found; using DEFAULT_SYSTEM_PROMPT.")
    return DEFAULT_SYSTEM_PROMPT

# --- Load header.md from several candidate directories
def load_hero_markdown(candidate_dirs: List[Path]) -> Tuple[str, bool, str, str]:
    """
    Returns (hero_markdown, header_found, app_title, header_path_used).
    """
    search_paths = []
    for d in candidate_dirs:
        try:
            search_paths.append((d / "header.md").resolve())
        except Exception:
            continue

    header_path = next((p for p in search_paths if p.exists() and p.is_file()), None)

    if header_path:
        try:
            md = header_path.read_text(encoding="utf-8", errors="ignore").strip()

            # Inline local images so the logo always renders in the app
            md = _inline_local_images_as_data_uri(md, header_path.parent)

            # Inject version next to the first heading
            md = _inject_version_into_title(md, VERSION)

            # Derive window title from first Markdown heading if present
            title = "Knowledge-Base Chat"
            for line in md.splitlines():
                if line.strip().startswith("#"):
                    title = line.strip("# ").strip() or title
                    break

            print(f"[header.md] loaded from: {header_path}")
            return md, True, title, header_path.as_posix()
        except Exception as e:
            print(f"[header.md] error reading {header_path}: {e}")

    # Fallback header (with version)
    default_title = "Chatbot"
    default_desc = "Ask questions specific to the knowledge base"
    md = f"# {default_title} <span style=\"font-size:0.85em; font-weight: normal; opacity: 0.75;\">v{VERSION}</span>\n\n{default_desc}\n"
    print("[header.md] not found; using default header.")
    return md, False, default_title, ""

def _rewrite_local_images_to_file_scheme(md: str, base_dir: Path) -> str:
    """
    Rewrites Markdown and HTML <img> image refs to use Gradio's file= scheme
    so that local images are served correctly by the app.
    """
    # Markdown images: ![alt](url)
    def md_repl(m):
        alt = m.group(1) or ""
        url = (m.group(2) or "").strip()
        if re.match(r"^(?:https?:|file=|data:)", url, flags=re.I):
            return m.group(0)
        abs_path = (base_dir / url).resolve().as_posix()
        return f"![{alt}](file={abs_path})"

    md = re.sub(r"!\[(.*?)\]\((.*?)\)", md_repl, md)

    # HTML <img src="url">
    def html_repl(m):
        url = (m.group(1) or "").strip()
        if re.match(r"^(?:https?:|file=|data:)", url, flags=re.I):
            return m.group(0)
        abs_path = (base_dir / url).resolve().as_posix()
        return m.group(0).replace(m.group(1), f"file={abs_path}")

    md = re.sub(r'<img[^>]*\ssrc=["\']([^"\']+)["\']', html_repl, md, flags=re.I)
    return md

def _inline_local_images_as_data_uri(md: str, base_dir: Path) -> str:
    """
    Convert local image refs in Markdown/HTML to data: URIs so they always render.
    Remote (http/https) images are left as-is.
    """
    def to_data_uri(path: Path) -> str | None:
        try:
            mime, _ = mimetypes.guess_type(path.name)
            mime = mime or "application/octet-stream"
            b64 = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime};base64,{b64}"
        except Exception:
            return None

    # Markdown images: ![alt](url)
    def md_repl(m):
        alt = m.group(1) or ""
        url = (m.group(2) or "").strip()
        if re.match(r"^(https?:|data:|file=)", url, flags=re.I):
            return m.group(0)
        p = (base_dir / url).resolve()
        if p.exists():
            data = to_data_uri(p)
            if data:
                return f"![{alt}]({data})"
        return m.group(0)

    md = re.sub(r"!\[(.*?)\]\((.*?)\)", md_repl, md)

    # HTML <img src="url">
    def html_repl(m):
        url = (m.group(1) or "").strip()
        if re.match(r"^(https?:|data:|file=)", url, flags=re.I):
            return m.group(0)
        p = (base_dir / url).resolve()
        if p.exists():
            data = to_data_uri(p)
            if data:
                return m.group(0).replace(m.group(1), data)
        return m.group(0)

    md = re.sub(r'<img([^>]*?)\ssrc=["\']([^"\']+)["\']', html_repl, md, flags=re.I)
    return md

def _inject_version_into_title(md: str, version: str) -> str:
    """
    Find the first Markdown heading and append a small, muted version tag.
    If no heading is found, returns md unchanged.
    """
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("<!-- Insert version -->"):
            # Append version tag with smaller font on the same line
            # Works for both Markdown and HTML renderers in Gradio/Cursor
            lines[i] = line.rstrip().lstrip("<!-- Insert version -->") + f' <span style="font-size:0.85em; font-weight: normal; opacity: 0.75;">(v{version})</span>'
            return "\n".join(lines)
    return md

# --- Fetch a URL and extract text based on content type
def _fetch_and_extract_url(url: str) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    try:
        headers = {"User-Agent": "HarpyBot/1.0 (+https://example.local)"}
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code != 200 or not resp.content:
            return out  # ignore not found / empty

        ctype = (resp.headers.get("Content-Type") or "").lower()

        # --- HTML / XML / text
        if any(x in ctype for x in ["text/html", "application/xhtml", "xml", "text/plain"]) or (
            not ctype and ("http" in url or "html" in url)
        ):
            text = None
            if "text/plain" in ctype:
                enc = resp.encoding or "utf-8"
                text = resp.content.decode(enc, errors="ignore")
            else:
                if BeautifulSoup is None:
                    # Fallback: decode as text if bs4 missing
                    enc = resp.encoding or "utf-8"
                    text = resp.content.decode(enc, errors="ignore")
                else:
                    html = resp.content.decode(resp.encoding or "utf-8", errors="ignore")
                    soup = BeautifulSoup(html, "html.parser")
                    text = soup.get_text(separator="\n")

            if text and text.strip():
                out.append((text, {"file": url, "type": "url-html"}))
            return out

        # --- PDF
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            if pypdf is None:
                return out
            try:
                reader = pypdf.PdfReader(BytesIO(resp.content))
                for i, page in enumerate(reader.pages):
                    try:
                        ptxt = page.extract_text() or ""
                    except Exception:
                        ptxt = ""
                    if ptxt.strip():
                        out.append((ptxt, {"file": url, "type": "url-pdf", "page": i + 1}))
            except Exception:
                return out
            return out

        # --- Unknown binary; best-effort decode as text
        try:
            text = resp.content.decode(resp.encoding or "utf-8", errors="ignore")
            if text.strip():
                out.append((text, {"file": url, "type": "url-unknown"}))
        except Exception:
            pass
        return out

    except Exception:
        return out


# --- Load URL list from knowledge_base/url_list.txt and extract texts
def load_url_documents(kb_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    docs: List[Tuple[str, Dict[str, Any]]] = []
    url_list_path = kb_dir / "url_list.txt"
    if not url_list_path.exists():
        return docs

    try:
        lines = url_list_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return docs

    seen = set()
    for raw in lines:
        url = raw.strip()
        if not url or url.startswith("#"):
            continue
        if url in seen:
            continue
        seen.add(url)

        extracted = _fetch_and_extract_url(url)
        # ignore not found / parse failures automatically
        docs.extend(extracted)

    return docs

# --- Token utilities
def _tick_encode(text: str):
    """Tokenization helper with graceful fallback to naive splitting."""
    if tiktoken is None:
        # Rough fallback: pretend ~4 chars/token (very approximate)
        return list(text)  # we'll chunk by characters if tiktoken isn't available
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)


def _tick_decode(tokens: List[int]):
    """Reverse tokenization helper with graceful fallback."""
    if tiktoken is None:
        return "".join(tokens) if isinstance(tokens, list) else ""
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(tokens)


def _chunk_by_tokens(text: str, chunk_tokens=CHUNK_TOKENS, overlap_tokens=CHUNK_OVERLAP_TOKENS):
    """Token-aware recursive chunking with overlap; falls back to char-based if no tiktoken."""
    toks = _tick_encode(text)
    chunks = []
    start = 0
    n = len(toks)
    if tiktoken is None:
        # Character-based fallback with similar sizing
        approx_chars = chunk_tokens * 4
        approx_overlap = overlap_tokens * 4
        while start < len(text):
            end = min(len(text), start + approx_chars)
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            if end == len(text):
                break
            start = end - approx_overlap
            if start < 0:
                start = 0
        return chunks

    while start < n:
        end = min(n, start + chunk_tokens)
        sub = toks[start:end]
        if sub:
            chunks.append(_tick_decode(sub))
        if end == n:
            break
        start = end - overlap_tokens
        if start < 0:
            start = 0
    return chunks


# --- File readers
def read_pdf(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract text per page from PDF."""
    if pypdf is None:
        return []
    out = []
    try:
        with path.open("rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages):
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                if text.strip():
                    out.append((text, {"file": str(path), "type": "pdf", "page": i + 1}))
    except Exception:
        return out
    return out


def read_docx(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract text from DOCX."""
    if docx is None:
        return []
    try:
        d = docx.Document(str(path))
        text = "\n".join(p.text for p in d.paragraphs)
        if text.strip():
            return [(text, {"file": str(path), "type": "docx"})]
    except Exception:
        return []
    return []


def read_text_like(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract text from TXT/MD files (UTF-8)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            return [(text, {"file": str(path), "type": path.suffix.lstrip(".")})]
    except Exception:
        return []
    return []


def read_html(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract visible text from HTML."""
    if BeautifulSoup is None:
        return []
    try:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator="\n")
        if text.strip():
            return [(text, {"file": str(path), "type": "html"})]
    except Exception:
        return []
    return []


def read_excel_like(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract text from Excel/CSV by converting to tabular text."""
    if pd is None:
        return []
    out = []
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, dtype=str, encoding="utf-8", engine="python")
            text = df.fillna("").to_csv(index=False)
            if text.strip():
                out.append((text, {"file": str(path), "type": "csv"}))
        else:
            # Excel
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                df = xls.parse(sheet, dtype=str)
                text = f"# Sheet: {sheet}\n" + df.fillna("").to_csv(index=False)
                if text.strip():
                    out.append((text, {"file": str(path), "type": "excel", "sheet": sheet}))
    except Exception:
        return out
    return out


def load_documents(kb_dir: Path) -> List[Tuple[str, Dict[str, Any]]]:
    """Walk the knowledge base and return list of (text, metadata) tuples."""
    docs = []
    for root, _, files in os.walk(kb_dir):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() not in SUPPORTED_EXTS:
                continue
            try:
                if p.suffix.lower() == ".pdf":
                    docs.extend(read_pdf(p))
                elif p.suffix.lower() == ".docx":
                    docs.extend(read_docx(p))
                elif p.suffix.lower() in {".txt", ".md"}:
                    docs.extend(read_text_like(p))
                elif p.suffix.lower() in {".html", ".htm"}:
                    docs.extend(read_html(p))
                elif p.suffix.lower() in {".xlsx", ".xls", ".csv"}:
                    docs.extend(read_excel_like(p))
            except Exception:
                # Skip problematic file; continue indexing others
                continue
    return docs


# --- Embeddings
def get_openai_client():
    """Instantiate OpenAI client, ensuring key exists."""
    if OpenAI is None:
        raise RuntimeError("openai package not installed. `pip install openai`")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


def embed_texts(client, texts: List[str], model: str = EMBED_MODEL, batch_size: int = 128) -> np.ndarray:
    """Embed a list of texts with batching; returns (N, D) float32 numpy array."""
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        arr = np.array([e.embedding for e in resp.data], dtype=np.float32)
        vecs.append(arr)
    if vecs:
        mat = np.vstack(vecs)
        # Normalize to unit length for cosine similarity
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        mat = mat / norms
        return mat.astype(np.float32)
    return np.zeros((0, 1536), dtype=np.float32)


# --- Hash local KB files and url_list.txt content to detect changes
def kb_fingerprint(kb_dir: Path) -> str:
    h = hashlib.sha256()

    # local files (paths + mtimes + sizes)
    for root, _, files in os.walk(kb_dir):
        for name in sorted(files):
            p = Path(root) / name
            if p.name == "url_list.txt":
                continue  # handled below
            try:
                stat = p.stat()
                h.update(str(p).encode())
                h.update(str(int(stat.st_mtime)).encode())
                h.update(str(stat.st_size).encode())
            except Exception:
                continue

    # url_list.txt (paths + content hash) — remote pages are not hashed; use --rebuild or Refresh
    url_list = kb_dir / "url_list.txt"
    if url_list.exists():
        try:
            h.update(b"url_list.txt")
            h.update(url_list.read_bytes())
        except Exception:
            pass

    return h.hexdigest()[:16]

def build_index(kb_dir: Path, index_dir: Path) -> Dict[str, Any]:
    """Build or rebuild the index from scratch."""
    client = get_openai_client()
    # Collect local docs + URL docs
    raw_docs = load_documents(kb_dir) + load_url_documents(kb_dir)

    texts = []
    metadatas = []
    # Comment: chunk documents
    for text, meta in raw_docs:
        for chunk in _chunk_by_tokens(text):
            if chunk.strip():
                texts.append(chunk)
                metadatas.append(meta)

    # Comment: compute embeddings
    emb = embed_texts(client, texts)

    # Comment: persist to disk
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", emb)
    with (index_dir / "texts.pkl").open("wb") as f:
        pickle.dump(texts, f)
    with (index_dir / "metadatas.pkl").open("wb") as f:
        pickle.dump(metadatas, f)
    fp = kb_fingerprint(kb_dir)
    with (index_dir / "fingerprint.json").open("w", encoding="utf-8") as f:
        json.dump({"fingerprint": fp, "ts": time.time()}, f)

    return {"embeddings": emb, "texts": texts, "metadatas": metadatas, "fingerprint": fp}


def load_index(kb_dir: Path, index_dir: Path, rebuild: bool = False, trust_cache: bool = False) -> Dict[str, Any]:
    index_dir.mkdir(parents=True, exist_ok=True)

    # print('index_dir: ', index_dir)
    # print('trust_cache: ', trust_cache)

    # Files we need in cache
    emb_p = index_dir / "embeddings.npy"
    txt_p = index_dir / "texts.pkl"
    meta_p = index_dir / "metadatas.pkl"
    fp_p   = index_dir / "fingerprint.json"  # optional in trust-cache mode

    # --- TRUST CACHE: load directly and skip fingerprint checks
    if trust_cache:
        if all(p.exists() for p in (emb_p, txt_p, meta_p)):
            emb = np.load(emb_p)
            with txt_p.open("rb") as f: texts = pickle.load(f)
            with meta_p.open("rb") as f: metadatas = pickle.load(f)
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = (emb / norms).astype(np.float32)
            return {"embeddings": emb, "texts": texts, "metadatas": metadatas, "fingerprint": "cache-only"}
        else:
            raise RuntimeError("trust-cache is enabled, but .index_cache is incomplete (missing embeddings/texts/metadatas).")

    # --- Normal path (fingerprint verify or rebuild)
    fp_now = kb_fingerprint(kb_dir)
    exists = all(p.exists() for p in (emb_p, txt_p, meta_p, fp_p))
    # print('exists: ', exists)
    #print('rebuild: ', rebuild)
    # print('fp_now: ', fp_now)

    if exists and not rebuild:
        try:
            # fp_data = json.loads(fp_p.read_text(encoding="utf-8"))
            # print('fp_data.get: ', fp_data.get("fingerprint"))
            # if fp_data.get("fingerprint") == fp_now:
            emb = np.load(emb_p)
            with txt_p.open("rb") as f: texts = pickle.load(f)
            with meta_p.open("rb") as f: metadatas = pickle.load(f)
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = (emb / norms).astype(np.float32)
            return {"embeddings": emb, "texts": texts, "metadatas": metadatas, "fingerprint": fp_now}
        except Exception as e:
            # raise RuntimeError(f".index_cache exists but is throwing an error: {e}")
            pass

    # If we get here, rebuild
    print('Rebuilding index')
    return build_index(kb_dir, index_dir)


# --- Retrieval
def retrieve(client, index: Dict[str, Any], query: str, top_k: int = TOP_K) -> List[Tuple[float, str, Dict[str, Any]]]:
    """Return top_k (score, text, meta) by cosine similarity."""
    if not index["texts"]:
        return []
    q_emb = embed_texts(client, [query])[0]  # (D,)
    db = index["embeddings"]  # (N, D)
    scores = (db @ q_emb)  # cosine since both normalized
    top_idx = np.argsort(-scores)[:top_k]
    out = []
    for i in top_idx:
        out.append((float(scores[i]), index["texts"][i], index["metadatas"][i]))
    return out


# --- Prompting and grounded generation
def grounded_answer(
    client,
    query: str,
    history: List[Dict[str, str]],
    retrieved: List[Tuple[float, str, Dict[str, Any]]],
    model: str,
    system_prompt: str,  # <-- new param
) -> str:
    # Filter weak results
    strong = [(s, t, m) for (s, t, m) in retrieved if s >= SIM_THRESHOLD]

    # Build source context block
    if strong:
        context_parts, citations = [], []
        for rank, (score, text, meta) in enumerate(strong, start=1):
            file = meta.get("file", "unknown")
            page = meta.get("page")
            sheet = meta.get("sheet")
            loc = f"{file}" + (f" (page {page})" if page else "") + (f" [sheet: {sheet}]" if sheet else "")
            context_parts.append(f"[{rank}] {text.strip()[:2000]}")
            citations.append(f"[{rank}] {loc}")
        context_str = "\n\n".join(context_parts)
        citations_str = "\n".join(citations)
    else:
        context_str = ""
        citations_str = ""

    # Enforce “only from KB” using passed-in system prompt
    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-MAX_CTX_MESSAGES:]:
        messages.append(m)
    if strong:
        messages.append({"role": "system", "content": f"Knowledge Base Context:\n{context_str}"})
    messages.append({"role": "user", "content": query})

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=messages,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error from model: {e}"

    if not strong:
        return "I couldn't find that in the knowledge base."
    if citations_str and citations_str not in answer:
        answer = f"{answer}\n\nSources:\n{citations_str}"
    return answer

# --- Gradio app factory
def build_ui(kb_dir: Path, index_state: Dict[str, Any], system_prompt: str):

    client = get_openai_client()
    state = {"index": index_state}

    try:
        app_dir = Path(__file__).resolve().parent
    except NameError:
        app_dir = Path.cwd()

    hero_md, header_found, page_title, header_used = load_hero_markdown(
        candidate_dirs=[app_dir, Path.cwd(), kb_dir]
    )

    CSS = """
    /* -----------------------------------------
       Header card: light/dark adaptive styling
    ------------------------------------------*/
    /* Base: treat as light mode defaults */
    .gradio-container .hero-box {
      border-radius: 12px;
      padding: 16px 18px;
      margin-bottom: 12px;
      border: 1px solid #e5e7eb !important;   /* light border */
      background: #f3f4f6 !important;          /* light gray bg */
      color: #000000 !important;                /* black text */
    }
    .gradio-container .hero-box * { color: inherit !important; }
    .gradio-container .hero-box a,
    .gradio-container .hero-box a:visited {
      color: inherit !important;
      text-decoration: underline;
    }

    /* Header image: exactly 700 px wide */
    .gradio-container .hero-box img {
      width: 700px !important;
      height: auto !important;
      vertical-align: middle;
      margin-right: 10px;
    }

    /* Dark mode override — handles common Gradio theme markers */
    .gradio-container[data-theme="dark"] .hero-box,
    .gradio-container[data-theme*="dark"] .hero-box,
    html.dark .gradio-container .hero-box,
    body.dark .gradio-container .hero-box {
      background: #1f2937 !important;          /* dark gray */
      border-color: #374151 !important;
      color: #ffffff !important;                /* white text */
    }
    .gradio-container[data-theme="dark"] .hero-box *,
    .gradio-container[data-theme*="dark"] .hero-box *,
    html.dark .gradio-container .hero-box *,
    body.dark .gradio-container .hero-box * {
      color: inherit !important;
    }

    /* Fallback if the app doesn't set a theme attribute/class but user prefers dark */
    @media (prefers-color-scheme: dark) {
      .gradio-container:not([data-theme]) .hero-box {
        background: #1f2937 !important;
        border-color: #374151 !important;
        color: #ffffff !important;
      }
      .gradio-container:not([data-theme]) .hero-box * { color: inherit !important; }
    }

    /* Make the Markdown image ![Harpy](...) 700px wide */
    .hero-box img[alt="Harpy"],
    #hero-box img[alt="Harpy"] {
      width: 700px !important;
      height: auto !important;
      vertical-align: middle;
      margin-right: 10px;
      margin-bottom: 6px;
    }

    /* -----------------------------------------
       Footer (status + refresh): 50% width, left
    ------------------------------------------*/
    #footer-row {
      width: 50%;
      justify-content: flex-start;        /* left-justify the row */
      gap: 8px;
    }
    #status-box { 
      max-width: 100%;
    }
    #status-box textarea {
      height: 36px;
      font-size: 0.9rem;
      resize: none;
    }
    #refresh-btn button {
      white-space: nowrap;
    }

    /* Responsive: full width on small screens */
    @media (max-width: 900px) {
      #footer-row { width: 100%; }
    }

    /* Keep the Send button a bit emphasized even if variant isn't supported */
    #send-btn button { font-weight: 600; }
    """

    def refresh_index():
        idx_dir = kb_dir / INDEX_DIRNAME
        state["index"] = build_index(kb_dir, idx_dir)
        n = len(state["index"]["texts"])
        return gr.update(value=f"Index rebuilt ✓ ({n} chunks)")

    def get_index_status_text():
        return f"Index loaded ({len(state['index']['texts'])} chunks)"

    with gr.Blocks(title=page_title, css=CSS) as demo:
        # Header “card”
        gr.Markdown(hero_md, elem_id="hero-box")

        chatbot = gr.Chatbot(height=460, type="messages")

        # Textbox: no label, new placeholder
        with gr.Row():
            msg = gr.Textbox(
                label=None,                    # remove "Textbox"
                show_label=False,              # hide label across Gradio versions
                placeholder="Ask a question about the knowledge base...",
                lines=2,
                autofocus=True,
            )

        # Buttons (Send highlighted)
        with gr.Row():
            send_btn  = gr.Button("Send", variant="primary", elem_id="send-btn")
            retry_btn = gr.Button("Retry")
            undo_btn  = gr.Button("Undo")
            clear_btn = gr.Button("Clear")

        # Footer: Index status + Refresh at the bottom (half-width, left-justified)
        # with gr.Row(elem_id="footer-row"):
        #     status_box = gr.Textbox(
        #         value=f"Index loaded ({len(state['index']['texts'])} chunks)",
        #         label="Index Status",
        #         interactive=False,
        #         lines=1,
        #         elem_id="status-box",
        #     )
        #     refresh = gr.Button("Refresh Knowledge Base", elem_id="refresh-btn")

        # respond: history is now a list of {"role": "...", "content": "..."} dicts
        def respond_fn(user_message, history):
            # Use history directly as OpenAI-style messages
            conv = history or []
            retrieved = retrieve(client, state["index"], user_message, top_k=TOP_K)
            answer = grounded_answer(
                client, user_message, conv, retrieved, model=CHAT_MODEL, system_prompt=system_prompt
            )
            # Append user + assistant messages
            new_history = (history or []) + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": answer},
            ]
            return new_history, gr.update(value="")

        # retry: drop the last assistant and regenerate using the same last user turn
        def retry_fn(history):
            history = list(history or [])
            if not history:
                return history
            if history[-1]["role"] != "assistant":
                # Nothing to retry yet
                return history

            # Remove last assistant
            history = history[:-1]

            # Find the last user message content
            last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
            if last_user is None:
                return history  # no user to retry

            retrieved = retrieve(client, state["index"], last_user, top_k=TOP_K)
            answer = grounded_answer(
                client, last_user, history, retrieved, model=CHAT_MODEL, system_prompt=system_prompt
            )
            history.append({"role": "assistant", "content": answer})
            return history

        # undo: remove the most recent turn (assistant + its user), if present
        def undo_fn(history):
            history = list(history or [])
            if not history:
                return history

            if history[-1]["role"] == "assistant":
                # Remove assistant
                history.pop()
                # Remove the paired user (if present)
                if history and history[-1]["role"] == "user":
                    history.pop()
            else:
                # If last is user without assistant yet, just remove it
                history.pop()
            return history

        def clear_fn():
            return [], ""

        # Wire up
        msg.submit(respond_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        send_btn.click(respond_fn, inputs=[msg, chatbot], outputs=[chatbot, msg])
        retry_btn.click(retry_fn, inputs=[chatbot], outputs=[chatbot])
        undo_btn.click(undo_fn, inputs=[chatbot], outputs=[chatbot])
        clear_btn.click(clear_fn, outputs=[chatbot, msg])
        # refresh.click(refresh_index, outputs=status_box)

    return demo


# --- Main entry point
def main():
    # Load environment variables
    load_dotenv(override=True)

    # Parse args (unchanged)
    parser = argparse.ArgumentParser(description="KB-grounded chatbot with RAG")
    parser.add_argument("--kb", type=str, default="knowledge_base")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--trust-cache", action="store_true",
                    help="Load knowledge_base/.index_cache/ without verifying fingerprint or rebuilding.")

    args = parser.parse_args()
    trust_cache = args.trust_cache or os.getenv("TRUST_CACHE", "0") == "1"

    if OpenAI is None:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY in your environment.", file=sys.stderr)
        sys.exit(1)

    kb_dir = Path(args.kb).resolve()
    kb_dir.mkdir(parents=True, exist_ok=True)
    index_dir = kb_dir / INDEX_DIRNAME

    # Determine app_dir even when __file__ is missing
    try:
        app_dir = Path(__file__).resolve().parent
    except NameError:
        app_dir = Path.cwd()

    # Load system prompt (prompts.yaml override if present)
    system_prompt = load_system_prompt(candidate_dirs=[app_dir, Path.cwd(), kb_dir])
    # print('System prompt: ', system_prompt)

    # Load/build index
    index_state = load_index(kb_dir, index_dir, rebuild=args.rebuild, trust_cache=trust_cache)

    # Launch UI with prompt
    app = build_ui(kb_dir, index_state, system_prompt=system_prompt)
    app.launch(server_name=args.host, server_port=args.port, show_error=True)


# --- Script runner
if __name__ == "__main__":
    main()
