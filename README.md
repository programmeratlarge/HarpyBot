# HarpyBot

A lightweight, Retrieval-Augmented Generation (RAG) chatbot that answers **only** from your local knowledge base (and optional URL sources). HarpyBot runs as a Gradio web app, cites the sources it used, and clearly says **“I don’t know”** when the answer isn’t supported by the knowledge base.

---

## Features

* 🔎 **RAG over your files**: PDFs, DOCX, TXT/MD/HTML, CSV/XLSX (+ optional URLs via `knowledge_base/url_list.txt`)
* 🧩 **Fast startup**: chunking + embeddings cached under `knowledge_base/.index_cache/`
* 🧠 **Session memory**: keeps a short recent chat history while staying grounded to retrieved context
* 📝 **Prompt overrides**: drop a `prompts.yaml` with `system_prompt:` to customize tone/behavior
* 🖼️ **Custom header**: optional `header.md` “hero” box (dark/light aware)
* 🚀 **Cache-only deploy**: ship just `.index_cache/` to Hugging Face (no raw documents) with `TRUST_CACHE=1`

---

## 1) What it does

HarpyBot scans a folder (default: `./knowledge_base/`) and optionally a list of URLs (`knowledge_base/url_list.txt`), chunks and embeds the text, and serves a chatbot UI. When you ask a question, it retrieves the most relevant chunks and generates an answer **only** from those chunks. If it can’t support an answer with retrieved text, it responds:

> *I don't know — I can only answer questions about information in my knowledge base.*

Every answer includes source numbers (file path and page/sheet when available).

---

## 2) How to use it

1. Put your documents under:

   ```
   knowledge_base/
     ├─ subfolder1/
     ├─ subfolder2/
     └─ ...
   ```

   Supported: `.pdf`, `.docx`, `.txt`, `.md`, `.html`, `.htm`, `.csv`, `.xlsx`, `.xls`

2. (Optional) Add URLs (one per line) to `knowledge_base/url_list.txt`:

   ```text
   https://example.com/whitepaper.html
   https://example.com/notes.pdf
   ```

3. (Optional) Override the system prompt with `prompts.yaml` (next to the app):

   ```yaml
   system_prompt: "You are Harpy Bot, a cautious research assistant. Answer ONLY using the provided knowledge base context. If the answer is not clearly supported by the context, say: I don't know - I can only answer questions about information in my knowledge base. Be concise. Always include source numbers at the end if any are used."
   ```

4. (Optional) Add a header hero via `header.md` (same folder as the app). Reference images with relative paths (e.g., `![Harpy](./harpy_bot_logo_text.png)`).

5. Build the index (first time only or when docs change), then launch:

   ```bash
   # 1) Set your OpenAI key
   export OPENAI_API_KEY=sk-...        # Windows PowerShell:  $env:OPENAI_API_KEY="sk-..."

   # 2) Install deps
   pip install -r requirements.txt

   # 3) Run (rebuild index on first run)
   python chatbot.py --kb ./knowledge_base --rebuild --host 0.0.0.0 --port 7860
   ```

6. Open the URL shown in the terminal (default [http://127.0.0.1:7860](http://127.0.0.1:7860)), type a question, and click **Send**.

### Example

**You:**

```
What does the pipeline recommend for demultiplexing, and where is that described?
```

**HarpyBot:**

```
The pipeline recommends using X for demultiplexing and provides a step-by-step guide.

Sources:
[1] knowledge_base/pipeline_guide.pdf (page 7)
[2] knowledge_base/notes/demux.md
```

---

## 3) Live app (Hugging Face)

➡️ **HarpyBot on Hugging Face**: [https://programmeratlarge-harpybot.hf.space](https://programmeratlarge-harpybot.hf.space)

> Tip: For fastest startup on Spaces, deploy **only the prebuilt index cache** (see “Cache-only deploy”) and set `TRUST_CACHE=1`.

---

## 4) Install & run locally

### Minimum software

* **Python** 3.10+ (3.11/3.12 fine)
* **pip**
* **OpenAI API key** in `OPENAI_API_KEY`

### Suggested `requirements.txt`

```
gradio>=4
openai>=1
python-dotenv
pyyaml
tiktoken
numpy
# Only needed if you'll (re)build the index on this machine:
pypdf
python-docx
beautifulsoup4
pandas
openpyxl
```

### Run

```bash
export OPENAI_API_KEY=sk-...                 # Windows PS: $env:OPENAI_API_KEY="sk-..."
pip install -r requirements.txt

# First run with rebuild to create the index
python chatbot.py --kb ./knowledge_base --rebuild

# Later runs can skip rebuild:
python chatbot.py --kb ./knowledge_base
```

**Useful flags**

* `--kb PATH` – path to knowledge base (default `knowledge_base`)
* `--rebuild` – force re-indexing
* `--host` / `--port` – server options
* `--trust-cache` – **cache-only mode**: load `.index_cache` and skip fingerprint/rebuild

**Environment variables**

* `OPENAI_API_KEY` – required for chat (and embeddings if rebuilding)
* `TRUST_CACHE=1` – same as `--trust-cache`

---

## Cache-only deploy (Hugging Face or any host)

Deploy with **no raw documents**—just the **prebuilt cache**:

```
knowledge_base/.index_cache/
  ├─ embeddings.npy
  ├─ texts.pkl
  ├─ metadatas.pkl
  └─ fingerprint.json     # optional in trust-cache mode
```

Run with:

```bash
export OPENAI_API_KEY=sk-...
export TRUST_CACHE=1
python chatbot.py --kb ./knowledge_base
```

On Hugging Face Spaces, set:

* **Secret**: `OPENAI_API_KEY`
* **Variable**: `TRUST_CACHE=1`

> If `embeddings.npy` > 100MB, use **Git LFS**:
>
> ```bash
> git lfs install
> git lfs track "knowledge_base/.index_cache/embeddings.npy"
> git add .gitattributes knowledge_base/.index_cache/*
> git commit -m "Add prebuilt index cache"
> git push
> ```

---

## 5) Update history

* **v1.2.0 — 2025-09-18**

  * `prompts.yaml` override for system prompt
  * Dark/Light header styling + customizable hero card
  * Footer controls (Index Status + “Refresh Knowledge Base”), compact status field
  * “Send” emphasized (`variant="primary"`)
  * Cache-only deploy option (`--trust-cache` / `TRUST_CACHE=1`)

* **v1.1.0 — 2025-09-11**

  * URL ingestion via `knowledge_base/url_list.txt` (HTML/PDF/text)
  * Robust header image handling

* **v1.0.0 — 2025-09-10**

  * Initial RAG chatbot with local KB indexing, citations, session memory

---

## 6) About

**HarpyBot** is maintained by the **Cornell Genomics Innovation Hub (GIH)**.
© **2025 Cornell Genomics Innovation Hub**. All rights reserved.

---

## Appendix

### Project layout (typical)

```
.
├─ chatbot.py                  # main app
├─ prompts.yaml                # optional prompt override
├─ header.md                   # optional hero header (images in same dir)
├─ requirements.txt
└─ knowledge_base/
   ├─ .index_cache/            # embeddings + chunk store (generated or shipped)
   ├─ url_list.txt             # optional URL list (one per line)
   ├─ docs/...                 # your documents (optional if cache-only deploy)
   └─ ...
```

### Customizing the prompt

Provide `prompts.yaml` with a `system_prompt` key:

```yaml
system_prompt: "You are Harpy Bot, a cautious research assistant. Answer ONLY using the provided knowledge base context. If the answer is not clearly supported by the context, say: I don't know - I can only answer questions about information in my knowledge base. Be concise. Always include source numbers at the end if any are used."
```

If `prompts.yaml` is missing, a generic, safe prompt is used.

### Custom header

Create `header.md` next to `chatbot.py`. Reference images with relative paths:

```markdown
![Harpy](./harpy_bot_logo_text.png)

# Harpy — Knowledge-Base Research Assistant
```

The app’s CSS targets a `.hero-box`/`#hero-box` wrapper to look good in both light and dark modes.
