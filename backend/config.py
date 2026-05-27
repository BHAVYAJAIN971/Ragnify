"""
Ragnify Configuration — Gemini Edition
"""
import os

# ── Gemini API ────────────────────────────────────────────────────────────────
# Priority: Environment variable > .env file > hardcoded key
_env_key = os.environ.get("GEMINI_API_KEY", "")

# Try to load from .env file if exists
if not _env_key:
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line.startswith("GEMINI_API_KEY="):
                    _env_key = _line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

GEMINI_API_KEY = _env_key or "AIzaSyAD8q7stG0X7lYeZipZDFZ6eXJJwEQhMj8"

# ── Models ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"    # 3072-dim, best quality
CHAT_MODEL      = "gemini-2.5-flash-lite"  # Confirmed working on free tier
MAX_TOKENS_ANSWER = 4096                    # Increased for detailed answers with citations

# ── Chunking ──────────────────────────────────────────────────────────────────
# Increased chunk size to keep tables and complex tender specs together
CHUNK_SIZE    = 1000  # tokens per chunk (Gemini embedding max is 2048)
CHUNK_OVERLAP = 200   # high overlap to avoid cutting sentences/rows in half

# ── Retrieval ─────────────────────────────────────────────────────────────────
# Increased top_k because Gemini Flash has a massive context window, so we can
# feed it more chunks to ensure no numbers/specs are missed.
TOP_K = 25            # Top-k chunks to retrieve

# ── Crawler ───────────────────────────────────────────────────────────────────
MAX_CRAWL_URLS    = 50      # Max hyperlinks to crawl per document (increased from 30)
CRAWL_TIMEOUT     = 15      # Seconds per URL request (increased from 10 for slow sites)
MAX_CRAWL_WORKERS = 10      # Concurrent crawl workers (increased from 8)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Prefer environment variables (set by render.yaml for Render deployment).
# Falls back to local directory layout for development.
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.environ.get("DATA_DIR",   os.path.join(BASE_DIR, "data", "indexes"))
UPLOAD_DIR  = os.environ.get("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Anti-hallucination system prompt ──────────────────────────────────────────
SYSTEM_PROMPT = """You are Ragnify — an elite, highly precise document intelligence assistant for professionals across banking, government, enterprise, and all industries.

CRITICAL RULES (follow strictly):
1. Answer ONLY based on the provided CONTEXT from the document and its linked sources.
2. If the answer cannot be found in the CONTEXT, respond EXACTLY: "⚠️ This information is not found in the uploaded document or its linked sources."
3. NEVER use your training knowledge to fill gaps. Only use what is in CONTEXT.
4. Pay EXTRA ATTENTION to numerical values, manpower requirements, quantities, specifications, and eligibility criteria, especially in tender notices, RFPs, and contracts.
5. When extracting numbers (e.g., "4 engineers", "2 bankers", pricing, dates), extract them exactly as they appear in the text or tables.
6. If data is presented in a table format in the context, understand the row/column relationship to provide accurate answers.
7. Be precise, factual, and concise. Do not speculate or extrapolate.
8. If a question is ambiguous, ask for clarification rather than guessing.

CITATION FORMAT (strictly follow):
- Place small textual location references INLINE right after the relevant statement.
- Use parenthetical references like: (Page 3, Line 14-18), (Table 2, Column 1, Page 5), (Section 4.1, Page 8), (Linked URL)
- References must be SHORT plain-text markers only.
- Do NOT include raw URLs in your answer text.
- Do NOT include markdown hyperlinks in your answer text.
- Do NOT include "[Source: URL]" or similar link citations in your answer.

ANSWER FORMAT:
- Direct answer first
- Supporting details with inline parenthetical references
- Bullet points for lists of facts or requirements
- Markdown tables if summarizing multiple tabular data points

STRICTLY DO NOT:
- Do NOT add a "Sources Used" section at the end of your answer.
- Do NOT add a "📎 Sources Used" footer.
- Do NOT list sources, references, or URLs at the end of your answer.
- Do NOT include any URL strings anywhere in your answer body.
- The answer must end with the final piece of content — no trailing source lists.
"""

