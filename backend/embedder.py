"""
Embedder — wraps Google Gemini text-embedding-001 with batching & caching.
Produces 3072-dimensional embeddings.
"""
import hashlib
import json
import logging
import os
import time
from typing import List, Dict

import numpy as np
from google import genai

from config import GEMINI_API_KEY, EMBEDDING_MODEL, DATA_DIR

logger = logging.getLogger(__name__)

# Gemini client (module-level so settings endpoint can update api_key)
client = genai.Client(api_key=GEMINI_API_KEY)

# In-memory embedding cache (chunk_hash → embedding)
_cache: Dict[str, List[float]] = {}
_CACHE_FILE = os.path.join(DATA_DIR, ".embedding_cache.json")


def _load_cache():
    global _cache
    if os.path.exists(_CACHE_FILE):
        try:
            with open(_CACHE_FILE, "r") as f:
                _cache = json.load(f)
            logger.info(f"Loaded {len(_cache)} cached embeddings")
        except Exception:
            _cache = {}


def _save_cache():
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump(_cache, f)
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _embed_batch_gemini(texts: List[str]) -> List[List[float]]:
    """Call Gemini embed_content for a batch (max 100 texts)."""
    import config as cfg
    # Use current API key (may have been updated via settings)
    curr_client = genai.Client(api_key=cfg.GEMINI_API_KEY)

    max_attempts = 5  # More retries for large docs hitting rate limits
    for attempt in range(max_attempts):
        try:
            resp = curr_client.models.embed_content(
                model=cfg.EMBEDDING_MODEL,
                contents=texts,
            )
            return [e.values for e in resp.embeddings]
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                # Progressive backoff: 5s, 15s, 30s, 60s, 90s
                wait = min(5 * (2 ** attempt), 90)
                logger.warning(f"Rate limit hit, waiting {wait}s (attempt {attempt+1}/{max_attempts})...")
                time.sleep(wait)
            else:
                if attempt < max_attempts - 1:
                    wait = 2 * (attempt + 1)
                    logger.warning(f"Embed attempt {attempt+1} failed: {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
    raise RuntimeError(f"Embedding failed after {max_attempts} attempts")


# Gemini embedding supports up to 100 texts per call
_GEMINI_BATCH_SIZE = 20  # Keep small to avoid rate limits


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts.
    Returns numpy array of shape (len(texts), embedding_dim), L2-normalized.
    Uses caching and batching.
    """
    _load_cache()
    embeddings = []
    to_embed_indices = []
    to_embed_texts = []

    # Check cache first
    for i, text in enumerate(texts):
        h = _text_hash(text)
        if h in _cache:
            embeddings.append((i, _cache[h]))
        else:
            to_embed_indices.append(i)
            to_embed_texts.append(text)

    # Batch-embed uncached texts
    if to_embed_texts:
        total_batches = (len(to_embed_texts) + _GEMINI_BATCH_SIZE - 1) // _GEMINI_BATCH_SIZE
        logger.info(f"Embedding {len(to_embed_texts)} texts in {total_batches} batches (size={_GEMINI_BATCH_SIZE})")
        new_embeddings = []

        for batch_num, start in enumerate(range(0, len(to_embed_texts), _GEMINI_BATCH_SIZE), 1):
            batch = to_embed_texts[start:start + _GEMINI_BATCH_SIZE]
            logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} texts)...")
            batch_vecs = _embed_batch_gemini(batch)
            new_embeddings.extend(batch_vecs)
            # Rate limit delay: Gemini free tier allows ~15 RPM for embeddings
            # Wait 4s between batches to stay well within limits
            if start + _GEMINI_BATCH_SIZE < len(to_embed_texts):
                time.sleep(4)

        # Cache and collect
        for orig_idx, vec in zip(to_embed_indices, new_embeddings):
            h = _text_hash(texts[orig_idx])
            _cache[h] = vec
            embeddings.append((orig_idx, vec))

        _save_cache()

    # Sort by original index and stack
    embeddings.sort(key=lambda x: x[0])
    matrix = np.array([vec for _, vec in embeddings], dtype=np.float32)

    # L2-normalize for cosine similarity via inner product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    matrix = matrix / norms

    return matrix


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, dim)."""
    return embed_texts([query])
