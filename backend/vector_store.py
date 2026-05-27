"""
Vector Store — FAISS-based per-document index with metadata.
"""
import json
import logging
import os
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

from config import DATA_DIR, TOP_K

logger = logging.getLogger(__name__)


def _index_dir(doc_id: str) -> str:
    path = os.path.join(DATA_DIR, doc_id)
    os.makedirs(path, exist_ok=True)
    return path


def _index_path(doc_id: str) -> str:
    return os.path.join(_index_dir(doc_id), "index.faiss")


def _meta_path(doc_id: str) -> str:
    return os.path.join(_index_dir(doc_id), "metadata.json")


def build_index(doc_id: str, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
    """
    Build and save a FAISS index for a document.
    embeddings: (N, dim) float32 numpy array, L2-normalized.
    chunks: list of chunk dicts with at least 'text', 'source', 'chunk_id'.
    """
    if embeddings.shape[0] == 0:
        logger.warning(f"No embeddings to index for {doc_id}")
        return

    dim = embeddings.shape[1]
    n   = embeddings.shape[0]

    # Use flat inner-product index (exact, cosine after normalization)
    # For very large docs (>10k chunks), switch to IVF
    if n > 5000:
        nlist = min(int(n ** 0.5), 256)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        logger.info(f"Built IVF index: {n} vecs, {nlist} cells")
    else:
        index = faiss.IndexFlatIP(dim)
        logger.info(f"Built Flat index: {n} vecs, dim={dim}")

    index.add(embeddings)

    # Save index
    faiss.write_index(index, _index_path(doc_id))

    # Save metadata (parallel to embeddings)
    meta = [
        {
            "chunk_id":    c["chunk_id"],
            "source":      c["source"],
            "chunk_index": c["chunk_index"],
            "text":        c["text"],
            "token_count": c.get("token_count", 0),
        }
        for c in chunks
    ]
    with open(_meta_path(doc_id), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    logger.info(f"Index saved for doc={doc_id}: {n} chunks")


def search(doc_id: str, query_vec: np.ndarray, k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Search the FAISS index for a document.
    query_vec: (1, dim) float32 numpy array, L2-normalized.
    Returns list of chunk dicts with added 'score'.
    """
    ipath = _index_path(doc_id)
    mpath = _meta_path(doc_id)

    if not os.path.exists(ipath):
        raise FileNotFoundError(f"No index found for document '{doc_id}'")

    index = faiss.read_index(ipath)
    with open(mpath, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    actual_k = min(k, index.ntotal)
    if actual_k == 0:
        return []

    scores, indices = index.search(query_vec, actual_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        chunk = dict(metadata[idx])
        chunk["score"] = float(score)
        results.append(chunk)

    # Sort by score descending (best first)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def index_exists(doc_id: str) -> bool:
    return os.path.exists(_index_path(doc_id))


def list_indexes() -> List[str]:
    """Return all doc_ids that have been indexed."""
    if not os.path.exists(DATA_DIR):
        return []
    return [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and os.path.exists(os.path.join(DATA_DIR, d, "index.faiss"))
    ]


def delete_index(doc_id: str) -> bool:
    """Delete a document's index and metadata."""
    import shutil
    idir = _index_dir(doc_id)
    if os.path.exists(idir):
        shutil.rmtree(idir)
        logger.info(f"Deleted index for {doc_id}")
        return True
    return False
