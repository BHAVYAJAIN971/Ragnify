"""
Text Chunker — splits text into token-aware overlapping chunks.
Each chunk includes metadata: source label, chunk index, char offsets.
"""
import logging
from typing import List, Dict, Any

import tiktoken

from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Use cl100k_base tokenizer (same as GPT-4 / text-embedding-3-small)
_enc = tiktoken.get_encoding("cl100k_base")


def _tokenize(text: str) -> List[int]:
    return _enc.encode(text, disallowed_special=())


def _detokenize(tokens: List[int]) -> str:
    return _enc.decode(tokens)


def chunk_text(
    text: str,
    source: str,
    doc_id: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Chunk a piece of text into overlapping token-based segments.
    Returns list of chunk dicts.
    """
    if not text or not text.strip():
        return []

    tokens = _tokenize(text)
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = _detokenize(chunk_tokens).strip()

        if chunk_text_str:
            chunks.append({
                "chunk_id":    f"{doc_id}_{source}_{chunk_idx}",
                "doc_id":      doc_id,
                "source":      source,
                "chunk_index": chunk_idx,
                "text":        chunk_text_str,
                "token_count": len(chunk_tokens),
            })
            chunk_idx += 1

        # Move forward by (chunk_size - overlap)
        start += max(1, chunk_size - overlap)

        if end == len(tokens):
            break

    logger.debug(f"Chunked '{source}': {len(chunks)} chunks from {len(tokens)} tokens")
    return chunks


def chunk_all_sources(
    text_by_source: List[tuple],
    doc_id: str,
) -> List[Dict[str, Any]]:
    """
    Chunk all (source_label, text) pairs.
    Returns flat list of all chunks across all sources.
    """
    all_chunks = []
    for source_label, text in text_by_source:
        chunks = chunk_text(text, source_label, doc_id)
        all_chunks.extend(chunks)
    logger.info(f"Total chunks for doc {doc_id}: {len(all_chunks)}")
    return all_chunks
