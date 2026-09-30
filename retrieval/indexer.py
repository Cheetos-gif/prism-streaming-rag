"""
Corpus indexer for the retrieval module.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        """Pure-Python fallback implementation of BM25Okapi."""
        def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
            import math
            self.k1 = k1
            self.b = b
            self.epsilon = epsilon
            self.corpus_size = len(corpus)
            self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 1.0
            self.doc_freqs = []
            self.idf = {}
            self.doc_len = [len(doc) for doc in corpus]
            nd = {}
            for document in corpus:
                frequencies = {}
                for word in document:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in frequencies:
                    nd[word] = nd.get(word, 0) + 1

            idf_sum = 0
            negative_idfs = []
            for word, freq in nd.items():
                idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
                self.idf[word] = idf
                idf_sum += idf
                if idf < 0:
                    negative_idfs.append(word)

            average_idf = idf_sum / len(self.idf) if self.idf else 0
            eps = self.epsilon * average_idf
            for word in negative_idfs:
                self.idf[word] = eps

        def get_scores(self, query):
            score = np.zeros(self.corpus_size, dtype=np.float32)
            doc_len = np.array(self.doc_len, dtype=np.float32)
            for q in query:
                q_freq = np.array([(doc.get(q, 0)) for doc in self.doc_freqs], dtype=np.float32)
                score += (self.idf.get(q, 0)) * (q_freq * (self.k1 + 1)) / (
                    q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                )
            return score

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    class SentenceTransformer:
        """Lightweight hash-projection fallback when sentence-transformers is not locally installed."""
        def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
            self.model_name = model_name
            self.dim = 384

        def encode(self, texts, convert_to_numpy: bool = True):
            is_single = isinstance(texts, str)
            if is_single:
                texts = [texts]
            vecs = []
            for text in texts:
                words = re.findall(r'\b[a-z0-9]+\b', text.lower())
                v = np.zeros(self.dim, dtype=np.float32)
                for w in words:
                    h = abs(hash(w)) % self.dim
                    v[h] += 1.0
                norm = np.linalg.norm(v)
                if norm > 0:
                    v /= norm
                vecs.append(v)
            arr = np.array(vecs, dtype=np.float32)
            return arr[0] if is_single else arr

from shared.schemas import ChunkRecord


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, split on whitespace and punctuation."""
    return [t for t in re.split(r'\W+', text.lower()) if t]


def slugify(text: str) -> str:
    """Convert text to a slug for chunk IDs."""
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def chunk_text(text: str, max_words: int = 300, overlap_words: int = 50) -> list[str]:
    """Chunk text into chunks of max_words with overlap_words."""
    words = text.split()
    if not words:
        return []
    
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_words]
        chunks.append(" ".join(chunk_words))
        i += (max_words - overlap_words)
        if i >= len(words) and len(words) > max_words:
            break
            
    # If no chunks were created but there are words (should not happen with the logic above, but as a fallback)
    if not chunks and words:
        chunks.append(" ".join(words))
        
    return chunks


class CorpusIndex:
    """
    Corpus Index holding chunks, BM25 index, and dense embeddings.
    """
    chunks: list[ChunkRecord]
    chunk_map: dict[str, ChunkRecord]
    bm25: BM25Okapi
    embeddings: np.ndarray
    model: SentenceTransformer

    @classmethod
    def build(cls, corpus_path: str | Path, model_name: str = 'all-MiniLM-L6-v2') -> 'CorpusIndex':
        """
        Build a corpus index from a directory of markdown files.
        """
        corpus_path = Path(corpus_path)
        model = SentenceTransformer(model_name)
        
        chunks = []
        chunk_map = {}
        
        for file_path in corpus_path.rglob("*.md"):
            doc_id = file_path.stem
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            current_section = "§0 Introduction"
            current_section_lines = []
            
            sections: list[tuple[str, str]] = []
            
            for line in lines:
                if line.startswith("## "):
                    if current_section_lines:
                        sections.append((current_section, "\n".join(current_section_lines)))
                    current_section = line[3:].strip()
                    current_section_lines = []
                else:
                    current_section_lines.append(line)
                    
            if current_section_lines:
                sections.append((current_section, "\n".join(current_section_lines)))
                
            for section_title, section_text in sections:
                if not section_text.strip():
                    continue
                
                section_slug = slugify(section_title)
                text_chunks = chunk_text(section_text, max_words=300, overlap_words=50)
                
                for idx, t_chunk in enumerate(text_chunks):
                    if not t_chunk.strip():
                        continue
                        
                    chunk_id = f"{doc_id}_{section_slug}_{idx}"
                    record = ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        section=section_title,
                        text=t_chunk,
                        embedding=None
                    )
                    chunks.append(record)
                    chunk_map[chunk_id] = record

        # BM25 Index
        tokenized_corpus = [tokenize(c.text) for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        
        # Embeddings
        texts = [c.text for c in chunks]
        if texts:
            embeddings = model.encode(texts, convert_to_numpy=True)
            # Ensure it is a 2D numpy array
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
        else:
            embeddings = np.array([])
            
        for i, c in enumerate(chunks):
            if i < len(embeddings):
                c.embedding = embeddings[i].tolist()
                
        index = cls()
        index.chunks = chunks
        index.chunk_map = chunk_map
        index.bm25 = bm25
        index.embeddings = embeddings
        index.model = model
        
        return index
