"""Chunking, embedding (cached on disk), dense + BM25 hybrid retrieval."""
from __future__ import annotations
import hashlib, json, os, re, threading, uuid
_save_lock = threading.Lock()
import numpy as np
from rank_bm25 import BM25Okapi
from llm import embed, EMBED_MODEL

CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "embed_cache.json")

def chunk_docs(docs, max_chars=700, contextual=True):
    """Paragraph-based chunks, each prefixed with the doc title (contextual header)."""
    chunks=[]
    for d in docs:
        paras=[p.strip() for p in d.text.split("\n\n") if p.strip()]
        cur=""
        for p in paras:
            if cur and len(cur)+len(p) > max_chars:
                chunks.append(dict(doc_id=d.id, title=d.title, type=d.type, text=cur)); cur=""
            cur = (cur+"\n\n"+p) if cur else p
        if cur: chunks.append(dict(doc_id=d.id, title=d.title, type=d.type, text=cur))
    for i,c in enumerate(chunks):
        c["id"]=i
        c["embed_text"] = (f"[{c['title']}]\n{c['text']}" if contextual else c["text"])
    return chunks

def _tok(s): return re.findall(r"[a-z0-9]+", s.lower())

def _atomic_save(path, data):
    """Merge with what is on disk (other processes may have written) and replace atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _save_lock:  # threads in one process share the pid; the lock serializes read-merge-write, the uuid keeps tmp names unique
        try:
            if os.path.exists(path): data = {**json.load(open(path)), **data}
        except Exception: pass
        tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"; json.dump(data, open(tmp, "w")); os.replace(tmp, path)

class Index:
    def __init__(self, docs, contextual=True, max_chars=700):
        self.docs = {d.id: d for d in docs}
        self.chunks = chunk_docs(docs, max_chars=max_chars, contextual=contextual)
        self.bm25 = BM25Okapi([_tok(c["embed_text"]) for c in self.chunks])
        self.emb = self._embed_cached([c["embed_text"] for c in self.chunks])
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True)
    def _embed_cached(self, texts):
        cache = {}
        if os.path.exists(CACHE): cache = json.load(open(CACHE))
        keys = [hashlib.sha1((EMBED_MODEL+"|"+t).encode()).hexdigest() for t in texts]
        missing = [i for i,k in enumerate(keys) if k not in cache]
        if missing:
            vecs = embed([texts[i] for i in missing])
            for i,v in zip(missing, vecs): cache[keys[i]] = v
            _atomic_save(CACHE, cache)
        return np.array([cache[k] for k in keys], dtype=np.float32)
    def dense(self, query, k=10, meter=None):
        q = np.array(embed([query], meter)[0], dtype=np.float32); q/=np.linalg.norm(q)
        s = self.emb @ q
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]
    def lexical(self, query, k=10):
        s = self.bm25.get_scores(_tok(query))
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx if s[i] > 0]
    def hybrid(self, query, k=10, meter=None, rrf_k=60):
        """Reciprocal rank fusion of dense and BM25 rankings."""
        scores={}
        for lst in (self.dense(query, k*3, meter), self.lexical(query, k*3)):
            for rank,(i,_) in enumerate(lst):
                scores[i] = scores.get(i,0) + 1.0/(rrf_k+rank+1)
        top = sorted(scores.items(), key=lambda x:-x[1])[:k]
        return [(i,s) for i,s in top]
    def render(self, hits, with_ids=True):
        out=[]
        for i,s in hits:
            c=self.chunks[i]
            out.append((f"[chunk {i} | doc {c['doc_id']} | {c['title']}]\n" if with_ids else f"[{c['title']}]\n") + c["text"])
        return "\n\n---\n\n".join(out)

class FactIndex:
    """Doc-level index over fact cards: dense over card text, BM25 over raw text + card text, RRF fused."""
    def __init__(self, docs, cards, resolved=False):
        from facts import card_text
        self.docs = {d.id: d for d in docs}; self.resolved = resolved
        self.cards = cards; self.card_by_doc = {c["doc_id"]: c for c in cards}
        self.texts = [card_text(c) for c in cards]
        raw = {d.id: d.text for d in docs}
        self.bm25 = BM25Okapi([_tok(t + " " + raw.get(c["doc_id"], "")) for t, c in zip(self.texts, cards)])
        tmp = Index.__new__(Index); self.emb = Index._embed_cached(tmp, self.texts)
        self.emb /= np.linalg.norm(self.emb, axis=1, keepdims=True)
    def hybrid(self, query, k=8, meter=None, rrf_k=60):
        q = np.array(embed([query], meter)[0], dtype=np.float32); q /= np.linalg.norm(q)
        d = self.emb @ q; dense = list(np.argsort(-d)[:k*3])
        b = self.bm25.get_scores(_tok(query)); lex = [i for i in np.argsort(-b)[:k*3] if b[i] > 0]
        scores = {}
        for lst in (dense, lex):
            for rank, i in enumerate(lst): scores[int(i)] = scores.get(int(i), 0) + 1.0/(rrf_k+rank+1)
        return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]]
    def render(self, idxs, max_chars=600):
        return "\n".join(f"- doc {self.cards[i]['doc_id']}: {self.texts[i][:max_chars]}" for i in idxs)
