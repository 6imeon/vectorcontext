"""The experimental conditions. Each is a factory `make(docs, index, findex, **kw) -> Session`;
`Session.ask(question)` returns (answer_text, meter, trace) and keeps conversation state for multi-turn use."""
from __future__ import annotations
import json
from llm import chat, Meter
from sandbox import Sandbox
from index import Index, FactIndex

ANSWER_RULE = ("When you know the answer, reply with a short justification followed by a final line of the form "
               "'ANSWER: <answer>'. For lists, give a comma-separated list. For counts, give just the number.")
MASK_NOTE = "[earlier tool output elided to save context; re-run the call if you need it again]"

class Session:
    """Conversation state for one condition. tools=None -> plain chat. `retriever(text)` (vector_rag) prepends passages each turn.
    `corpus_text` (baseline) goes into the first user message only. mask_keep: within-turn observation masking (keep last k tool outputs).
    mask_prev_turns: elide tool outputs of earlier turns before a new question (cross-turn folding)."""
    def __init__(self, system, tools=None, dispatch=None, max_turns=10, mask_keep=None, mask_prev_turns=False, corpus_text=None, retriever=None):
        self.messages = [{"role": "system", "content": system}]
        self.tools, self.dispatch, self.max_turns = tools, dispatch, max_turns
        self.mask_keep, self.mask_prev_turns, self.corpus_text, self.retriever = mask_keep, mask_prev_turns, corpus_text, retriever
        self.turn_no = 0
    def ask(self, text):
        m = Meter(); trace = []
        self.turn_no += 1
        if self.mask_prev_turns:
            for i, mm in enumerate(self.messages):
                if mm["role"] == "tool" and mm["content"] != MASK_NOTE: self.messages[i] = {**mm, "content": MASK_NOTE}
        user = f"Question: {text}"
        if self.corpus_text is not None and self.turn_no == 1: user = f"<documents>\n{self.corpus_text}\n</documents>\n\n" + user
        if self.retriever is not None:
            ctx, hits = self.retriever(text, m); trace.append(dict(hits=hits)); user = f"<passages>\n{ctx}\n</passages>\n\n" + user
        self.messages.append({"role": "user", "content": user})
        ans = self._loop(m, trace) if self.tools else (chat(self.messages, m).content or "")
        self.messages.append({"role": "assistant", "content": ans})
        return ans, m, trace
    def _loop(self, m, trace):
        messages = self.messages; start = len(messages)
        for turn in range(self.max_turns):
            force_final = turn == self.max_turns - 1
            if self.mask_keep is not None:
                tool_idx = [i for i, mm in enumerate(messages) if mm["role"] == "tool" and i >= start]
                for i in tool_idx[:-self.mask_keep] if self.mask_keep else tool_idx:
                    if messages[i]["content"] != MASK_NOTE: messages[i] = {**messages[i], "content": MASK_NOTE}
            msg = chat(messages, m, tools=None if force_final else self.tools)
            if not msg.tool_calls: return msg.content or ""
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                m.tool_calls += 1
                try: args = json.loads(tc.function.arguments or "{}")
                except Exception: args = {"code": tc.function.arguments}
                out = self.dispatch(tc.function.name, args)
                trace.append(dict(turn=turn, tool=tc.function.name, args=args, out=out[:2000]))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
            if force_final and messages[-1]["role"] == "tool":
                messages.append({"role": "user", "content": "You are out of tool calls. Give your best final answer now. " + ANSWER_RULE})
        return chat(messages, m).content or ""

def _docs_ns(docs): return [dict(id=d.id, type=d.type, title=d.title, text=d.text) for d in docs]
def _types(docs): return ", ".join(sorted({d.type for d in docs}))

# ---------------------------------------------------------------- 1. baseline: whole corpus in the prompt
def baseline(docs, index=None, findex=None, **kw):
    corpus = "\n\n====\n\n".join(f"[doc {d.id}]\n{d.text}" for d in docs)
    return Session("You answer questions about an internal engineering knowledge base. Use only the documents provided. " + ANSWER_RULE, corpus_text=corpus)

# ---------------------------------------------------------------- 2. REPL workspace: corpus lives in a python variable
PY_TOOL = {"type":"function","function":{"name":"python","description":"Execute Python code in a persistent REPL and return stdout (truncated). The variable `docs` is a list of dicts with keys id, type, title, text.","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}}

def repl(docs, index=None, findex=None, output_limit=4000, max_turns=12, mask_prev_turns=False, **kw):
    sb = Sandbox({"docs": _docs_ns(docs)}, output_limit=output_limit)
    system = ("You answer questions about an internal engineering knowledge base. The documents are NOT in your context; they are loaded in a Python REPL "
              f"as the variable `docs` (list of {len(docs)} dicts with keys id, type, title, text; types: {_types(docs)}). "
              f"Use the `python` tool to search, slice and aggregate (re, json, collections are imported). Output is truncated to {output_limit} chars, so print only what you need. "
              "Services are often referred to by an alias like 'the Billing API' as well as their canonical name like 'billing-service'; both appear in the service overview. "
              "Be efficient: prefer a few well-targeted searches over dumping whole documents. " + ANSWER_RULE)
    return Session(system, [PY_TOOL], lambda n, a: sb.run(a.get("code", "")), max_turns=max_turns, mask_prev_turns=mask_prev_turns)

# ---------------------------------------------------------------- 3. vector RAG: single-shot top-k retrieval per turn
def vector_rag(docs, index: Index, findex=None, k=8, mode="dense", **kw):
    def retriever(text, m):
        hits = index.dense(text, k, m) if mode == "dense" else index.hybrid(text, k, m)
        return index.render(hits, with_ids=False), [index.chunks[i]["doc_id"] for i, _ in hits]
    return Session("You answer questions about an internal engineering knowledge base using the retrieved passages. If the passages are insufficient, answer as best you can. " + ANSWER_RULE, retriever=retriever)

# ---------------------------------------------------------------- 4. vectorcontext v1-v3: agentic hybrid chunk search (+ optional python)
SEARCH_TOOL = {"type":"function","function":{"name":"search","description":"Semantic + keyword hybrid search over the knowledge base. Returns the top-k chunks with their doc ids. Use several focused queries rather than one broad one.","parameters":{"type":"object","properties":{"query":{"type":"string"},"k":{"type":"integer","description":"number of chunks, default 6, max 30"}},"required":["query"]}}}
OPEN_TOOL = {"type":"function","function":{"name":"open_doc","description":"Return the full text of a document by doc id (e.g. 'svc-billing-api', 'inc-INC-2041', 'team-payments').","parameters":{"type":"object","properties":{"doc_id":{"type":"string"}},"required":["doc_id"]}}}
LIST_TOOL = {"type":"function","function":{"name":"list_docs","description":"List document ids and titles, optionally filtered by type. Cheap: use it to enumerate before aggregating.","parameters":{"type":"object","properties":{"type":{"type":"string"}},"required":[]}}}
VPY_TOOL = {"type":"function","function":{"name":"python","description":"Execute Python in a persistent REPL. Available: `docs` (list of dicts id,type,title,text), `search(query, k=10)` -> list of chunk dicts (doc_id,title,text) via hybrid vector+keyword retrieval, `open_doc(doc_id)` -> text. Use it for aggregation over many documents (counting, summing) where a single search would not suffice. stdout is truncated.","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}}

def vectorcontext(docs, index: Index, findex=None, version="v1", k_default=6, max_turns=10, output_limit=3000, mask_keep=None, mask_prev_turns=False, **kw):
    def search(query, k=k_default):
        k = max(1, min(int(k or k_default), 30)); return index.hybrid(query, k, _m[0])
    _m = [Meter()]
    def open_doc(doc_id):
        d = index.docs.get(doc_id); return d.text if d else f"(no such doc: {doc_id}; use search or list_docs)"
    def list_docs(t=None): return "\n".join(f"{d.id} | {d.title}" for d in index.docs.values() if not t or d.type == t)
    ns = {"docs": _docs_ns(docs), "search": lambda query, k=10: [dict(doc_id=index.chunks[i]["doc_id"], title=index.chunks[i]["title"], text=index.chunks[i]["text"]) for i, _ in search(query, k)], "open_doc": open_doc}
    sb = Sandbox(ns, output_limit=output_limit)
    def dispatch(name, a):
        if name == "search": return index.render(search(a.get("query", ""), a.get("k")))
        if name == "open_doc": return open_doc(a.get("doc_id", ""))
        if name == "list_docs": return list_docs(a.get("type"))
        if name == "python": return sb.run(a.get("code", ""))
        return f"unknown tool {name}"
    tools = {"v1": [SEARCH_TOOL, OPEN_TOOL], "v2": [SEARCH_TOOL, OPEN_TOOL, LIST_TOOL], "v3": [SEARCH_TOOL, OPEN_TOOL, LIST_TOOL, VPY_TOOL]}[version]
    system = ("You answer questions about an internal engineering knowledge base. The documents are NOT in your context. "
              f"The base has {len(docs)} documents (types: {_types(docs)}). "
              "Use the tools to find evidence. Services have a canonical name like 'billing-service' and an alias like 'the Billing API'; the service overview doc (id 'svc-<name>') lists both plus port, language, owner team, on-call, datastore, region, dependencies. "
              "Team pages (id 'team-<name>') list the lead and Slack channel. Incident reports (id 'inc-INC-nnnn') list severity, affected service, duration and root cause. "
              + ("For counting or summing across many documents, use the python tool and iterate over `docs` programmatically instead of reading them one by one. " if version == "v3" else
                 "For counting or summing across many documents, use list_docs and search with a large k, and be systematic. ")
              + "Stop searching once you have the evidence. " + ANSWER_RULE)
    if mask_keep is not None: system += " Older tool outputs are elided from your history, so write down key facts you learn in your reply text before the next call."
    s = Session(system, tools, dispatch, max_turns=max_turns, mask_keep=mask_keep, mask_prev_turns=mask_prev_turns)
    _ask = s.ask
    def ask(text):
        _m[0] = Meter(); ans, m, tr = _ask(text); m.embed_tokens += _m[0].embed_tokens; m.embed_cost += _m[0].embed_cost; return ans, m, tr
    s.ask = ask
    return s

# ---------------------------------------------------------------- 5. vectorcontext v4/v5: fact-card semantic layer + python over cards
FSEARCH_TOOL = {"type":"function","function":{"name":"search","description":"Hybrid semantic+keyword search over per-document fact cards (normalized key:value facts extracted from each document). Returns the top-k matching documents' fact cards. Paraphrase-robust; use several focused queries.","parameters":{"type":"object","properties":{"query":{"type":"string"},"k":{"type":"integer","description":"default 5, max 40"}},"required":["query"]}}}
TSEARCH_TOOL = {"type":"function","function":{"name":"search_text","description":"Hybrid semantic+keyword search over raw document passages (not fact cards). Use it for details that fact cards do not carry: timelines, exact wording, dates, anything narrative. Returns top-k passages with doc ids.","parameters":{"type":"object","properties":{"query":{"type":"string"},"k":{"type":"integer","description":"default 5, max 20"}},"required":["query"]}}}
FPY_TOOL = {"type":"function","function":{"name":"python","description":"Persistent Python REPL. Variables: `facts` (list of dicts, one per document, with doc_id, type, title and normalized fact keys), `docs` (raw docs: id,type,title,text), `search(query,k)` -> list of fact dicts, `open_doc(doc_id)` -> raw text. Use it to count/sum/filter over `facts` for aggregation questions; inspect a couple of cards first to learn the keys (they are harmonized per doc type but values can vary in form, e.g. a service may be named by alias). stdout truncated.","parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}}

PROMPTS = {
 "default": ("For single facts: use `search`. For counting/summing/listing across many documents: use `python` over `facts` (match on normalized values, handle aliases by joining service cards on subject/aliases). "
             "If a card seems incomplete, `open_doc` gives the raw text. Stop once you have the evidence. "),
 "minimal": "Use the tools to find evidence, then answer. ",
 "guided": ("Strategy: (1) single facts -> `search`; (2) counts, sums, lists, 'never'/'no'/'not' questions -> `python` over `facts` with explicit set differences over the full list of entities (never guess from search results alone); "
            "(3) 'currently'/'now'/'as of' questions -> some documents record CHANGES that supersede facts stated elsewhere; check cards of update/change records for the entity and apply the latest dated record; "
            "(4) narrative details such as timelines, exact times or wording -> `search_text` or `open_doc` on the raw document, not the card; "
            "(5) incident cards distinguish the primary affected service from services mentioned as collateral/side effects; do not conflate them. "
            "Resolve aliases by joining on subject/aliases. Verify list answers by printing them. Stop once you have the evidence. "),
}

def vectorcontext_v4(docs, index, findex: FactIndex, max_turns=10, output_limit=3000, mask_keep=None, mask_prev_turns=False, prompt="default", text_search=False, **kw):
    _m = [Meter()]
    def search(query, k=5):
        k = max(1, min(int(k or 5), 40)); return findex.hybrid(query, k, _m[0])
    def search_text(query, k=5):
        k = max(1, min(int(k or 5), 20)); return index.render(index.hybrid(query, k, _m[0]))
    def open_doc(doc_id):
        d = findex.docs.get(doc_id); return d.text if d else f"(no such doc: {doc_id})"
    ns = {"facts": findex.cards, "docs": _docs_ns(docs), "search": lambda query, k=10: [findex.cards[i] for i in search(query, k)], "open_doc": open_doc}
    sb = Sandbox(ns, output_limit=output_limit)
    def dispatch(name, a):
        if name == "search": return findex.render(search(a.get("query", ""), a.get("k")))
        if name == "search_text": return search_text(a.get("query", ""), a.get("k"))
        if name == "open_doc": return open_doc(a.get("doc_id", ""))
        if name == "python": return sb.run(a.get("code", ""))
        return f"unknown tool {name}"
    types = sorted({c["type"] for c in findex.cards})
    keys = {t: sorted({k for c in findex.cards if c["type"] == t for k in c if k not in ("doc_id", "type", "title")})[:25] for t in types}
    system = ("You answer questions about an internal engineering knowledge base. The documents are NOT in your context. Each document has been distilled into a fact card "
              f"(normalized key: value facts). There are {len(docs)} documents. Fact keys by doc type: {json.dumps(keys)}. "
              "Services have a canonical name like 'billing-service' and an alias like 'the Billing API'; cards may use either, and incident cards may name the affected service under subject or affected_service. "
              + PROMPTS[prompt] + ANSWER_RULE)
    if findex.resolved: system += (" Change records have been resolved at index time: a card's own fields always state what its source document says; where a later change record superseded a fact, "
                                   "the card also has `current` ({field: latest value}) and `history` ([{field, from, to, date, source}]). Use `current` for 'currently/now/as of today' questions and the plain fields for 'according to the overview' questions.")
    if mask_keep is not None: system += " Older tool outputs are elided from your history, so write down key facts you learn in your reply text before the next call."
    tools = [FSEARCH_TOOL, OPEN_TOOL, FPY_TOOL] + ([TSEARCH_TOOL] if text_search else [])
    s = Session(system, tools, dispatch, max_turns=max_turns, mask_keep=mask_keep, mask_prev_turns=mask_prev_turns)
    _ask = s.ask
    def ask(text):
        _m[0] = Meter(); ans, m, tr = _ask(text); m.embed_tokens += _m[0].embed_tokens; m.embed_cost += _m[0].embed_cost; return ans, m, tr
    s.ask = ask
    return s

CONDITIONS = {
    "baseline": baseline,
    "repl": repl,
    "vector_rag": vector_rag,
    "vectorcontext_v1": lambda d, index, findex=None, **kw: vectorcontext(d, index, version="v1", **kw),
    "vectorcontext_v2": lambda d, index, findex=None, **kw: vectorcontext(d, index, version="v2", **kw),
    "vectorcontext_v3": lambda d, index, findex=None, **kw: vectorcontext(d, index, version="v3", **kw),
    "vectorcontext_v4": lambda d, index, findex=None, **kw: vectorcontext_v4(d, index, findex, **kw),
    # v5 = v4 + raw-passage search tool + guided strategy prompt + (index-time) supersession resolution, see facts.resolve_updates
    "vectorcontext_v5": lambda d, index, findex=None, **kw: vectorcontext_v4(d, index, findex, **{"prompt": "guided", "text_search": True, **kw}),
}
