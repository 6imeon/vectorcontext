"""Wild-idea conditions for the no-documents setting. Each factory takes (chat, art, **kw) -> Session (probe semantics like conditions_chat).
Ideas: no raw window at all; terse ledger serialisation; question-sliced ledger; tools with nothing in the prompt; one-shot self-shorthand;
single-shot RAG over turns; and zero-LLM deterministic transcript reductions (user turns only, clipped assistant turns, regex fact filter, stop-word pruning)."""
from __future__ import annotations
import hashlib, json, os, re
import llm
from llm import Meter, chat
from corpus import Doc
from index import Index, _atomic_save
from conditions import Session
from conditions_chat import SYS, transcript, _preload, _probe, _state_block, chat_state, chat_vc
from chat_facts import CACHE, _pair, ledger_text

def _msgs(block, ack="Understood."):
    return [{"role": "user", "content": block}, {"role": "assistant", "content": ack}]

# ---------------------------------------------------------------- ledger-only variants (no raw window)
def chat_ledger0(chat_obj, art, **kw):
    """chat_state with keep=0: the ledger is the whole context."""
    return chat_state(chat_obj, art, keep=0)

def terse_text(ledger, history=None):
    """One line per entity: 'entity: attr=value; attr=value'; a changed value carries [t<set>; was v1@t1, v2@t2] in chronological order."""
    lines = []
    for e, d in ledger.items():
        parts = []
        for a, v in d.items():
            h = (history or {}).get((e, a), [])
            old = f" [t{h[-1][0]}; was " + ", ".join(f"{json.dumps(pv)}@t{pt}" for pt, pv in h[:-1]) + "]" if len(h) > 1 else ""
            parts.append(f"{a}={json.dumps(v)}{old}")
        lines.append(f"{e}: " + "; ".join(parts))
    return "\n".join(lines)

def chat_terse(chat_obj, art, **kw):
    block = "[Fact ledger folded from the conversation: one line per entity, attr=current value, a changed value carries [turn it was set; was: earlier values in order]]\n" + terse_text(art["ledger"], art["history"])
    return _probe(_preload(Session(SYS), _msgs(block)))

# ---------------------------------------------------------------- question-sliced ledger: hybrid top-k entries + attribute expansion + entity directory
def _entries(art):
    out = []
    for e, d in art["ledger"].items():
        for a, v in d.items():
            h = art["history"].get((e, a), [])
            old = f"   (was: {', '.join(f'{json.dumps(pv)} @t{pt}' for pt, pv in h[:-1])})" if len(h) > 1 else ""
            out.append((e, a, f"{e}.{a} = {json.dumps(v)}  [t{h[-1][0] if h else '?'}]{old}"))
    return out

def chat_slice(chat_obj, art, k=12, **kw):
    ents = _entries(art)
    idx = Index([Doc(f"e{i}", "fact", f"{e}.{a}", txt) for i, (e, a, txt) in enumerate(ents)], contextual=False)
    directory = ", ".join(art["ledger"].keys())
    def retriever(q, m):
        hits = [i for i, _ in idx.hybrid(q, k, m)]
        qt = set(re.findall(r"[a-z0-9]+", q.lower()))
        for i, (e, a, _) in enumerate(ents):   # deterministic expansion: every entry whose attribute name is named in the question (sums/counts/lists need all of them)
            if i not in hits and set(re.findall(r"[a-z0-9]+", a.lower())) - {"k"} & qt: hits.append(i)
        chosen = [ents[i][2] for i in sorted(hits)]
        return ("[Entities in the fact ledger: " + directory + "]\n[Ledger entries selected for this question (entity.attribute = current value [turn], superseded values in parentheses); "
                "an entity.attribute absent here may simply not have been selected, but an entity absent from the directory was never discussed]\n" + "\n".join(chosen)), hits
    return _probe(Session(SYS, retriever=retriever))

# ---------------------------------------------------------------- tools only: nothing in the prompt
def chat_tools0(chat_obj, art, **kw):
    return chat_vc(chat_obj, art, keep=0, state_in_prompt=False)

# ---------------------------------------------------------------- one-shot self-shorthand: the model compresses the whole transcript for itself
SHORT_SYS = ("You will later have to answer detailed questions about the conversation below WITHOUT seeing it again: only the notes you write now. "
             "Write the notes for yourself in the fewest tokens you can. Any shorthand, abbreviations, symbols or tables you like, as long as YOU can decode them. "
             "Must be recoverable: every project fact the user stated (owners, dates, budgets, datastores, priorities, scope changes, names, order of things), every number in pasted tool output, "
             "and when a value was changed, both the old and the current value marked clearly. Skip the assistant's general explanations and the user's side questions entirely. "
             "Output only the notes.")

def shorthand_chat(chat_obj, model=None, cache_path=CACHE, salt=""):
    model = model or llm.MODEL
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    user = "\n\n".join(f"(turn {t['i']})\n{_pair(t)}" for t in chat_obj.turns)
    key = f"shorthand1|{salt}" + hashlib.sha1((model + SHORT_SYS + user).encode()).hexdigest()
    cost = dict(prompt_tokens=0, completion_tokens=0, cost_usd=0.0, llm_calls=0, cached_calls=0)
    if key in cache: cost["cached_calls"] += 1
    else:
        m = Meter()
        msg = chat([{"role": "system", "content": SHORT_SYS}, {"role": "user", "content": user}], m, max_tokens=8000, reasoning_effort="low", model=model)
        cache[key] = (msg.content or "").strip(); u = m.as_dict()
        for kk in ("prompt_tokens", "completion_tokens", "cost_usd"): cost[kk] += u[kk]
        cost["llm_calls"] += 1
        _atomic_save(cache_path, {key: cache[key]})
    return cache[key], cost

def chat_shorthand(chat_obj, art, **kw):
    block = "[Your own shorthand notes on the conversation so far]\n" + art["shorthand"]
    return _probe(_preload(Session(SYS), _msgs(block, "Understood, those are my notes on our earlier conversation.")))

# ---------------------------------------------------------------- single-shot RAG over raw exchanges (control)
def chat_rag(chat_obj, art=None, k=6, **kw):
    tdocs = [Doc(f"t{t['i']}", "turn", f"exchange {t['i']}", f"User: {t['user']}\nAssistant: {t['assistant']}") for t in chat_obj.turns]
    idx = Index(tdocs, contextual=False)
    def retriever(q, m):
        hits = idx.hybrid(q, k, m)
        return "[Exchanges retrieved for this question; the conversation had %d exchanges and facts may have been revised in ones not shown]\n" % len(chat_obj.turns) + idx.render(hits), [i for i, _ in hits]
    return _probe(Session(SYS, retriever=retriever))

# ---------------------------------------------------------------- zero-LLM deterministic reductions of the transcript itself
def _turns_block(turns, fmt):
    return "\n".join(fmt(t) for t in turns)

def user_only_text(chat_obj):
    return _turns_block(chat_obj.turns, lambda t: f"[{t['i']}] {t['user']}")

def chat_useronly(chat_obj, art=None, **kw):
    block = "[The user's messages from the conversation so far, numbered by exchange; the assistant's replies are omitted]\n" + user_only_text(chat_obj)
    return _probe(_preload(Session(SYS), _msgs(block)))

def _first_sentence(s, n=1):
    parts = re.split(r"(?<=[.!?])\s+", s.strip())
    return " ".join(parts[:n])

def asst_clip_text(chat_obj):
    return _turns_block(chat_obj.turns, lambda t: f"[{t['i']}] User: {t['user']}\n    Assistant: {_first_sentence(t['assistant'])}")

def chat_asstclip(chat_obj, art=None, **kw):
    block = "[Conversation so far; each assistant reply is clipped to its first sentence]\n" + asst_clip_text(chat_obj)
    return _probe(_preload(Session(SYS), _msgs(block)))

FACTY = re.compile(r"\d|\$|%|[A-Za-z]+-[a-z0-9]+|\b(?<![.!?]\s)[A-Z][a-z]{2,}\b")
def regex_filter_text(chat_obj):
    """Keep only user sentences that look like they carry a fact: a number, money, percent, a hyphenated identifier or a capitalised name mid-sentence. Drop questions."""
    out = []
    for t in chat_obj.turns:
        u = t["user"]
        if "\n" in u.strip():                      # multi-line = pasted output, keep whole
            out.append(f"[{t['i']}] {u}"); continue
        keep = [s for s in re.split(r"(?<=[.!?])\s+", u) if s and FACTY.search(s[1:]) and (not s.rstrip().endswith("?") or re.search(r"\d|\$", s))]
        if keep: out.append(f"[{t['i']}] " + " ".join(keep))
    return "\n".join(out)

def chat_regex(chat_obj, art=None, **kw):
    block = "[Fact-bearing user sentences kept by a regex filter, numbered by exchange; everything else was dropped]\n" + regex_filter_text(chat_obj)
    return _probe(_preload(Session(SYS), _msgs(block)))

STOP = set("the a an is are was were be been being we our us i you your it its this that these those to for of in on at by with and or as from will would should could can just also please let me".split())
def prune_text(chat_obj):
    """LLMLingua-lite: user turns only, stop-words and filler punctuation removed (negations kept)."""
    def prune(s):
        if "\n" in s.strip(): return s
        toks = re.findall(r"\S+", s)
        return " ".join(w for w in toks if re.sub(r"[^a-z']", "", w.lower()) not in STOP)
    return _turns_block(chat_obj.turns, lambda t: f"[{t['i']}] {prune(t['user'])}")

def chat_prune(chat_obj, art=None, **kw):
    block = "[The user's messages so far with function words removed, numbered by exchange]\n" + prune_text(chat_obj)
    return _probe(_preload(Session(SYS), _msgs(block)))

# ---------------------------------------------------------------- hand-off workers on the same ideas
def _worker(block):
    from run_chat import WSYS
    s = Session(WSYS); s.messages += _msgs(block); return s
def worker_terse(chat_obj, art, task):
    return _worker("[Fact ledger from the conversation: one line per entity, attr=current value, a changed value carries [turn it was set; was: earlier values in order]]\n" + terse_text(art["ledger"], art["history"])).ask(task), None
def worker_useronly(chat_obj, art, task):
    return _worker("[The user's messages from the conversation, numbered by exchange]\n" + user_only_text(chat_obj)).ask(task), None
def worker_regex(chat_obj, art, task):
    return _worker("[Fact-bearing user sentences kept by a regex filter, numbered by exchange]\n" + regex_filter_text(chat_obj)).ask(task), None
def worker_shorthand(chat_obj, art, task):
    return _worker("[Shorthand notes on the conversation written by the orchestrator]\n" + art["shorthand"]).ask(task), None
def worker_slice(chat_obj, art, task, k=30):
    s = chat_slice(chat_obj, art, k=k); from run_chat import WSYS; s.messages[0]["content"] = WSYS; return s.ask(task), None

WILD = {"chat_ledger0": chat_ledger0, "chat_terse": chat_terse, "chat_slice": chat_slice, "chat_tools0": chat_tools0, "chat_shorthand": chat_shorthand, "chat_rag": chat_rag,
        "chat_useronly": chat_useronly, "chat_asstclip": chat_asstclip, "chat_regex": chat_regex, "chat_prune": chat_prune}
WILD_NEEDS = {"chat_ledger0": "ledger", "chat_terse": "ledger", "chat_slice": "ledger", "chat_tools0": "ledger", "chat_shorthand": "shorthand"}
WILD_WORKERS = {"worker_terse": worker_terse, "worker_useronly": worker_useronly, "worker_regex": worker_regex, "worker_shorthand": worker_shorthand, "worker_slice": worker_slice}
WILD_WNEEDS = {"worker_terse": "ledger", "worker_shorthand": "shorthand", "worker_slice": "ledger"}
