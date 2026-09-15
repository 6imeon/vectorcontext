"""Conditions for the no-documents setting: the context to reduce is the conversation itself. Each factory takes a Chat and the
folded artifacts (cards/ledger/summary) and returns a Session preloaded with history; `ask(question)` probes it."""
from __future__ import annotations
import copy, json
from corpus import Doc
from conditions import Session, ANSWER_RULE, PY_TOOL, SEARCH_TOOL, TSEARCH_TOOL
from index import Index
from sandbox import Sandbox
from chat_facts import ledger_text
from llm import Meter

SYS = ("You are the assistant in an ongoing project-planning conversation with the user. Answer the user's question from the conversation. "
       "If the conversation never established the fact, say 'not stated' instead of guessing. " + ANSWER_RULE)
KEEP = 10  # raw recent exchanges kept verbatim by the reduced conditions

def transcript(chat_obj, turns=None):
    out = []
    for t in (turns if turns is not None else chat_obj.turns):
        out += [{"role": "user", "content": t["user"]}, {"role": "assistant", "content": t["assistant"]}]
    return out

def _preload(session, msgs):
    session.messages = [session.messages[0]] + msgs; return session

def _probe(session):
    """Each question is an independent probe of the same history: clone the session per ask."""
    base = session.messages
    def ask(text):   # thread-safe: a private shallow copy of the session with its own message list (probes run in parallel on the same history)
        s2 = copy.copy(session); s2.messages = copy.deepcopy(base); s2.turn_no = 0
        return Session.ask(s2, text)
    session.ask = ask; return session

# 1. full history: every turn stays in context (what plain chatting does)
def chat_full(chat_obj, art=None, **kw):
    return _probe(_preload(Session(SYS), transcript(chat_obj)))

# 2. window: only the last KEEP exchanges (naive truncation)
def chat_window(chat_obj, art=None, keep=KEEP, **kw):
    return _probe(_preload(Session(SYS), transcript(chat_obj, chat_obj.turns[-keep:] if keep else [])))

# 3. rolling summary compaction + last KEEP exchanges (what most agent frameworks do)
def chat_summary(chat_obj, art, keep=KEEP, **kw):
    msgs = [{"role": "user", "content": f"[Summary of the conversation so far]\n{art['summary']}"}, {"role": "assistant", "content": "Understood, I have the summary of our earlier conversation."}] + transcript(chat_obj, chat_obj.turns[-keep:] if keep else [])
    return _probe(_preload(Session(SYS), msgs))

# 4. REPL over the transcript + last KEEP exchanges (the "REPL workspace" idea applied to history)
def chat_repl(chat_obj, art=None, keep=KEEP, output_limit=3000, max_turns=10, **kw):
    sb = Sandbox({"turns": [dict(i=t["i"], user=t["user"], assistant=t["assistant"]) for t in chat_obj.turns]}, output_limit=output_limit)
    system = (SYS + f" Only the last {keep} exchanges are in your context; the FULL transcript ({len(chat_obj.turns)} exchanges) is loaded in a Python REPL as `turns` "
              "(list of dicts i, user, assistant, in order). Use the `python` tool to search it (re is imported). Facts may have been revised later in the conversation: "
              "the latest mention wins, so check all mentions in order. Output is truncated, print only what you need.")
    return _probe(_preload(Session(system, [PY_TOOL], lambda n, a: sb.run(a.get("code", "")), max_turns=max_turns), transcript(chat_obj, chat_obj.turns[-keep:] if keep else [])))

# 5. folded state in the prompt (ledger with superseded values) + last KEEP exchanges, no tools
def _state_block(art):
    return ("[Fact ledger folded from the conversation so far: entity.attribute = current value [turn it was set], with superseded values in parentheses]\n"
            + ledger_text(art["ledger"], art["history"]))

def chat_state(chat_obj, art, keep=KEEP, **kw):
    msgs = [{"role": "user", "content": _state_block(art)}, {"role": "assistant", "content": "Understood, I have the ledger of our earlier conversation."}] + transcript(chat_obj, chat_obj.turns[-keep:] if keep else [])
    return _probe(_preload(Session(SYS), msgs))

# 6. vectorcontext for chat: ledger in prompt + tools (search over facts, search_text over raw turns, open_turn, python over ledger/facts/turns)
OPEN_TURN = {"type": "function", "function": {"name": "open_turn", "description": "Return the raw text of exchange number i (user message and assistant reply).", "parameters": {"type": "object", "properties": {"i": {"type": "integer"}}, "required": ["i"]}}}
CPY_TOOL = {"type": "function", "function": {"name": "python", "description": "Persistent Python REPL. Variables: `state` (dict entity -> {attribute: current value}), `facts` (list of {turn, entity, attribute, value} in order; earlier entries for the same entity.attribute are superseded values), `turns` (full raw transcript: list of {i, user, assistant}), `search(query, k)` -> facts, `search_text(query, k)` -> raw passages. Use it to count/sum/filter over `state` or to grep `turns`. stdout truncated.", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}

def chat_vc(chat_obj, art, keep=KEEP, output_limit=3000, max_turns=8, state_in_prompt=True, **kw):
    fdocs = [Doc(f"f{k}", "fact", f"turn {c['turn']}", f"{c['entity']}.{c['attribute']} = {json.dumps(c['value'])} (turn {c['turn']})") for k, c in enumerate(art["cards"])]
    tdocs = [Doc(f"t{t['i']}", "turn", f"exchange {t['i']}", f"User: {t['user']}\nAssistant: {t['assistant']}") for t in chat_obj.turns]
    fidx = Index(fdocs, contextual=False) if fdocs else None; tidx = Index(tdocs, contextual=False)
    _m = [Meter()]
    def search(q, k=8):
        k = max(1, min(int(k or 8), 40)); return [fidx.chunks[i]["text"] for i, _ in fidx.hybrid(q, k, _m[0])] if fidx else []
    def search_text(q, k=4):
        k = max(1, min(int(k or 4), 12)); return tidx.render(tidx.hybrid(q, k, _m[0]))
    def open_turn(i):
        t = next((t for t in chat_obj.turns if t["i"] == int(i)), None); return f"User: {t['user']}\nAssistant: {t['assistant']}" if t else "(no such exchange)"
    ns = {"state": {e: dict(d) for e, d in art["ledger"].items()}, "facts": art["cards"], "turns": [dict(i=t["i"], user=t["user"], assistant=t["assistant"]) for t in chat_obj.turns],
          "search": lambda q, k=8: search(q, k), "search_text": lambda q, k=4: search_text(q, k)}
    sb = Sandbox(ns, output_limit=output_limit)
    def dispatch(name, a):
        if name == "search": return "\n".join(search(a.get("query", ""), a.get("k"))) or "(no facts)"
        if name == "search_text": return search_text(a.get("query", ""), a.get("k"))
        if name == "open_turn": return open_turn(a.get("i", 0))
        if name == "python": return sb.run(a.get("code", ""))
        return f"unknown tool {name}"
    system = (SYS + f" Only the last {keep} exchanges are in your context. Earlier exchanges were folded into a fact ledger" + (" (given below)" if state_in_prompt else "") +
              "; the raw transcript is still reachable through tools. Strategy: current values -> the ledger (`state`); values before a change -> the superseded entries; counts/sums/lists -> `python` over `state`; "
              "exact wording or numbers from pasted output -> `search_text`/`open_turn`; before answering 'not stated', check with `search_text` that the transcript really never mentions it. "
              "Stop once you have the evidence.")
    tools = [SEARCH_TOOL | {"function": {**SEARCH_TOOL["function"], "description": "Hybrid search over the fact ledger entries (entity.attribute = value, with turn numbers)."}},
             TSEARCH_TOOL | {"function": {**TSEARCH_TOOL["function"], "description": "Hybrid semantic+keyword search over the raw conversation exchanges. Use it for exact wording, pasted output, or to verify that something was never mentioned."}},
             OPEN_TURN, CPY_TOOL]
    msgs = ([{"role": "user", "content": _state_block(art)}, {"role": "assistant", "content": "Understood, I have the ledger of our earlier conversation."}] if state_in_prompt else []) + transcript(chat_obj, chat_obj.turns[-keep:] if keep else [])
    s = _probe(_preload(Session(system, tools, dispatch, max_turns=max_turns), msgs))
    _ask = s.ask
    def ask(text):
        _m[0] = Meter(); ans, m, tr = _ask(text); m.embed_tokens += _m[0].embed_tokens; m.embed_cost += _m[0].embed_cost; return ans, m, tr
    s.ask = ask; return s

CHAT_CONDITIONS = {"chat_full": chat_full, "chat_window": chat_window, "chat_summary": chat_summary, "chat_repl": chat_repl, "chat_state": chat_state, "chat_vc": chat_vc,
                   "chat_vc_nostate": lambda c, art, **kw: chat_vc(c, art, state_in_prompt=False, **kw)}
NEEDS = {"chat_summary": "summary", "chat_state": "ledger", "chat_vc": "ledger", "chat_vc_nostate": "ledger"}
