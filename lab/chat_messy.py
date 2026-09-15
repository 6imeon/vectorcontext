"""Messier variant of the chat benchmark, same ground truth: every fact-bearing USER turn is paraphrased by an LLM into an informal
team-chat message (facts buried mid-message, asides, lowercase, occasional typos; every name/date/number kept verbatim and checked),
and the assistant's reply no longer echoes the fact ("noted." instead of "ingest is now due Oct 7"), so the folder has to resolve
"pull it in by three days" / "same budget as X" / "swap the owners" itself. Questions, tasks and generator state are unchanged."""
from __future__ import annotations
import hashlib, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from corpus_chat import Chat, MODULES, DATASTORES, PROJECTS, REGIONS
from corpus import FIRST
from chat_facts import CACHE
from index import _atomic_save
from llm import chat, Meter
import llm

PARA_SYS = ("Rewrite the message below as a realistic, informal message from a project lead to a colleague in a team chat: relaxed phrasing, lowercase is fine, "
            "an occasional typo or abbreviation, a short unrelated aside (a meeting, lunch, the weather) before or after, and do not lead with the fact: bury it mid-message. "
            "HARD RULES: every person name, module name, date, amount, number, priority and technology in the original must appear in the rewrite EXACTLY as written "
            "(same spelling, capitalization and format); the meaning must not change; add no new facts or numbers. Output only the rewritten message.")
NOECHO = ["sounds good, noted.", "ok, I'll keep track of that.", "got it.", "makes sense. anything else?", "understood.", "alright, logged that on my side.", "sure thing.",
          "okay. want me to update anything else?", "noted, thanks.", "will do."]
VOCAB = set(FIRST) | set(MODULES) | set(DATASTORES) | set(PROJECTS) | set(REGIONS)
MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

def required_tokens(text):
    """Values the rewrite must keep verbatim: names, module names, datastores, dates, money, priorities, times, repo names."""
    req = set(re.findall(rf"\b(?:{MONTHS}) \d{{1,2}}\b", text)) | set(re.findall(r"\$\d+k", text)) | set(re.findall(r"\bP[0-2]\b", text)) | set(re.findall(r"\b\d{1,2}:\d{2}\b", text)) | set(re.findall(r"\b[a-z]+-launch\b", text))
    req |= {w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]*", text) if w in VOCAB}
    return req

def paraphrase(text, model=None, cache_path=CACHE, tries=2):
    """LLM rewrite with a verbatim-value check; falls back to the original text if the check fails twice. Returns (text, status, cost)."""
    model = model or llm.MODEL
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    cost = dict(cost_usd=0.0, llm_calls=0); req = required_tokens(text)
    for n in range(tries):
        key = f"para1|{n}|" + hashlib.sha1((model + PARA_SYS + text).encode()).hexdigest()
        if key not in cache:
            m = Meter()
            msg = chat([{"role": "system", "content": PARA_SYS}, {"role": "user", "content": text}], m, max_tokens=400, reasoning_effort="low", model=model)
            cache[key] = (msg.content or "").strip(); cost["cost_usd"] += m.as_dict()["cost_usd"]; cost["llm_calls"] += 1
            _atomic_save(cache_path, {key: cache[key]})
        out = cache[key]
        if out and all(v in out for v in req): return out, "rewritten", cost
    return text, "kept", cost

def messy_chat(c: Chat, model=None):
    rng = random.Random(c.meta["seed"] * 101 + 3); turns = []; stats = dict(rewritten=0, kept=0, cost_usd=0.0)
    for t in c.turns:
        u, a = t["user"], t["assistant"]
        if t["kind"] != "filler":
            if "```" in u:   # pasted output: only the sentence before the block is rewritten, the paste stays verbatim
                head, rest = u.split("```", 1); new, status, cost = paraphrase(head.strip(), model); u = new + "\n```" + rest
            else:
                u, status, cost = paraphrase(u, model)
                a = rng.choice(NOECHO)   # the assistant no longer restates the fact
            stats[status] += 1; stats["cost_usd"] += cost["cost_usd"]
        turns.append(dict(t, user=u, assistant=a))
    return Chat(c.id + "m", c.project, turns, c.questions, c.tasks, c.state, c.history, dict(c.meta, style="messy", **stats)), stats

def messy_chats(chats, model=None):
    out = []
    for c in chats:
        mc, stats = messy_chat(c, model); out.append(mc)
        print(f"messy {mc.id}: {stats}", flush=True)
    return out

if __name__ == "__main__":
    import tiktoken
    from corpus_chat import build_chats
    enc = tiktoken.get_encoding("o200k_base")
    ids = sys.argv[1].split(",") if len(sys.argv) > 1 else ["chat1", "chat2", "chat3"]
    for c in messy_chats([c for c in build_chats() if c.id in ids]):
        print(c.id, "tokens", sum(len(enc.encode(t["user"] + t["assistant"])) for t in c.turns))
        for t in c.turns:
            if t["kind"] != "filler": print(f"  ({t['i']}) {t['user'][:160]!r} -> {t['assistant'][:40]!r}")
