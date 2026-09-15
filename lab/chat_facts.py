"""Online context folding for conversations (no documents): every exchange is turned into ledger facts as it arrives,
and a rolling LLM summary (the usual 'compaction') is built the same way for comparison. Both are cached on disk."""
from __future__ import annotations
import hashlib, json, os, re
from collections import defaultdict
from llm import chat, Meter
import llm
from index import _atomic_save

CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "chat_cache.json")

EXTRACT_SYS = ("You maintain a fact ledger for an ongoing conversation between a user and an assistant about the user's project. You get the LEDGER so far (entity.attribute = value), "
               "the previous exchanges for reference, and ONE NEW exchange. Output the durable PROJECT facts the NEW exchange establishes or changes, as a JSON array of {\"entity\", \"attribute\", \"value\"}. "
               "A project fact is something the USER decides, assigns, states or pastes about their own plan: owners, deadlines, budgets, choices, dates, names, and numbers inside pasted tool output "
               "(logs, tables, configs; entity = what the paste is, e.g. load_test, ci_run, config; one item per number). "
               "NOT facts: the assistant's explanations of general concepts (even when they contain numbers), the user's questions, requests for explanations, and chit-chat -> output []. "
               "Rules: reuse the ledger's entity and attribute names when the same thing is meant (prefer owner, deadline, budget_k, datastore, priority for modules); "
               "values normalized (dates as 'Mon D' like 'Oct 7'; money in thousands as a number under an attribute ending in _k; other numbers as numbers; people by name; lists as JSON arrays); "
               "resolve references ('that', 'it', 'same as X', 'push it back a week', 'swap the owners') against the ledger and previous exchanges and output the resolved ABSOLUTE new values, one item per changed fact; "
               "if something is dropped or withdrawn output attribute in_scope = false, and when something new is added to the plan output its in_scope = true; "
               "whenever the ledger holds a membership list (e.g. project.in_scope_modules) and the exchange adds or removes a member, ALSO output the updated full list so no ledger entry is left stale. "
               "When several exchanges are given, add \"turn\": <turn number> to each item. JSON array only, no prose.")

SUMMARY_SYS = ("You maintain a compact running summary of a conversation for an assistant that will no longer see the raw turns. Update the summary with the new turns: "
               "keep every concrete fact, decision, name, number, date and change (when a value replaces an earlier one, keep both and say which is current), keep numbers from pasted tool output, "
               "drop tangents and explanations. Output the updated summary as plain text, at most about 700 words.")

def _pair(t): return f"User: {t['user']}\nAssistant: {t['assistant']}"

def _json_list(txt):
    txt = re.sub(r"^```(?:json)?|```$", "", (txt or "").strip(), flags=re.M).strip()
    try: v = json.loads(txt)
    except Exception:
        m = re.search(r"\[.*\]", txt, re.S)
        try: v = json.loads(m.group(0)) if m else []
        except Exception: v = []
    return [x for x in v if isinstance(x, dict) and x.get("entity") and x.get("attribute")] if isinstance(v, list) else []

def _nn(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())

def apply_items(ledger, history, cards, items, default_turn, lo=None, hi=None):
    """Write extracted items into the ledger, then enforce the membership invariant: when X.in_scope changes, every list-valued entry
    that names modules (attribute contains 'module' or 'scope') gains or loses X. Found necessary because the folder sometimes re-emitted
    the updated list under a drifted name ('in_scope_modules.value') or not at all, leaving a stale list for agents that read it by name."""
    changed = []
    for f in items:
        e, a, v = str(f["entity"]), re.sub(r"\.value$", "", str(f["attribute"])), f.get("value")
        ti = f["turn"] if isinstance(f.get("turn"), int) and (lo is None or lo <= f["turn"] <= hi) else default_turn
        ledger.setdefault(e, {})[a] = v; history.setdefault((e, a), []).append((ti, v)); cards.append(dict(turn=ti, entity=e, attribute=a, value=v))
        if a == "in_scope": changed.append((e, v, ti))
    # every membership list must agree with the in_scope flags: dropped entities leave every list; an entity whose in_scope=true flag is newer than
    # the list (or set in this batch) joins it, but only if it looks like a member (shares attributes with the current members) so a spurious
    # flag on e.g. 'standup' cannot pollute the module list
    just = {e for e, _, _ in changed}
    for le, d in list(ledger.items()):
        for la, lv in list(d.items()):
            if not _is_membership(la, lv): continue
            lt = history.get((le, la), [(default_turn, lv)])[-1][0]
            ents = [ledger[k] for k in ledger if any(_nn(k) == _nn(x) for x in lv)]
            common = {a2 for ent in ents for a2 in ent if a2 != "in_scope" and sum(a2 in x for x in ents) * 2 >= len(ents)}
            new = list(lv)
            for e, ed in ledger.items():
                if "in_scope" not in ed: continue
                v = ed["in_scope"]; ft = history.get((e, "in_scope"), [(default_turn, v)])[-1][0]
                inside = not (v is False or str(v).strip().lower() in ("false", "no", "dropped", "out", "removed", "out of scope", "next quarter"))
                has = any(_nn(x) == _nn(e) for x in new)
                if not inside and has: new = [x for x in new if _nn(x) != _nn(e)]
                elif inside and not has and (e in just or ft > lt) and (not common or common & set(ed)): new.append(e)
            if new != lv:
                ledger[le][la] = new; history.setdefault((le, la), []).append((default_turn, new)); cards.append(dict(turn=default_turn, entity=le, attribute=la, value=new))

def _is_membership(a, v):
    """A list of names that is the project's membership list (in_scope_modules, modules_in_scope, modules); not e.g. ci_run.failed_modules."""
    return isinstance(v, list) and bool(v) and all(isinstance(x, str) for x in v) and ("scope" in a.lower() or a.lower() in ("modules", "module_list", "launch_modules"))

def ledger_text(ledger, history=None):
    """Compact state rendering used in prompts: 'entity.attr = value' plus superseded values when history is given."""
    lines = []
    for e, d in ledger.items():
        for a, v in d.items():
            h = (history or {}).get((e, a), [])
            old = f"   (was: {', '.join(f'{json.dumps(pv)} @t{pt}' for pt, pv in h[:-1])})" if len(h) > 1 else ""
            lines.append(f"{e}.{a} = {json.dumps(v)}  [t{h[-1][0] if h else '?'}]{old}")
    return "\n".join(lines)

def extract_chat(chat_obj, model=None, ctx_pairs=2, cache_path=CACHE, upto=None, batch=1, salt=""):
    """Sequential online extraction: one call per exchange (or per `batch` exchanges), conditioned on the ledger so far (so coreference resolves).
    Returns (cards, ledger, history, cost). cards: [{turn, entity, attribute, value}]. `salt` busts the cache (independent repeat of the folding)."""
    model = model or llm.MODEL
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    ledger = defaultdict(dict); history = defaultdict(list); cards = []
    cost = dict(prompt_tokens=0, completion_tokens=0, cost_usd=0.0, llm_calls=0, cached_calls=0)
    turns = chat_obj.turns[:upto] if upto else chat_obj.turns
    for k in range(0, len(turns), batch):
        group = turns[k:k + batch]; t = group[-1]
        prev = "\n\n".join(_pair(p) for p in turns[max(0, k - ctx_pairs):k]) or "(none)"
        new = "\n\n".join(f"(turn {g['i']})\n{_pair(g)}" for g in group)
        user = f"LEDGER:\n{ledger_text(ledger) or '(empty)'}\n\nPREVIOUS EXCHANGES:\n{prev}\n\nNEW EXCHANGE{'S' if batch > 1 else ''} (turn{'s' if batch > 1 else ''} {group[0]['i']}{'-' + str(t['i']) if batch > 1 else ''}):\n{new}"
        key = f"chatfact2|b{batch}|{salt}" + hashlib.sha1((model + EXTRACT_SYS + user).encode()).hexdigest()
        if key in cache: cost["cached_calls"] += 1
        else:
            m = Meter()
            msg = chat([{"role": "system", "content": EXTRACT_SYS}, {"role": "user", "content": user}], m, max_tokens=800, reasoning_effort="low", model=model)
            cache[key] = _json_list(msg.content); u = m.as_dict()
            for kk in ("prompt_tokens", "completion_tokens", "cost_usd"): cost[kk] += u[kk]
            cost["llm_calls"] += 1
            _atomic_save(cache_path, {key: cache[key]})
        apply_items(ledger, history, cards, cache[key], t["i"], group[0]["i"], t["i"])
    return cards, dict(ledger), dict(history), cost

def summarize_chat(chat_obj, every=20, keep=10, model=None, cache_path=CACHE, salt=""):
    """Rolling compaction: turns older than the last `keep` are folded into a running summary, `every` turns at a time."""
    model = model or llm.MODEL
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    cost = dict(prompt_tokens=0, completion_tokens=0, cost_usd=0.0, llm_calls=0, cached_calls=0)
    summary = ""; turns = chat_obj.turns[:-keep] if keep else chat_obj.turns
    for k in range(0, len(turns), every):
        block = "\n\n".join(_pair(t) for t in turns[k:k + every])
        user = f"CURRENT SUMMARY:\n{summary or '(empty)'}\n\nNEW TURNS:\n{block}"
        key = f"chatsum1|{salt}" + hashlib.sha1((model + SUMMARY_SYS + user).encode()).hexdigest()
        if key in cache: cost["cached_calls"] += 1
        else:
            m = Meter()
            msg = chat([{"role": "system", "content": SUMMARY_SYS}, {"role": "user", "content": user}], m, max_tokens=2000, reasoning_effort="low", model=model)
            cache[key] = (msg.content or "").strip(); u = m.as_dict()
            for kk in ("prompt_tokens", "completion_tokens", "cost_usd"): cost[kk] += u[kk]
            cost["llm_calls"] += 1
            _atomic_save(cache_path, {key: cache[key]})
        summary = cache[key]
    return summary, cost

AUDIT_SYS = ("You audit a fact ledger that was folded online from a conversation between a user and an assistant about the user's project. You get the LEDGER "
             "(entity.attribute = value [turn it was set], superseded values in parentheses) built from the WHOLE conversation, and one CHUNK of the raw exchanges. "
             "Report project facts in the chunk that the ledger missed or got wrong. A project fact is something the USER decides, assigns, states or pastes about their own plan: "
             "owners, deadlines, budgets, choices, dates, names, and every number inside pasted tool output (logs, tables, configs; entity = what the paste is, e.g. load_test, ci_run, config; one item per number). "
             "NOT facts: the assistant's explanations of general concepts, the user's questions and chit-chat. "
             "The ledger may legitimately differ from this chunk because a LATER exchange changed the value: an entry set at a turn after this chunk is NOT an error. Report only "
             "(a) facts with no ledger entry at all for that entity.attribute, and (b) entries set inside this chunk whose value does not match what the user said (resolve 'that', 'same as X', 'three days earlier' to absolute values). "
             "Reuse the ledger's entity and attribute names when the same thing is meant (prefer owner, deadline, budget_k, datastore, priority for modules); values normalized "
             "(dates as 'Mon D'; money in thousands as a number under an attribute ending in _k; other numbers as numbers; people by name; lists as JSON arrays). "
             "Output a JSON array of {\"entity\", \"attribute\", \"value\", \"turn\"}; [] if the ledger is complete for this chunk. JSON only, no prose.")

def audit_ledger(chat_obj, ledger, history, cards, model=None, chunk=20, cache_path=CACHE, salt=""):
    """Second pass for weak folders: re-read the raw exchanges in chunks against the finished ledger and patch what the online pass missed.
    A reported item is applied only when its key is absent or when no later exchange already revised that key (revisions are never undone).
    Returns (ledger, history, cards, cost, applied)."""
    model = model or llm.MODEL
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    ledger = {e: dict(d) for e, d in ledger.items()}; history = {k: list(v) for k, v in history.items()}; cards = list(cards)
    cost = dict(prompt_tokens=0, completion_tokens=0, cost_usd=0.0, llm_calls=0, cached_calls=0); applied = []
    turns = chat_obj.turns
    for k in range(0, len(turns), chunk):
        group = turns[k:k + chunk]
        user = (f"LEDGER:\n{ledger_text(ledger, history) or '(empty)'}\n\nCHUNK (turns {group[0]['i']}-{group[-1]['i']}):\n" + "\n\n".join(f"(turn {g['i']})\n{_pair(g)}" for g in group))
        key = f"chataudit1|{salt}" + hashlib.sha1((model + AUDIT_SYS + user).encode()).hexdigest()
        if key in cache: cost["cached_calls"] += 1
        else:
            m = Meter()
            msg = chat([{"role": "system", "content": AUDIT_SYS}, {"role": "user", "content": user}], m, max_tokens=1200, reasoning_effort="low", model=model)
            cache[key] = _json_list(msg.content); u = m.as_dict()
            for kk in ("prompt_tokens", "completion_tokens", "cost_usd"): cost[kk] += u[kk]
            cost["llm_calls"] += 1
            _atomic_save(cache_path, {key: cache[key]})
        keep = []
        for f in cache[key]:
            e, a, v = str(f["entity"]), re.sub(r"\.value$", "", str(f["attribute"])), f.get("value")
            ti = f["turn"] if isinstance(f.get("turn"), int) else group[-1]["i"]
            cur = history.get((e, a))
            if cur and (cur[-1][0] > ti or cur[-1][1] == v): continue
            keep.append(dict(entity=e, attribute=a, value=v, turn=ti)); applied.append(keep[-1])
        apply_items(ledger, history, cards, keep, group[-1]["i"])
    return ledger, history, cards, cost, applied

def check_ledger(chat_obj, ledger):
    """Compare the extracted ledger with generator state: every gold (entity, attr) must be present with the same value.
    Entity/attribute names may drift ('project' vs the project name, 'oncall.order' vs 'launch_week.on_call_order'), so names are
    matched loosely by synonyms/containment; only a missing or different VALUE counts as an error."""
    def n(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
    def nv(v):
        if isinstance(v, bool): return str(v).lower()
        if isinstance(v, (int, float)): return str(float(v))
        if isinstance(v, str) and v.strip().startswith("["):   # a list serialized as a JSON string
            try: v = json.loads(v)
            except Exception: pass
        if isinstance(v, list): return ",".join(n(x) for x in v)
        s = re.sub(r"\s*(utc|ms|percent|pct|usd|%|k)$", "", str(v).strip().lower())
        try: return str(float(s.replace(",", "").replace("$", "")))   # '0.4%' == 0.4, '$45k' == 45
        except ValueError: pass
        s = re.sub(r"(utc|ms|percent|pct|usd)$", "", n(v))
        try: return str(float(s))
        except ValueError: return s
    ESYN = {"project": ["project", "launch", n(chat_obj.project), "repository", "repo", "deployment", "deploy", "plan"], "oncall": ["oncall", "onc", "rotation", "launchweek", "launch", "project", n(chat_obj.project)], "standup": ["standup", "project", "launch", n(chat_obj.project)], "load_test": ["loadtest", "load"], "ci_run": ["ci", "test"], "config": ["config", "service"]}
    ASYN = {"budget_k": ["budget"], "time_utc": ["standuptime", "time"], "order": ["order", "rotation", "oncall", "sequence"], "p95_ms": ["p95"], "error_rate_pct": ["error"], "db_pool_size": ["pool"], "failed": ["failed"], "passed": ["passed"],
            "launch_date": ["launch", "target", "deadline", "date"], "in_scope": ["scope", "dropped", "status"], "datastore": ["datastore", "backend", "store", "database", "db", "storage"], "owner": ["owner", "lead", "assign"],
            "deadline": ["deadline", "due", "target"], "region": ["region"], "repo": ["repo"], "priority": ["priority", "prio"], "rps": ["rps"], "max_workers": ["worker"]}
    flat = defaultdict(dict)
    for e, d in ledger.items():
        for a, v in d.items(): flat[n(e)][n(a)] = v
    errors, ok, total = [], 0, 0
    for e, d in chat_obj.state.items():
        for a, v in d.items():
            if a == "in_scope" and v is True: continue   # implicit
            total += 1
            ecands = ESYN.get(e, [n(e)]); acands = ASYN.get(a, [n(a)])
            ents = [flat[x] for x in flat if any(c in x or x in c for c in ecands)]
            got = next((ent[k] for ent in ents for k in ent if any(c in k for c in acands)), None)
            if got is None:   # the attribute became its own entity ('repository.name = delta-launch')
                got = next((ent[k] for x, ent in flat.items() if any(c in x for c in acands) for k in ent if k in ("name", "value")), None)
            good = got is not None and (nv(got) == nv(v) or (a == "in_scope" and nv(got) in ("false", "dropped", "out", "no", "outofscope", "removed", "nextquarter")))
            if good: ok += 1
            else: errors.append(dict(entity=e, attr=a, gold=v, got=got))
    # membership lists: any list-valued entry naming modules must equal the gold in-scope set (a stale list misleads agents that read it by name)
    ins = {n(m) for m, d in chat_obj.state.items() if d.get("in_scope") is True}
    for e, d in ledger.items():
        for a, v in d.items():
            if _is_membership(a, v) and {n(x) for x in v} & ins:
                total += 1
                if {n(x) for x in v} == ins: ok += 1
                else: errors.append(dict(entity=e, attr=a, gold=sorted(ins), got=v))
    return dict(fields=total, ok=ok, errors=errors)
