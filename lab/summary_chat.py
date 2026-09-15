"""Markdown summary tables for the no-documents benchmark (results/chat_*.jsonl, results/handoff_*.jsonl)."""
from __future__ import annotations
import glob, json, os, statistics as st, sys
from collections import defaultdict
RES = os.path.join(os.path.dirname(__file__), "..", "results")
QT = ["early", "updated", "original", "count", "sum", "negation", "list", "paste", "absent"]

def load(pattern):
    recs = {}
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        if "pilot" in f or "index_costs" in f: continue
        for line in open(f):
            r = json.loads(line); key = (r["condition"], r["tag"], r["model"], r.get("fold_model"), r.get("batch", 1), r.get("style", "clean"), r.get("audit", False), r.get("fold_salt", ""), r["rep"], r["chat"], r.get("qid") or r.get("task"))
            recs[key] = r   # last write wins (reruns)
    return list(recs.values())

def lab(r): return r["condition"] + (f"[{r['tag']}]" if r["tag"] != "main" else "") + (f"@{r['model'].split('/')[-1]}" if r["model"] != "openai/gpt-5.6-luna" else "") + (f" fold@{r['fold_model'].split('/')[-1]}" if r.get("fold_model") and r["fold_model"] != "openai/gpt-5.6-luna" else "") + (f" batch{r['batch']}" if r.get("batch", 1) != 1 and r["condition"] in ("chat_state", "chat_vc", "chat_vc_nostate", "worker_state", "worker_vc") else "") + (" +audit" if r.get("audit") else "") + (f" foldrep:{r['fold_salt']}" if r.get("fold_salt") else "")

def size(r): return "messy" if r["chat"].endswith("m") else "dense" if r["chat"].startswith("dense") else "long" if r["chat"].startswith("long") else "normal"

def regrade(recs):
    """Retroactive alias fixes: a stored 'wrong' flips to right only when the rule grader now accepts the stored answer and no distractor value is in it."""
    sys.path.insert(0, os.path.dirname(__file__))
    from corpus_chat import build_chats
    from grade import rule_grade
    from run_chat import _present
    qs = {q.id: q for c in build_chats() for q in c.questions}
    for r in recs:
        q = qs.get(r["qid"])
        if q and not r["correct"] and not r.get("error") and rule_grade(r["answer"], q) is True and not any(_present(r["answer"], d) for d in q.distractors): r["correct"] = True; r["grader"] = "rule(regrade)"
    return recs

def questions(pattern="chat_*.jsonl"):
    recs = regrade(load(pattern)); g = defaultdict(list)
    SIZES = {"normal": "3 x ~29k tokens, 114 exchanges, ~72 facts each", "long": "1 x ~72k tokens, 186 exchanges, ~74 facts", "dense": "1 x ~30k tokens, 154 exchanges, 113 facts (14 modules)",
             "messy": "3 x ~29k tokens, same facts as the normal chats, user turns paraphrased by an LLM into informal chat, assistant replies no longer echo the fact"}
    for r in recs: g[(size(r), lab(r))].append(r)
    for sz in SIZES:
        rows = [(k, v) for k, v in g.items() if k[0] == sz]
        if not rows: continue
        types = [t for t in QT if any(r["qtype"] == t for _, v in rows for r in v)]
        print(f"\n### chats: {sz} ({SIZES[sz]})\n")
        print("| condition | n | accuracy | " + " | ".join(types) + " | prompt tok/q | peak tok | calls | cost/q |")
        print("|" + "---|" * (7 + len(types)))
        for (_, cond), v in sorted(rows, key=lambda x: x[0][1]):
            n = len(v); acc = sum(r["correct"] for r in v) / n
            bt = " | ".join((f"{sum(r['correct'] for r in v if r['qtype']==t)/max(1,sum(1 for r in v if r['qtype']==t)):.0%}" if any(r["qtype"] == t for r in v) else "") for t in types)
            print(f"| {cond} | {n} | {acc:.0%} | {bt} | {st.mean(r['prompt_tokens'] for r in v):,.0f} | {st.mean(r['peak_prompt_tokens'] for r in v):,.0f} | {st.mean(r['llm_calls'] for r in v):.1f} | ${st.mean(r['cost_usd'] for r in v):.4f} |")

def handoff(pattern="handoff_*.jsonl"):
    recs = load(pattern); g = defaultdict(list)
    # regrade from the stored note so grader fixes apply to old runs
    sys.path.insert(0, os.path.dirname(__file__))
    from corpus_chat import build_chats
    from run_chat import grade_task
    tasks = {t.id: t for c in build_chats() for t in c.tasks}
    for r in recs:
        if r["task"] in tasks and not r.get("error") and len(r["answer"]) < 1500:   # older records stored a truncated note; keep their stored grade
            r.update(grade_task(r["answer"], tasks[r["task"]])); r["full_credit"] = r["found"] == r["required"]
        g[(size(r), lab(r))].append(r)
    if not g: return
    print("\n### agent-to-agent hand-off (2 tasks per chat; worker must produce a note with every required fact)\n")
    print("| chats | worker condition | tasks | facts found | tasks fully correct | stale values presented | tokens to worker | orchestrator tokens | total cost/task |\n|---|---|---|---|---|---|---|---|---|")
    for (sz, cond), v in sorted(g.items()):
        def orch(r): return (r.get("extra") or {}).get("orchestrator") or {}
        print(f"| {sz} | {cond} | {len(v)} | {sum(r['found'] for r in v)}/{sum(r['required'] for r in v)} ({sum(r['found'] for r in v)/sum(r['required'] for r in v):.0%}) | {sum(r['full_credit'] for r in v)}/{len(v)} | {sum(len(r['stale_present']) for r in v)} | {st.mean(r['prompt_tokens'] for r in v):,.0f} | {st.mean(orch(r).get('prompt_tokens', 0) for r in v):,.0f} | ${st.mean(r['cost_usd'] + orch(r).get('cost_usd', 0) for r in v):.4f} |")

def fold_merge():
    """Folding records keyed by (chat, fold model, batch, audit, salt); cost from the record that did the real work, checks from the latest."""
    f = os.path.join(RES, "chat_index_costs.jsonl"); last = {}
    if not os.path.exists(f): return last
    for line in open(f):
        r = json.loads(line); k = (r["chat"], r["fold_model"], r.get("batch", 1), bool(r.get("audit")), r.get("fold_salt", ""))
        merged = dict(last.get(k, {}), **{kk: vv for kk, vv in r.items() if kk not in ("ledger", "summary")})
        for part in ("ledger", "summary"):   # cost from the record that did the real extraction work (cached reruns report $0); accuracy check from the latest record (the checker was refined)
            old, new = last.get(k, {}).get(part), r.get(part)
            if new is None: cur = old
            elif old is None or new.get("llm_calls", 0) >= old.get("llm_calls", 0): cur = dict(new)
            else: cur = {**old, **{kk: new[kk] for kk in ("check", "errors", "n_facts", "ledger_tokens", "summary_tokens") if kk in new}}
            if part == "ledger" and old and new and old.get("audit") and new.get("audit"):   # audit sub-record: cost from the run that made the calls, checks from the latest
                oa, na = old["audit"], new["audit"]
                cur["audit"] = {**(na if na.get("llm_calls", 0) >= oa.get("llm_calls", 0) else oa), **{kk: na[kk] for kk in ("check_before", "applied") if kk in na}}
            if cur is not None: merged[part] = cur
        last[k] = merged
    return last

def folding():
    last = fold_merge()
    if not last: return
    print("\n### folding cost per chat (paid once, online, as the conversation happens)\n")
    print("| chat | fold model | variant | exchanges | chat tokens | ledger: facts | ledger tokens | ledger fields correct | extraction cost | audit: fields before -> after, patches, cost | summary tokens | summary cost |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (c, fm, b, au, salt), r in sorted(last.items()):
        L, S = r.get("ledger", {}), r.get("summary", {}); A = L.get("audit") or {}
        var = ", ".join(x for x in [f"batch{b}" if b != 1 else "", "audit" if au else "", f"repeat:{salt}" if salt else ""] if x) or "-"
        aud = f"{A['check_before']['ok']} -> {L.get('check',{}).get('ok','')}, {len(A.get('applied', []))}, ${A.get('cost_usd',0):.3f}" if A else ""
        print(f"| {c} | {fm.split('/')[-1]} | {var} | {r['n_turns']} | {r['chat_tokens']:,} | {L.get('n_facts','')} | {L.get('ledger_tokens','')} | {L.get('check',{}).get('ok','')}/{L.get('check',{}).get('fields','')} | ${L.get('cost_usd',0):.3f} | {aud} | {S.get('summary_tokens','')} | ${S.get('cost_usd',0):.3f} |")

if __name__ == "__main__":
    questions(); handoff(); folding()
