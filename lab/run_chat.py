"""Run the no-documents benchmark: chat conditions x probe questions, and agent-to-agent hand-off tasks.
Results -> results/chat_<tag>[_model].jsonl (questions) / results/handoff_<tag>[_model].jsonl (tasks); folding costs -> results/chat_index_costs.jsonl."""
from __future__ import annotations
import argparse, json, os, re, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))
import llm, tiktoken
from corpus_chat import build_chats
from chat_facts import extract_chat, summarize_chat, check_ledger, ledger_text, audit_ledger
from conditions_chat import CHAT_CONDITIONS, NEEDS, transcript, chat_vc, _state_block, SYS
from conditions import Session, ANSWER_RULE
from grade import rule_grade, grade, extract_answer, _norm, _has
import conditions_wild as W
CHAT_CONDITIONS = {**CHAT_CONDITIONS, **W.WILD}; NEEDS = {**NEEDS, **W.WILD_NEEDS}

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ENC = tiktoken.get_encoding("o200k_base")

def _present(text, val):
    t = _norm(text)
    if isinstance(val, bool): return False
    if isinstance(val, str) and " then " in val:   # ordered list ("A then B then C"): the parts must appear in this order
        pos, at = [], 0
        for part in val.split(" then "):
            m = re.search(rf"(?<![a-z0-9]){re.escape(_norm(part))}(?![a-z0-9])", t[at:])
            if not m: return False
            at += m.end()
        return True
    if isinstance(val, (int, float)):
        s = str(val) if isinstance(val, int) or val == int(val) else f"{val}"
        return re.search(rf"(?<![0-9]){re.escape(_norm(s))}(?![0-9])", t) is not None
    return _has(t, _norm(val))

def grade_chat(ans, q):
    """Rule first; a superseded/other-entity value in the answer sends it to the judge."""
    r = rule_grade(ans, q)
    if r is True and q.distractors and any(_present(extract_answer(ans), d) for d in q.distractors): r = None
    if r is not None: return r, "rule"
    return grade(ans, q)

def grade_task(text, task):
    found = [lab for lab, val, al in task.required if any(_present(text, x) for x in [val] + list(al))]
    stale = [lab for lab, val in task.stale if _present(text, val)]
    return dict(required=len(task.required), found=len(found), missing=[lab for lab, _, _ in task.required if lab not in found], stale_present=stale, score=len(found) / max(1, len(task.required)))

def prepare(chat_obj, need_summary, need_ledger, model=None, batch=1, audit=False, salt="", need_shorthand=False):
    art = {}; costs = {}
    if need_shorthand:
        sh, c = W.shorthand_chat(chat_obj, model=model, salt=salt); art["shorthand"] = sh; costs["shorthand"] = {**c, "shorthand_tokens": len(ENC.encode(sh))}
    if need_ledger:
        cards, ledger, history, c = extract_chat(chat_obj, model=model, batch=batch, salt=salt)
        if audit:   # second pass over the raw exchanges with the finished ledger (same model as the folder)
            before = {k: v for k, v in check_ledger(chat_obj, ledger).items() if k != "errors"}
            ledger, history, cards, ac, applied = audit_ledger(chat_obj, ledger, history, cards, model=model, salt=salt)
            c = {**c, "audit": {**ac, "applied": applied, "check_before": before}}
        art.update(cards=cards, ledger=ledger, history=history); costs["ledger"] = {**c, "batch": batch, "n_facts": len(cards), "check": {k: v for k, v in check_ledger(chat_obj, ledger).items() if k != "errors"}, "ledger_tokens": len(ENC.encode(ledger_text(ledger, history)))}
        costs["ledger"]["errors"] = check_ledger(chat_obj, ledger)["errors"]
    if need_summary:
        s, c = summarize_chat(chat_obj, model=model, salt=salt); art["summary"] = s; costs["summary"] = {**c, "summary_tokens": len(ENC.encode(s))}
    return art, costs

# ------------------------------------------------------------------ agent-to-agent hand-off workers
WSYS = ("You are a worker agent completing a delegated task. Use only the information you are given (and tools if you have them). Never invent values; "
        "if a value cannot be found write 'unknown' for it. Reply with the finished note.")
BRIEF_SYS = "A colleague agent will complete a task for us but cannot see this conversation. Write a brief for them containing every fact from the conversation they need for the task, with current values (note superseded values as superseded). Be complete and concrete."

def worker_full(chat_obj, art, task):
    s = Session(WSYS); s.messages += transcript(chat_obj); return s.ask(task), None
def worker_summary(chat_obj, art, task):
    s = Session(WSYS); s.messages += [{"role": "user", "content": "[Summary of the conversation]\n" + art["summary"]}, {"role": "assistant", "content": "Understood."}]; return s.ask(task), None
def worker_state(chat_obj, art, task):
    s = Session(WSYS); s.messages += [{"role": "user", "content": _state_block(art)}, {"role": "assistant", "content": "Understood."}]; return s.ask(task), None
def worker_brief(chat_obj, art, task):
    o = Session(BRIEF_SYS); o.messages += transcript(chat_obj)
    brief, om, _ = o.ask(f"The task they will do: {task}\nWrite the brief now.")
    s = Session(WSYS); s.messages += [{"role": "user", "content": "[Brief from the orchestrator]\n" + brief}, {"role": "assistant", "content": "Understood."}]
    return s.ask(task), dict(orchestrator=om.as_dict(), brief_tokens=len(ENC.encode(brief)))
def worker_vc(chat_obj, art, task):
    s = chat_vc(chat_obj, art, keep=0); s.messages[0]["content"] = s.messages[0]["content"].replace(SYS, WSYS + " ").replace("Only the last 0 exchanges are in your context.", "The conversation is not in your context.")
    return s.ask(task), None
WORKERS = {"worker_full": worker_full, "worker_summary": worker_summary, "worker_brief": worker_brief, "worker_state": worker_state, "worker_vc": worker_vc, **W.WILD_WORKERS}
WNEEDS = {"worker_summary": "summary", "worker_state": "ledger", "worker_vc": "ledger", **W.WILD_WNEEDS}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="chat_full,chat_window,chat_summary,chat_repl,chat_state,chat_vc")
    ap.add_argument("--chats", default="", help="comma list of chat ids (default all)")
    ap.add_argument("--qtypes", default=""); ap.add_argument("--qids", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--handoff", action="store_true", help="run hand-off tasks with worker conditions instead of probe questions")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--tag", default="main"); ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--model", default=None, help="agent model; folding (extraction/summary) and judge stay on the default model unless --fold_model")
    ap.add_argument("--fold_model", default=None, help="model used for online extraction and rolling summaries")
    ap.add_argument("--batch", type=int, default=1, help="exchanges per extraction call (online folding granularity)")
    ap.add_argument("--kw", default="{}"); ap.add_argument("--prepare_only", action="store_true")
    ap.add_argument("--style", default="clean", help="clean (generator text) or messy (LLM-paraphrased user turns, non-echoing assistant replies; same ground truth)")
    ap.add_argument("--audit", action="store_true", help="run the ledger audit pass after online folding (same model as the folder)")
    ap.add_argument("--fold_salt", default="", help="cache-busting salt: an independent repeat of the folding/summary")
    args = ap.parse_args()
    kw = json.loads(args.kw)
    fold_model = args.fold_model or llm.MODEL
    if args.model: llm.MODEL = args.model
    chats = build_chats()
    if args.chats: chats = [c for c in chats if c.id in args.chats.split(",")]
    if args.style == "messy":
        from chat_messy import messy_chats
        chats = messy_chats(chats, model=fold_model)   # paraphrases come from the folding model, never the agent model
    conds = args.conditions.split(",")
    needs = {(WNEEDS if args.handoff else NEEDS).get(c) for c in conds} | ({"ledger", "summary"} if args.prepare_only else set())
    arts = {}
    with ThreadPoolExecutor(min(4, len(chats))) as ex:
        for c, (art, costs) in zip(chats, ex.map(lambda c: prepare(c, "summary" in needs, "ledger" in needs, fold_model, args.batch, args.audit, args.fold_salt, "shorthand" in needs), chats)):
            arts[c.id] = art
            if costs:
                rec = dict(chat=c.id, batch=args.batch, audit=args.audit, fold_salt=args.fold_salt, style=args.style, n_turns=len(c.turns), chat_tokens=sum(len(ENC.encode(t["user"] + t["assistant"])) for t in c.turns), fold_model=fold_model, **costs)
                print("fold:", c.id, json.dumps({k: (v if k != "ledger" else {kk: (vv if kk != "audit" else {a: b for a, b in vv.items() if a != "applied"}) for kk, vv in v.items() if kk != "errors"}) for k, v in costs.items()}), flush=True)
                if costs.get("ledger", {}).get("audit", {}).get("applied"): print("  audit applied:", costs["ledger"]["audit"]["applied"][:12], flush=True)
                if costs.get("ledger", {}).get("errors"): print("  ledger errors:", costs["ledger"]["errors"][:12], flush=True)
                with open(os.path.join(RES, "chat_index_costs.jsonl"), "a") as f: f.write(json.dumps(rec) + "\n")
    if args.prepare_only: print(f"session cost so far: ${llm.global_cost():.3f}"); return
    suffix = f"_{args.tag}" + (f"_{args.model.split('/')[-1]}" if args.model else "")
    base = dict(tag=args.tag, rep=args.rep, kw=kw, model=llm.MODEL, fold_model=fold_model, batch=args.batch, style=args.style, audit=args.audit, fold_salt=args.fold_salt)
    if args.handoff:
        out = os.path.join(RES, f"handoff{suffix}.jsonl")
        jobs = [(c, ch, t) for c in conds for ch in chats for t in ch.tasks if not args.qids or t.id in args.qids.split(",")]
        def work(c, ch, t):
            t0 = time.time()
            try:
                (ans, m, trace), extra = WORKERS[c](ch, arts[ch.id], t.task); g = grade_task(ans, t); err = None
            except Exception:
                ans, m, trace, extra, err = "", llm.Meter(), [], None, traceback.format_exc()[-500:]; g = grade_task("", t)
            return dict(condition=c, **base, chat=ch.id, task=t.id, n_turns=len(ch.turns), answer=ans[:6000], **g, full_credit=g["found"] == g["required"], error=err, extra=extra, trace=trace, wall_s=time.time() - t0, **m.as_dict())
    else:
        out = os.path.join(RES, f"chat{suffix}.jsonl")
        sel = [(ch, q) for ch in chats for q in ch.questions if (not args.qtypes or q.qtype in args.qtypes.split(",")) and (not args.qids or q.id in args.qids.split(","))]
        if args.limit:
            per = {}; sel2 = []
            for ch, q in sel:
                per[(ch.id, q.qtype)] = per.get((ch.id, q.qtype), 0) + 1
                if per[(ch.id, q.qtype)] <= args.limit: sel2.append((ch, q))
            sel = sel2
        jobs = [(c, ch, q) for c in conds for ch, q in sel]
        sessions = {}
        def work(c, ch, q):
            t0 = time.time()
            try:
                s = sessions.get((c, ch.id)) or CHAT_CONDITIONS[c](ch, arts[ch.id], **kw); sessions[(c, ch.id)] = s
                ans, m, trace = s.ask(q.question); ok, how = grade_chat(ans, q); err = None
            except Exception:
                ans, m, trace, ok, how, err = "", llm.Meter(), [], False, "error", traceback.format_exc()[-500:]
            return dict(condition=c, **base, chat=ch.id, n_turns=len(ch.turns), qid=q.id, qtype=q.qtype, question=q.question, gold=q.answer, answer=extract_answer(ans)[:300], correct=bool(ok), grader=how,
                        error=err, trace=trace, wall_s=time.time() - t0, **m.as_dict())
    print(f"{len(jobs)} jobs -> {out}", flush=True)
    with ThreadPoolExecutor(args.workers) as ex, open(out, "a") as f:
        futs = {ex.submit(work, *j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            rec = fut.result(); f.write(json.dumps(rec) + "\n"); f.flush()
            if args.handoff: print(f"[{k}/{len(jobs)}] {rec['condition']:15s} {rec['task']:14s} {rec['found']}/{rec['required']} stale={rec['stale_present']} tok={rec['prompt_tokens']:6d} ${rec['cost_usd']:.4f} {('ERR ' + rec['error'][-100:]) if rec['error'] else ''}", flush=True)
            else: print(f"[{k}/{len(jobs)}] {rec['condition']:15s} {rec['qid']:18s} {'OK ' if rec['correct'] else 'BAD'} tok={rec['prompt_tokens']:6d} calls={rec['llm_calls']} ${rec['cost_usd']:.4f} | {str(rec['gold'])[:28]} vs {rec['answer'][:40]!r} {('ERR ' + rec['error'][-100:]) if rec['error'] else ''}", flush=True)
    print(f"session cost so far: ${llm.global_cost():.3f}")

if __name__ == "__main__":
    main()
