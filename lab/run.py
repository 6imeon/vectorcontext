"""Run a matrix of conditions x questions (and multi-turn conversations), append results to results/<run>.jsonl."""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(__file__))
import corpus, corpus_hard, llm
from conditions import CONDITIONS
from index import Index, FactIndex
from facts import build_facts, resolve_updates
from grade import grade, extract_answer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="baseline,repl,vector_rag,vectorcontext_v1")
    ap.add_argument("--corpus", default="base", choices=["base", "hard"])
    ap.add_argument("--scale", type=int, default=3, help="filler_k: 1≈23k, 3≈45k, 6≈78k tokens")
    ap.add_argument("--qtypes", default="", help="comma list; default all single-question types of the corpus (use 'multiturn' for conversations)")
    ap.add_argument("--qids", default="", help="comma list of question/conversation ids to run (overrides qtypes)")
    ap.add_argument("--limit", type=int, default=0, help="max questions per qtype (0=all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--rep", type=int, default=0, help="repetition index (records are keyed by it, so repeats are kept)")
    ap.add_argument("--kw", default="{}", help="json kwargs passed to the condition")
    ap.add_argument("--chunk_chars", type=int, default=700)
    ap.add_argument("--model", default=None, help="agent model (index-time extraction and judge stay on the default model)")
    ap.add_argument("--mult", type=int, default=1, help="hard corpus only: multiply the number of documents (entities), not the prose")
    ap.add_argument("--resolve", action="store_true", help="apply index-time supersession resolution to fact cards (v5)")
    args = ap.parse_args()
    kw = json.loads(args.kw)
    if args.model: llm.MODEL = args.model
    if args.corpus == "hard":
        docs, qs, convos, _ = corpus_hard.build_hard(filler_k=args.scale, mult=args.mult)
        if args.mult > 1: args.corpus = f"hardx{args.mult}"
    else:
        docs, qs, _ = corpus.build(filler_k=args.scale); convos = []
    all_types = sorted({q.qtype for q in qs}) + (["multiturn"] if convos else [])
    qtypes = args.qtypes.split(",") if args.qtypes else [t for t in all_types if t != "multiturn"]
    sel = []
    for t in qtypes:
        lst = [q for q in qs if q.qtype == t] if t != "multiturn" else convos
        sel += lst[:args.limit] if args.limit else lst
    if args.qids:
        ids = set(args.qids.split(",")); sel = [x for x in qs + convos if x.id in ids]
    conds = args.conditions.split(",")
    index = Index(docs, max_chars=args.chunk_chars) if any(c not in ("baseline", "repl") for c in conds) else None
    findex = None
    if any(c in ("vectorcontext_v4", "vectorcontext_v5") for c in conds):
        cards, fcost = build_facts(docs)
        if args.resolve:
            cards, rcost = resolve_updates(cards); fcost["resolve"] = {k: v for k, v in rcost.items() if k != "skipped_detail"}
            for k in ("prompt_tokens", "completion_tokens", "cost_usd"): fcost[k] += rcost[k]
            print("resolve:", {k: v for k, v in rcost.items() if k != "skipped_detail"}, flush=True)
        findex = FactIndex(docs, cards, resolved=args.resolve)
        print("fact layer:", {k: v for k, v in fcost.items() if k != "schemas"}, flush=True)
        with open(os.path.join(os.path.dirname(__file__), "..", "results", "index_costs.jsonl"), "a") as f:
            f.write(json.dumps(dict(scale=args.scale, corpus=args.corpus, n_docs=len(docs), **fcost)) + "\n")
    out = args.out or os.path.join(os.path.dirname(__file__), "..", "results", f"run_{args.corpus}{args.scale}{('_' + args.tag) if args.tag else ''}{('_' + args.model.split('/')[-1]) if args.model else ''}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    jobs = [(c, x) for c in conds for x in sel]
    print(f"{len(jobs)} jobs; corpus {args.corpus} {len(docs)} docs; scale {args.scale}; -> {out}", flush=True)
    base = dict(scale=args.scale, corpus=args.corpus, tag=args.tag, rep=args.rep, kw=kw, model=llm.MODEL)
    def one(c, session, q, convo=None, turn=None, cum=None):
        t0 = time.time()
        try:
            ans, m, trace = session.ask(q.question)
            ok, how = grade(ans, q); err = None
        except Exception:
            ans, m, trace, ok, how, err = "", llm.Meter(), [], False, "error", traceback.format_exc()[-500:]
        rec = dict(condition=c, **base, qid=q.id, qtype=q.qtype, question=q.question, gold=q.answer, answer=extract_answer(ans)[:300], correct=bool(ok), grader=how,
                   wall_s=time.time() - t0, error=err, trace=trace, convo=convo, turn=turn, **m.as_dict())
        if cum is not None:
            cum["prompt_tokens"] += rec["prompt_tokens"]; cum["cost_usd"] += rec["cost_usd"]
            rec["convo_prompt_tokens_cum"] = cum["prompt_tokens"]; rec["convo_cost_cum"] = cum["cost_usd"]
        return rec
    def work(c, x):
        session = CONDITIONS[c](docs, index, findex=findex, **kw)
        if hasattr(x, "turns"):
            cum = dict(prompt_tokens=0, cost_usd=0.0)
            return [one(c, session, q, convo=x.id, turn=k + 1, cum=cum) for k, q in enumerate(x.turns)]
        return [one(c, session, x)]
    with ThreadPoolExecutor(args.workers) as ex, open(out, "a") as f:
        futs = {ex.submit(work, c, x): (c, x) for c, x in jobs}
        done = 0
        for fut in as_completed(futs):
            done += 1
            for rec in fut.result():
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[{done}/{len(jobs)}] {rec['condition']:18s} {rec['qid']:12s} {'OK ' if rec['correct'] else 'BAD'} tok={rec['prompt_tokens']:6d} calls={rec['llm_calls']} ${rec['cost_usd']:.4f} | {str(rec['gold'])[:30]} vs {rec['answer'][:40]!r} {('ERR ' + rec['error'][-120:]) if rec['error'] else ''}", flush=True)
    print(f"session cost so far: ${llm.global_cost():.3f}")

if __name__ == "__main__":
    main()
