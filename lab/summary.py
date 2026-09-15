"""Print markdown summary tables from results/run_*.jsonl (same dedupe rules as report.py)."""
from __future__ import annotations
import os, sys, statistics as st
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__))
from report import load, label, QTYPES

def main(patterns):
    recs = load(patterns)
    g = defaultdict(list)
    for r in recs: g[(r["corpus"], r["scale"], label(r))].append(r)
    for corp_scale in sorted({(k[0], k[1]) for k in g}):
        rows = [(k, v) for k, v in g.items() if (k[0], k[1]) == corp_scale]
        types = [t for t in QTYPES if any(r["qtype"] == t for _, v in rows for r in v)]
        print(f"\n### corpus {corp_scale[0]}, scale {corp_scale[1]}\n")
        print("| condition | n | accuracy | " + " | ".join(types) + " | prompt tok/q | peak tok | cached | calls | cost/q |")
        print("|" + "---|" * (8 + len(types)))
        for (corp, sc, cond), v in sorted(rows, key=lambda x: x[0][2]):
            n = len(v); acc = sum(r["correct"] for r in v) / n
            bt = " | ".join((f"{sum(r['correct'] for r in v if r['qtype']==t)/max(1,sum(1 for r in v if r['qtype']==t)):.0%}" if any(r["qtype"] == t for r in v) else "") for t in types)
            print(f"| {cond} | {n} | {acc:.0%} | {bt} | {st.mean(r['prompt_tokens'] for r in v):,.0f} | {st.mean(r['peak_prompt_tokens'] for r in v):,.0f} | "
                  f"{sum(r.get('cached_tokens',0) for r in v)/max(1,sum(r['prompt_tokens'] for r in v)):.0%} | {st.mean(r['llm_calls'] for r in v):.1f} | ${st.mean(r['cost_usd'] for r in v):.4f} |")
    # conversations
    gc = defaultdict(list)
    for r in recs:
        if r.get("convo"): gc[(r["corpus"], r["scale"], label(r), r["convo"])].append(r)
    agg = defaultdict(list)
    for (corp, sc, cond, cv), v in gc.items(): agg[(corp, sc, cond)].append((sum(r["prompt_tokens"] for r in v), all(r["correct"] for r in v), max(r["peak_prompt_tokens"] for r in v)))
    if agg:
        print("\n### conversations (4 turns each)\n\n| corpus | condition | conversations | all 4 turns correct | prompt tokens per conversation | peak single call |\n|---|---|---|---|---|---|")
        for (corp, sc, cond), v in sorted(agg.items()):
            print(f"| {corp} | {cond} | {len(v)} | {sum(x[1] for x in v)} | {sum(x[0] for x in v)/len(v):,.0f} | {sum(x[2] for x in v)/len(v):,.0f} |")

if __name__ == "__main__":
    main(sys.argv[1:] or ["run_*.jsonl"])
