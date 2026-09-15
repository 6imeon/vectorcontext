"""Tables for the wild-idea conditions: accuracy, tokens and cost per condition (pooled over chats), plus per-qtype failures."""
import glob, json, os, sys, collections
RES = os.path.join(os.path.dirname(__file__), "..", "results")
def load(pattern):
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, pattern))):
        if "pilot" in f: continue
        rows += [json.loads(l) for l in open(f)]
    for r in rows:
        if r.get("model") and r["model"] != r.get("fold_model", r["model"]): r["condition"] += " @" + r["model"].split("/")[-1]
    return rows
def qtable(rows, conds=None, chats=None, style=None):
    agg = collections.defaultdict(lambda: dict(n=0, ok=0, tok=0, cost=0.0, calls=0, bad=collections.Counter()))
    for r in rows:
        if conds and r["condition"] not in conds: continue
        if chats and r["chat"] not in chats: continue
        if style and r.get("style", "clean") != style: continue
        a = agg[r["condition"]]; a["n"] += 1; a["ok"] += r["correct"]; a["tok"] += r["prompt_tokens"]; a["cost"] += r["cost_usd"]; a["calls"] += r["llm_calls"]
        if not r["correct"]: a["bad"][r["qtype"]] += 1
    print("| condition | n | accuracy | prompt tok/q | LLM calls/q | $/q | failures by type |"); print("|---|---|---|---|---|---|---|")
    for c, a in sorted(agg.items(), key=lambda kv: (-kv[1]["ok"] / kv[1]["n"], kv[1]["tok"])):
        bad = ", ".join(f"{k} {v}" for k, v in a["bad"].most_common())
        print(f"| {c} | {a['n']} | {100*a['ok']/a['n']:.0f}% | {a['tok']/a['n']:,.0f} | {a['calls']/a['n']:.1f} | {a['cost']/a['n']:.4f} | {bad} |")
def htable(rows, conds=None, chats=None, style=None):
    agg = collections.defaultdict(lambda: dict(n=0, found=0, req=0, full=0, tok=0, cost=0.0, stale=0))
    for r in rows:
        if conds and r["condition"] not in conds: continue
        if chats and r["chat"] not in chats: continue
        if style and r.get("style", "clean") != style: continue
        a = agg[r["condition"]]; a["n"] += 1; a["found"] += r["found"]; a["req"] += r["required"]; a["full"] += r["full_credit"]; a["tok"] += r["prompt_tokens"]; a["cost"] += r["cost_usd"]; a["stale"] += len(r["stale_present"])
    print("| worker | tasks | facts found | complete notes | stale values | prompt tok/task | $/task |"); print("|---|---|---|---|---|---|---|")
    for c, a in sorted(agg.items(), key=lambda kv: (-kv[1]["found"], kv[1]["tok"])):
        print(f"| {c} | {a['n']} | {a['found']}/{a['req']} | {a['full']}/{a['n']} | {a['stale']} | {a['tok']/a['n']:,.0f} | {a['cost']/a['n']:.4f} |")
if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "wild"
    rows = [r for r in load(f"chat_{tag}*.jsonl") if r["tag"] == tag]
    if rows:
        for style in sorted({r.get("style", "clean") for r in rows}):
            print(f"\n### probes, style={style}\n"); qtable(rows, style=style)
    hrows = [r for r in load(f"handoff_{tag}*.jsonl") if r["tag"] == tag]
    if hrows:
        for style in sorted({r.get("style", "clean") for r in hrows}):
            print(f"\n### hand-off, style={style}\n"); htable(hrows, style=style)
