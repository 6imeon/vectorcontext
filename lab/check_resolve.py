"""Verify the index-time supersession resolver against generator ground truth (no LLM calls unless caches are cold):
for every service, the resolved card's owner/on-call/region must equal the generator's current state."""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import corpus_hard
from facts import build_facts, resolve_updates

SYN = {"team": ["owner_team", "owning_team", "team", "owner"], "oncall": ["on_call_engineer", "on_call", "oncall", "primary_on_call", "on_call_person", "pager_contact", "primary_on_call_engineer"],
       "region": ["region", "deployment_region", "deployed_region"]}

def main(scale=3, mult=1):
    docs, qs, convos, meta = corpus_hard.build_hard(filler_k=scale, mult=mult)
    cards, _ = build_facts(docs); cards, rcost = resolve_updates(cards)
    by_doc = {c["doc_id"]: c for c in cards}
    changed = {c["service"] for c in meta["changes"]}
    errs = []; checked = 0
    for s in meta["services"]:
        card = by_doc[f"svc-{s['name']}"]; cur = meta["current"][s["name"]]
        for f, keys in SYN.items():
            k = next((k for k in keys if k in card), None)
            if k is None: errs.append((s["name"], f, "no key")); continue
            got = card.get("current", {}).get(k, card[k]); want = cur[f]
            checked += 1
            ok = (got is None and want is None) or (got is not None and want is not None and str(want).lower() in str(got).lower())
            if not ok: errs.append((s["name"], f, f"card={got!r} want={want!r} changed={s['name'] in changed}"))
    print(json.dumps(dict(scale=scale, mult=mult, patches=rcost["patches"], applied=rcost["applied"], skipped=rcost["skipped"], fields_checked=checked, errors=len(errs))))
    for e in errs: print("  ", e)
    # which generator changes were not reflected?
    for c in meta["changes"]:
        card = by_doc[f"svc-{c['service']}"]; hist = card.get("history", [])
        if not any(h.get("source") == f"chg-{c['id']}" for h in hist): print("   missing patch for", c["id"], c["service"], c["field"], c["old"], "->", c["new"])
    with open(os.path.join(os.path.dirname(__file__), "..", "results", "resolve_accuracy.jsonl"), "a") as f:
        f.write(json.dumps(dict(scale=scale, corpus=f"hard{'x'+str(mult) if mult>1 else ''}", fields_checked=checked, errors=len(errs), patches=rcost["patches"], applied=rcost["applied"], detail=[list(map(str, e)) for e in errs])) + "\n")

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3, int(sys.argv[2]) if len(sys.argv) > 2 else 1)
