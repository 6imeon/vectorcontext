"""Measure fact-card extraction accuracy against the generator's ground truth (service docs)."""
import sys, os, json, re; sys.path.insert(0, os.path.dirname(__file__))
import corpus
from facts import build_facts
docs, qs, meta = corpus.build(filler_k=int(sys.argv[1]) if len(sys.argv)>1 else 3)
cards, cost = build_facts(docs)
by = {c["doc_id"]: c for c in cards}
def norm(v): return re.sub(r"[^a-z0-9]+"," ",str(v).lower()).strip()
SYN = {"team": ["owner_team","owning_team","team","maintaining_team","owner"], "db": ["primary_datastore","datastore","database","data_store","state_store"],
       "oncall": ["primary_on_call","on_call_engineer","on_call","oncall","pager_owner","primary_on_call_engineer","on_call_contact","oncall_engineer","on_call_person"], "port": ["network_port","listening_port","port","http_port","tcp_port"],
       "lang": ["implementation_language","programming_language","language"], "region": ["deployment_region","region","deploy_region"]}
def get(c, gf):
    for k in SYN[gf]:
        if k in c: return c[k]
    return None
tot=0; ok=0; errs=[]
for s in meta["services"]:
    c = by[f"svc-{s['name']}"]
    for gf in SYN:
        tot += 1; g = s[gf]; v = get(c, gf); cf = gf
        good = (v in (None, "", [], "null", "unassigned") ) if g is None else (v is not None and norm(g) in norm(v))
        ok += good
        if not good: errs.append((s["name"], cf, g, v))
print(f"service field accuracy: {ok}/{tot} = {ok/tot:.1%}")
for e in errs: print("  miss:", e)
# incidents
tot=ok=0
for inc in meta["incidents"]:
    c = by[f"inc-{inc['id']}"]
    for cf, g in (("severity", inc["sev"]), ("root_cause", inc["cause"]), ("duration_minutes", inc["minutes"])):
        tot += 1; v = c.get(cf) if cf != "duration_minutes" else (c.get(cf) if cf in c else c.get("incident_duration_minutes")); good = v is not None and norm(g) in norm(v); ok += good
        if not good: errs.append((inc["id"], cf, g, v))
print(f"incident field accuracy: {ok}/{tot} = {ok/tot:.1%}")
for e in errs[-6:]: print("  miss:", e)
sc=int(sys.argv[1]) if len(sys.argv)>1 else 3
with open(os.path.join(os.path.dirname(__file__), "..", "results", "extraction_accuracy.jsonl"), "a") as f:
    f.write(json.dumps(dict(scale=sc, service_fields_checked=240, service_field_errors=len([e for e in errs if not e[0].startswith("INC")]), incident_fields_checked=180, incident_field_errors=len([e for e in errs if e[0].startswith("INC")]), fact_layer_docs=len(cards)))+"\n")
