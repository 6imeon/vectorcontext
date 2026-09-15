"""Index-time fact-card extraction: one LLM call per document (cached), then key harmonization per doc type.
This is the 'semantic layer' of vectorcontext: paraphrase-robust, compact, and aggregatable from code."""
from __future__ import annotations
import hashlib, json, os, re
from concurrent.futures import ThreadPoolExecutor
from llm import chat, Meter, MODEL
from index import _atomic_save

CACHE = os.path.join(os.path.dirname(__file__), "..", "results", "facts_cache.json")
EXTRACT_SYS = ("Extract every concrete, specific fact from the document as ONE flat JSON object. Rules: short snake_case keys; values normalized "
               "(numbers as numbers, people as 'First Last', lists as JSON arrays, keep identifiers verbatim). Always include 'subject' (the entity the doc is about) "
               "and 'aliases' (other names used for the subject). Skip generic process boilerplate (deploy pipelines, backups, retention...). Output compact single-line JSON only, no prose, no markdown.")

SCHEMA_SYS = ("You design a compact extraction schema. Given sample documents of one type, output a JSON object mapping snake_case field names to one-line descriptions, "
              "covering every kind of concrete, query-worthy fact these documents state (identifiers, names, people, teams, numbers, categories, dates, dependencies, decisions). "
              "Always include 'subject' and 'aliases'. Every field must hold ONE atomic value (a name, number, identifier, category, date) or a flat list of such values; never combine two facts in one field "
              "(e.g. deployment_region and replica_count are two fields, not deployment_topology). Prefer one canonical field per fact (owner_team, not both team and owner). "
              "Skip generic process boilerplate and long event timelines. 8-16 fields. JSON only.")

def induce_schema(doc_type, sample_docs):
    m = Meter()
    msg = chat([{"role":"system","content":SCHEMA_SYS},{"role":"user","content":"\n\n=====\n\n".join(d.text for d in sample_docs)}], m, max_tokens=1200, reasoning_effort="low", model=MODEL)
    txt = re.sub(r"^```(?:json)?|```$", "", (msg.content or "").strip(), flags=re.M).strip()
    try: schema = json.loads(txt)
    except Exception:
        mm = re.search(r"\{.*\}", txt, re.S); schema = json.loads(mm.group(0)) if mm else {}
    return schema, m.as_dict()

def _extract_one(doc, schema=None, _retry=False):
    m = Meter()
    sysmsg = EXTRACT_SYS if not schema else (
        "Extract facts from the document into ONE flat JSON object using EXACTLY these fields (use null when the document does not state a value; never invent):\n"
        + json.dumps(schema) + "\nRules: values normalized (numbers as numbers, people as 'First Last', lists as JSON arrays, identifiers verbatim, "
        "categorical values lower-case as written in the text). If a service is referred to by an alias like 'the Billing API', keep it as written. Output compact single-line JSON only.")
    msg = chat([{"role":"system","content":sysmsg},{"role":"user","content":doc.text}], m, max_tokens=2500 if _retry else 1500, reasoning_effort="low", model=MODEL)
    txt = (msg.content or "").strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    card = None
    try: card = json.loads(txt)
    except Exception:
        mm = re.search(r"\{.*\}", txt, re.S)
        if mm:
            try: card = json.loads(mm.group(0))
            except Exception: card = None
    if not isinstance(card, dict) or len(card) < 3:
        if not _retry:
            card2, u2 = _extract_one(doc, schema, _retry=True)
            u = m.as_dict()
            for k in ("prompt_tokens","completion_tokens","cost_usd"): u2[k] += u[k]
            return card2, u2
        card = {"raw": txt[:1500]}
    return card, m.as_dict()

def build_facts(docs, workers=8, schema_first=True, samples=8):
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    cost = dict(prompt_tokens=0, completion_tokens=0, cost_usd=0.0, docs_extracted=0, mode="schema" if schema_first else "free")
    schemas = {}
    if schema_first:
        by_type = {}
        for d in docs: by_type.setdefault(d.type, []).append(d)
        for t, ds in by_type.items():
            sample = sorted(ds, key=lambda d: d.id)[:samples]
            skey = "schema2|" + t + "|" + hashlib.sha1("".join(d.text for d in sample).encode()).hexdigest()
            if skey not in cache:
                cache[skey], u = induce_schema(t, sample)
                for k in ("prompt_tokens","completion_tokens","cost_usd"): cost[k] += u[k]
            schemas[t] = cache[skey]
    tag = "fact2|" if schema_first else "fact1|"
    keys = {d.id: hashlib.sha1((MODEL+"|"+tag+json.dumps(schemas.get(d.type), sort_keys=True)+"|"+d.text).encode()).hexdigest() for d in docs}
    def _bad(c): return not isinstance(c, dict) or "raw" in c or len(c) < 3
    todo = [d for d in docs if keys[d.id] not in cache or _bad(cache[keys[d.id]])]
    cost["docs_extracted"] = len(todo)
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for d, (card, u) in zip(todo, ex.map(lambda d: _extract_one(d, schemas.get(d.type)), todo)):
                cache[keys[d.id]] = card
                for k in ("prompt_tokens","completion_tokens","cost_usd"): cost[k] += u[k]
    _atomic_save(CACHE, cache)
    cards = [dict(doc_id=d.id, type=d.type, title=d.title, **{k: _nullify(v) for k, v in cache[keys[d.id]].items() if k not in ("doc_id","type","title") and _atomic(v)}) for d in docs]
    if not schema_first:
        cards, hcost = harmonize(cards, cache)
        for k in ("prompt_tokens","completion_tokens","cost_usd"): cost[k] += hcost[k]
    cards, cost["type_nulled"] = type_check(cards, schemas)
    cards, cost["linked_values"] = link_entities(cards)
    cards, ccost = canonicalize_values(cards, cache)
    for k in ("prompt_tokens","completion_tokens","cost_usd"): cost[k] += ccost[k]
    cards, n2 = link_entities(cards); cost["linked_values"] += n2  # canonicalization may re-introduce alias forms
    cost["schemas"] = schemas
    return cards, cost

CANON_SYS = ("You canonicalize field values extracted from documents. Given a field name and its distinct raw string values, group values that refer to the same real-world thing "
             "(e.g. 'Platform', 'Platform team', 'the Platform Team'; 'Redis', 'Redis cluster'; 'sev1', 'SEV1'). Output a JSON object mapping EVERY raw value to a canonical value. "
             "If a group contains a machine-style identifier (hyphenated or underscored, e.g. 'billing-service'), that identifier IS the canonical value; never shorten or strip identifiers. "
             "Otherwise choose the shortest clean form (e.g. 'Platform', 'Redis', 'SEV1'). Never merge genuinely different entities. JSON only.")
CANON_TAG = "canon2|"

def link_entities(cards):
    """Deterministic entity linking: any value equal to a card's alias (or subject) is replaced by that card's subject, so
    'the Billing API' / 'Billing API' / 'billing-service' all become the identifier 'billing-service' and joins are exact."""
    def norm(v): return re.sub(r"^(the|a|an)\s+", "", re.sub(r"[^a-z0-9]+", " ", str(v).lower()).strip())
    amap = {}
    for pass_ in ("alias", "self"):  # real alias->subject entries win over a card's own subject (a change record's subject may itself be an alias)
        for c in cards:
            subj = c.get("subject")
            if not isinstance(subj, str) or not subj.strip(): continue
            als = c.get("aliases") if isinstance(c.get("aliases"), list) else [c.get("aliases")] if isinstance(c.get("aliases"), str) else []
            for a in ([a for a in als if isinstance(a, str) and a.strip() and norm(a) != norm(subj)] if pass_ == "alias" else [subj]):
                amap.setdefault(norm(a), subj)
    n = 0
    def link(v):
        nonlocal n
        if isinstance(v, str):
            t = amap.get(norm(v))
            if t and t != v: n += 1; return t
            return v
        if isinstance(v, list): return [link(x) for x in v]
        return v
    for c in cards:
        for k in list(c):
            if k in ("doc_id", "type", "title", "aliases", "history", "current"): continue
            c[k] = link(c[k])
    return cards, n

def canonicalize_values(cards, cache, max_distinct=80):
    total=dict(prompt_tokens=0, completion_tokens=0, cost_usd=0.0)
    by_type={}
    for c in cards: by_type.setdefault(c["type"], []).append(c)
    for t, cs in by_type.items():
        fields = sorted({k for c in cs for k in c if k not in ("doc_id","type","title","subject","aliases")})
        for f in fields:
            vals = sorted({c[f] for c in cs if isinstance(c.get(f), str) and c[f].strip()})
            if len(vals) < 2 or len(vals) > max_distinct: continue
            ckey = CANON_TAG+t+"|"+f+"|"+hashlib.sha1(json.dumps(vals).encode()).hexdigest()
            if ckey not in cache:
                m = Meter()
                msg = chat([{"role":"system","content":CANON_SYS},{"role":"user","content":f"field: {f}\nvalues: {json.dumps(vals)}"}], m, max_tokens=3000, reasoning_effort="low", model=MODEL)
                txt = re.sub(r"^```(?:json)?|```$", "", (msg.content or "").strip(), flags=re.M).strip()
                try: mapping = json.loads(txt)
                except Exception: mapping = {}
                cache[ckey] = mapping if isinstance(mapping, dict) else {}
                u = m.as_dict()
                for k in ("prompt_tokens","completion_tokens","cost_usd"): total[k] += u[k]
            mapping = cache[ckey]
            for c in cs:
                if isinstance(c.get(f), str) and c[f] in mapping and isinstance(mapping[c[f]], str): c[f] = mapping[c[f]]
    _atomic_save(CACHE, cache)
    return cards, total

def harmonize(cards, cache):
    """Ask the model once per doc type for a synonym->canonical key map, so python filters see consistent keys."""
    from collections import Counter
    out=[]; total=dict(prompt_tokens=0, completion_tokens=0, cost_usd=0.0)
    by_type={}
    for c in cards: by_type.setdefault(c["type"], []).append(c)
    for t, cs in by_type.items():
        keys = Counter(k for c in cs for k in c if k not in ("doc_id","type","title"))
        hkey = "harm1|"+t+"|"+hashlib.sha1(json.dumps(sorted(keys)).encode()).hexdigest()
        if hkey in cache: mapping = cache[hkey]
        else:
            m = Meter()
            msg = chat([{"role":"system","content":"You normalize JSON schemas. Given key names (with frequencies) extracted independently from documents of the same type, output a JSON object mapping EVERY key to a canonical key, merging synonyms (e.g. owner_team/owning_team/team -> owner_team). Keep the most common name as canonical. JSON only."},
                        {"role":"user","content":json.dumps(keys.most_common())}], m, max_tokens=1500)
            txt = re.sub(r"^```(?:json)?|```$", "", (msg.content or "").strip(), flags=re.M).strip()
            try: mapping = json.loads(txt)
            except Exception: mapping = {}
            cache[hkey] = mapping; _atomic_save(CACHE, cache)
            u = m.as_dict()
            for k in ("prompt_tokens","completion_tokens","cost_usd"): total[k] += u[k]
        for c in cs:
            nc = {"doc_id": c["doc_id"], "type": c["type"], "title": c["title"]}
            for k, v in c.items():
                if k in nc: continue
                ck = mapping.get(k, k)
                if ck in nc and nc[ck] != v:
                    nc[ck] = nc[ck] if isinstance(nc[ck], list) and v in nc[ck] else ([nc[ck], v] if not isinstance(nc[ck], list) else nc[ck] + [v])
                else: nc[ck] = v
            out.append(nc)
    return out, total

RESOLVE_SYS = ("You perform temporal fact resolution over fact cards extracted from documents. Some cards describe a CHANGE to an attribute of ANOTHER entity "
               "(ownership transfer, on-call handover, migration to another region/datastore, renames) effective on a date. For each such change output one patch object: "
               '{"source": <doc_id of the change card>, "target": <doc_id of the card whose attribute is superseded>, "field": <exact existing field name on the target card>, '
               '"from": <old value or null>, "to": <new value or null>, "date": "YYYY-MM-DD"}. Match targets by subject/aliases (an alias like "the Billing API" refers to billing-service). '
               "Use the target card's field names and value forms exactly. Never patch based on plans or proposals that have not taken effect. Output a JSON array only; empty array if there are no change records.")

def resolve_updates(cards, cache_path=CACHE, batch=60):
    """Index-time supersession: one LLM call finds change records among the cards and emits patches; patches are applied in date order
    (latest wins) and recorded on the target card as `history`. Mirrors temporal fact invalidation in Graphiti/Zep (arXiv:2501.13956)."""
    import copy
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    cost = dict(prompt_tokens=0, completion_tokens=0, cost_usd=0.0)
    # entity directory (doc_id, subject, aliases, keys) so a batch can target any card; cards are resolved in batches so recall
    # does not degrade with corpus size (a single 67k-token call missed 2 of 16 change records at 504 docs)
    directory = "\n".join(f"{c['doc_id']} | {c.get('subject')} | aliases: {c.get('aliases')} | fields: {', '.join(k for k in c if k not in ('doc_id','type','title','subject','aliases'))}" for c in cards)
    patches = []
    for i in range(0, len(cards), batch):
        chunk = cards[i:i+batch]
        texts = "\n".join(f"{c['doc_id']} :: {card_text(c)}" for c in chunk)
        key = "resolve2|" + hashlib.sha1((MODEL + directory + texts).encode()).hexdigest()
        if key not in cache:
            m = Meter()
            msg = chat([{"role":"system","content":RESOLVE_SYS + " You receive a directory of ALL cards (id | subject | aliases | fields) and then a BATCH of full cards; emit patches only for change records that appear in the batch, targeting any card in the directory."},
                        {"role":"user","content":f"<directory>\n{directory}\n</directory>\n\n<batch>\n{texts}\n</batch>"}], m, max_tokens=6000, reasoning_effort="low", model=MODEL)
            txt = re.sub(r"^```(?:json)?|```$", "", (msg.content or "").strip(), flags=re.M).strip()
            try: got = json.loads(txt)
            except Exception:
                mm = re.search(r"\[.*\]", txt, re.S); got = json.loads(mm.group(0)) if mm else []
            cache[key] = got if isinstance(got, list) else []
            _atomic_save(cache_path, cache); u = m.as_dict()
            for k in ("prompt_tokens","completion_tokens","cost_usd"): cost[k] += u[k]
        patches += cache[key]
    cards = copy.deepcopy(cards); by_id = {c["doc_id"]: c for c in cards}
    applied = 0; skipped = []
    for p in sorted((p for p in patches if isinstance(p, dict)), key=lambda p: str(p.get("date", ""))):
        tgt = by_id.get(p.get("target")); f = p.get("field")
        if not tgt or not f or f not in tgt: skipped.append(p); continue
        cur = tgt.setdefault("current", {})
        cur.setdefault(f, tgt.get(f))  # provenance: the card's own fields keep what the source document states
        tgt.setdefault("history", []).append(dict(field=f, **{"from": cur[f]}, to=_nullify(p.get("to")), date=p.get("date"), source=p.get("source")))
        cur[f] = _nullify(p.get("to")); applied += 1
    return cards, dict(cost, patches=len(patches), applied=applied, skipped=len(skipped), skipped_detail=skipped[:10])

PERSON_RE = re.compile(r"^[A-Z][a-z'\-]+(?: [A-Z][a-z'\-]+){1,3}$")
def type_check(cards, schemas):
    """Deterministic validation against the induced schema: a field whose description says it holds a person (engineer, contact,
    lead, author, approver) must contain a proper name; role words such as 'team lead' or 'unassigned' become null."""
    n = 0
    for c in cards:
        sch = schemas.get(c["type"]) or {}
        for k, desc in sch.items():
            d = str(desc).lower()
            if k in c and isinstance(c[k], str) and re.search(r"\b(person|engineer|contact|lead|author|approver|approved by|owner name|individual)\b", d) and not re.search(r"\bteam\b(?!.*\b(lead|leader)\b)", d):
                if not PERSON_RE.match(c[k].strip()): c[k] = None; n += 1
    return cards, n

NULLISH = {"null","none","n/a","na","unassigned","not specified","not stated","unknown","unspecified","tbd","-",""}
def _nullify(v):
    if isinstance(v, str) and v.strip().lower() in NULLISH: return None
    if isinstance(v, list): return [x for x in (_nullify(x) for x in v) if x is not None] or None
    return v

def _atomic(v, limit=80):
    """Keep only atomic facts: short scalars or short lists of short scalars. Long prose fields belong to the raw doc, not the card."""
    if v is None or isinstance(v, (int, float, bool)): return True
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v): return len(v) <= 12  # history entries
    if isinstance(v, dict): return len(v) <= 12  # `current` view
    if isinstance(v, str): return len(v) <= limit
    if isinstance(v, list): return len(v) <= 12 and all(_atomic(x, limit) for x in v)
    return False

def card_text(c):
    return f"[{c['title']}] " + "; ".join(f"{k}: {json.dumps(v) if not isinstance(v,str) else v}" for k, v in c.items() if k not in ("doc_id","type","title"))

if __name__ == "__main__":
    import sys; sys.path.insert(0, os.path.dirname(__file__)); import corpus
    docs, qs, _ = corpus.build(filler_k=int(sys.argv[1]) if len(sys.argv)>1 else 3)
    cards, cost = build_facts(docs)
    print(cost)
    for c in cards[:6]: print(card_text(c)[:300])
    import tiktoken; enc=tiktoken.get_encoding("o200k_base")
    print("fact layer tokens:", sum(len(enc.encode(card_text(c))) for c in cards), "vs corpus", sum(len(enc.encode(d.text)) for d in docs))
