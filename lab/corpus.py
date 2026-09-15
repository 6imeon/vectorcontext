"""Synthetic engineering knowledge-base corpus with exact ground-truth questions.

Everything is seeded so runs are reproducible. Facts are stated with varied
phrasing and entity aliases so that naive lexical search and naive vector
search each have realistic failure modes.
"""
from __future__ import annotations
import random, json, itertools
from dataclasses import dataclass, field, asdict

@dataclass
class Doc:
    id: str
    type: str
    title: str
    text: str

@dataclass
class Question:
    id: str
    qtype: str          # needle | multihop | aggregation | global
    question: str
    answer: str | int | list
    aliases: list = field(default_factory=list)  # alternative acceptable strings
    notes: str = ""

FIRST = ["Priya","Marcus","Elena","Tomasz","Aiko","Daniel","Fatima","Lucas","Ingrid","Rafael","Chen","Amara","Noah","Sofia","Kwame","Hannah","Yusuf","Mei","Oliver","Zara","Ivan","Leila","Bruno","Nadia","Sven","Rosa","Tariq","Ana","Jonas","Keiko"]
LAST = ["Nair","Bell","Petrova","Kowalski","Tanaka","Reyes","Haddad","Moreau","Lindqvist","Costa","Wei","Okafor","Fischer","Rossi","Mensah","Berg","Demir","Lin","Hart","Malik","Volkov","Farah","Silva","Novak","Larsen","Ortega","Aziz","Duarte","Meyer","Sato"]
TEAMS = ["Payments","Identity","Search","Growth","Platform","Data","Messaging","Commerce"]
SVC_A = ["billing","ledger","auth","session","catalog","cart","checkout","pricing","inventory","notify","mailer","search","index","ranker","profile","feed","recsys","export","audit","metrics","tracing","gateway","edge","scheduler","queue","webhook","fraud","risk","kyc","payout","refund","invoice","tax","geo","i18n","media","thumbnail","upload","cdn","config"]
SVC_A += ["orders","payments","shipping","returns","loyalty","coupons","wallet","escrow","settlement","dispute","chargeback","statement","reconcile","forecast","budget","quota","tenant","org","roles","permissions","tokens","oauth","sso","mfa","device","push","sms","email","digest","campaign","segment","experiment","flags","rollout","canary","builder","deploy","registry","artifact","backup","archive","restore","snapshot","replica","shard","cache","lookup","autocomplete","spell","synonym","crawler","parser","tokenizer","embed","vector","cluster","stream","batch","etl","warehouse","lake","report","dashboard","alerting","pager","status","health","probe","chaos","loadgen","ledgerx","vault","secrets","license","entitlement","subscription","trial","referral","survey","support","ticket"]  # extra prefixes for document-count scaling (first 40 unchanged, so mult=1 corpora are identical)
SVC_B = ["service","api","worker","engine","svc","core","gateway","daemon"]
LANGS = ["Go","Python","Java","Rust","TypeScript","Kotlin"]
REGIONS = ["us-east-1","eu-west-1","ap-southeast-2","us-west-2"]
DBS = ["Postgres","MySQL","DynamoDB","Cassandra","Redis","MongoDB","CockroachDB"]
CAUSES = ["configuration error","capacity exhaustion","bad deploy","dependency outage","certificate expiry","database failover","memory leak","network partition"]
SEV = ["SEV1","SEV2","SEV3"]

def _name(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"

def build(seed: int = 7, n_services: int = 40, n_incidents: int = 60, n_notes: int = 30, n_filler: int = 30, filler_k: int = 3):
    rng = random.Random(seed)
    # ---- teams
    teams = []
    for t in TEAMS:
        teams.append(dict(name=t, lead=_name(rng), slack=f"#{t.lower()}-{rng.choice(['eng','team','oncall','dev'])}", floor=rng.randint(2,9)))
    # ---- services
    services = []
    prefixes = rng.sample(SVC_A[:max(40, n_services)], n_services)  # first 40 only at default size, so existing corpora are unchanged
    for i in range(n_services):
        a=prefixes[i]; b=rng.choice(SVC_B); nm=f"{a}-{b}"
        alias = f"the {a.capitalize()} {rng.choice(['API','Service','System','Platform'])}"
        team = rng.choice(teams)
        oncall = _name(rng) if rng.random() > 0.15 else None   # some have none -> global/negation Qs
        services.append(dict(name=nm, alias=alias, team=team["name"], oncall=oncall, port=rng.randint(3000,9999),
            lang=rng.choice(LANGS), region=rng.choice(REGIONS), db=rng.choice(DBS), slo=rng.choice([99.5,99.9,99.95,99.99]),
            replicas=rng.randint(2,24), deps=[]))
    for s in services:
        s["deps"] = rng.sample([x["name"] for x in services if x is not s], rng.randint(1,3))
    # ---- incidents
    incidents=[]
    for i in range(n_incidents):
        s=rng.choice(services)
        incidents.append(dict(id=f"INC-{2000+i*7+rng.randint(0,5)}", service=s["name"], sev=rng.choice(SEV), cause=rng.choice(CAUSES),
            minutes=rng.randint(4,240), month=rng.randint(1,12), day=rng.randint(1,28), author=_name(rng), year=2025))
    # ---- notes / ADRs
    notes=[]
    for i in range(n_notes):
        s=rng.choice(services); newdb=rng.choice([d for d in DBS if d!=s["db"]])
        notes.append(dict(id=f"ADR-{100+i}", service=s["name"], newdb=newdb, quarter=rng.choice(["Q1","Q2","Q3","Q4"]), decided_by=_name(rng)))

    trng = random.Random(seed*31+1)  # prose rng, independent of entity/question rng
    docs: list[Doc] = []
    # team pages
    for t in teams:
        text = (f"# Team page: {t['name']}\n\nThe {t['name']} team is led by {t['lead']}. "
                f"For urgent requests, page the team via Slack in {t['slack']}. The team sits on floor {t['floor']} of the Berlin office. "
                f"Weekly planning happens on {trng.choice(['Monday','Tuesday','Wednesday'])} mornings. "
                + _filler(trng, filler_k))
        docs.append(Doc(f"team-{t['name'].lower()}", "team", f"Team page: {t['name']}", text))
    # service docs: facts spread over paragraphs, alias used in about half of the fact sentences
    for s in services:
        n = lambda: s["name"] if trng.random()<0.5 else s["alias"]
        port_s = trng.choice([f"{n()} listens on port {s['port']}.", f"Traffic to {n()} is bound to TCP port {s['port']}.", f"The process for {n()} exposes HTTP on :{s['port']}."])
        lang_s = trng.choice([f"It is implemented in {s['lang']}.", f"The codebase is written in {s['lang']}.", f"{n()} is a {s['lang']} service."])
        own_s = trng.choice([f"{n()} is owned by the {s['team']} team.", f"Ownership: {s['team']}.", f"The {s['team']} team maintains {n()}."])
        oc_s = (trng.choice([f"The primary on-call engineer is {s['oncall']}.", f"Pages go to {s['oncall']} first.", f"{s['oncall']} carries the pager for {n()}."])
                if s["oncall"] else trng.choice(["There is currently no designated on-call engineer; pages fall through to the team lead.", "On-call: unassigned (see team page)."]))
        db_s = trng.choice([f"State is persisted in {s['db']}.", f"{n()} stores its data in a {s['db']} cluster.", f"Primary datastore: {s['db']}."])
        reg_s = trng.choice([f"It runs in {s['region']} with {s['replicas']} replicas.", f"Deployment region is {s['region']}; the deployment has {s['replicas']} replicas.", f"{s['replicas']} replicas are deployed to {s['region']}."])
        dep_s = f"Runtime dependencies: {', '.join(s['deps'])}."
        slo_s = trng.choice([f"The availability SLO is {s['slo']}%.", f"We commit to {s['slo']}% availability."])
        paras = [port_s, lang_s, own_s, oc_s, db_s, reg_s, dep_s, slo_s]
        trng.shuffle(paras)
        # interleave filler sentences
        body = []
        for p in paras:
            body.append(p + " " + _filler(trng, filler_k))
        text = f"# Service overview: {s['name']}\n\n{s['name']} (also referred to internally as {s['alias']}) is a backend component. " + "\n\n".join(body)
        docs.append(Doc(f"svc-{s['name']}", "service", f"Service overview: {s['name']}", text))
    # incidents
    for inc in incidents:
        svc = next(x for x in services if x["name"]==inc["service"])
        n = svc["name"] if trng.random()<0.6 else svc["alias"]
        text = (f"# Incident report {inc['id']}\n\nSeverity: {inc['sev']}\nDate: {inc['year']}-{inc['month']:02d}-{inc['day']:02d}\n\n"
                f"Summary: {trng.choice(['Elevated error rates', 'Latency spike', 'Partial outage', 'Complete outage'])} affecting {n}. "
                f"The incident lasted {inc['minutes']} minutes before full recovery. "
                f"Root cause was determined to be a {inc['cause']}. "
                f"This report was written by {inc['author']}. " + _filler(trng, 3*filler_k) + "\n\nTimeline:\n" + _timeline(trng))
        docs.append(Doc(f"inc-{inc['id']}", "incident", f"Incident report {inc['id']}", text))
    # notes
    for nt in notes:
        text = (f"# Architecture decision {nt['id']}\n\nDecision: migrate {nt['service']} to {nt['newdb']} in {nt['quarter']} 2026. "
                f"Approved by {nt['decided_by']}. " + _filler(trng, 3*filler_k))
        docs.append(Doc(f"adr-{nt['id']}", "adr", f"Architecture decision {nt['id']}", text))
    # filler runbooks
    for i in range(n_filler):
        docs.append(Doc(f"runbook-{i}", "runbook", f"Runbook {i}: {trng.choice(['rotating certificates','scaling consumers','draining a node','rolling back a deploy','rotating secrets','pruning logs'])}", "# Runbook\n\n" + "\n\n".join(_filler(trng, 5) for _ in range(filler_k*2))))
    trng.shuffle(docs)

    # ---- questions
    qs: list[Question] = []
    by_name = {s["name"]: s for s in services}
    tm = {t["name"]: t for t in teams}
    # needle (10): mix of canonical name / alias in question
    for i, s in enumerate(rng.sample(services, 10)):
        ref = s["name"] if i%2==0 else s["alias"]
        attr = rng.choice(["port","lang","db","region"])
        qtext = {"port": f"What port does {ref} listen on?", "lang": f"What language is {ref} written in?", "db": f"What datastore does {ref} use?", "region": f"Which region is {ref} deployed in?"}[attr]
        qs.append(Question(f"needle-{i}", "needle", qtext, s[attr]))
    # multihop (10)
    for i in range(10):
        kind = i % 3
        if kind == 0:
            inc = rng.choice(incidents); s = by_name[inc["service"]]
            if not s["oncall"]:
                qs.append(Question(f"multihop-{i}", "multihop", f"Which team owns the service affected by {inc['id']}?", s["team"]))
            else:
                qs.append(Question(f"multihop-{i}", "multihop", f"Who is the on-call engineer for the service affected by {inc['id']}?", s["oncall"]))
        elif kind == 1:
            s = rng.choice(services); dep = by_name[s["deps"][0]]
            qs.append(Question(f"multihop-{i}", "multihop", f"What Slack channel reaches the team that owns {s['deps'][0]}, the first listed runtime dependency of {s['name']}?", tm[dep["team"]]["slack"]))
        else:
            nt = rng.choice(notes); s = by_name[nt["service"]]
            qs.append(Question(f"multihop-{i}", "multihop", f"According to {nt['id']}, a service is being migrated to a new datastore. Who leads the team that owns that service?", tm[s["team"]]["lead"]))
    # aggregation (10)
    for i in range(10):
        kind = i % 4
        if kind == 0:
            t = rng.choice(teams)
            cnt = sum(1 for inc in incidents if by_name[inc["service"]]["team"]==t["name"] and inc["sev"]=="SEV1")
            qs.append(Question(f"agg-{i}", "aggregation", f"How many SEV1 incidents affected services owned by the {t['name']} team?", cnt))
        elif kind == 1:
            s = rng.choice([x for x in services if any(inc["service"]==x["name"] for inc in incidents)])
            tot = sum(inc["minutes"] for inc in incidents if inc["service"]==s["name"])
            qs.append(Question(f"agg-{i}", "aggregation", f"What is the total duration in minutes of all incidents affecting {s['name']}?", tot))
        elif kind == 2:
            c = rng.choice(CAUSES)
            cnt = sum(1 for inc in incidents if inc["cause"]==c)
            qs.append(Question(f"agg-{i}", "aggregation", f"How many incident reports list '{c}' as the root cause?", cnt))
        else:
            from collections import Counter
            if i == 3:
                cc = Counter(inc["service"] for inc in incidents)
                top, n = cc.most_common(1)[0]
                ties = [k for k,v in cc.items() if v==n]
                qs.append(Question(f"agg-{i}", "aggregation", "Which service appears in the most incident reports?", top, aliases=ties + [by_name[t]["alias"] for t in ties]))
            else:
                sev = rng.choice(SEV); reg = rng.choice(REGIONS)
                cnt = sum(1 for inc in incidents if inc["sev"]==sev and by_name[inc["service"]]["region"]==reg)
                qs.append(Question(f"agg-{i}", "aggregation", f"How many {sev} incidents affected services deployed in {reg}?", cnt))
    # global (10): list / negation / cross-cutting
    for i in range(10):
        kind = i % 4
        if kind == 0:
            lang = rng.choice(LANGS); reg = rng.choice(REGIONS)
            lst = sorted(s["name"] for s in services if s["lang"]==lang and s["region"]==reg)
            qs.append(Question(f"global-{i}", "global", f"List every service written in {lang} that is deployed in {reg}.", lst))
        elif kind == 1:
            if i == 1:
                cnt = sum(1 for s in services if not s["oncall"])
                qs.append(Question(f"global-{i}", "global", "How many services have no designated on-call engineer?", cnt))
            else:
                db = rng.choice(DBS)
                cnt = sum(1 for s in services if s["db"]==db)
                qs.append(Question(f"global-{i}", "global", f"How many services use {db} as their primary datastore?", cnt))
        elif kind == 2:
            db = rng.choice(DBS)
            lst = sorted(nt["service"] for nt in notes if nt["newdb"]==db)
            qs.append(Question(f"global-{i}", "global", f"Which services have an approved architecture decision to migrate to {db}?", lst))
        else:
            t = rng.choice(teams)
            cnt = sum(1 for s in services if s["team"]==t["name"])
            qs.append(Question(f"global-{i}", "global", f"How many services does the {t['name']} team own?", cnt))
    # dedupe by question text (keep first), renumber
    seen=set(); out=[]
    for q in qs:
        if q.question in seen: continue
        seen.add(q.question); out.append(q)
    qs = out
    return docs, qs, dict(teams=teams, services=services, incidents=incidents, notes=notes)

_FILL = [
    "Deploys are performed through the standard pipeline and require a green build.", "Metrics are exported to the shared observability stack every fifteen seconds.",
    "Logs are retained for thirty days in the central log store.", "Feature flags are managed through the configuration service and rolled out gradually.",
    "Load tests run nightly against the staging environment.", "Alerts route through the paging system according to the escalation policy.",
    "Backups are taken daily and verified weekly by restoring to a scratch cluster.", "The runbook is reviewed quarterly by the owning team.",
    "Rate limiting is enforced at the gateway with a token bucket per client.", "Secrets are rotated automatically every ninety days.",
    "Canary releases receive five percent of traffic for thirty minutes before promotion.", "The dashboard shows p50, p95 and p99 latency alongside error budgets.",
    "Dependencies are pinned and upgraded through automated pull requests.", "Post-incident reviews are blameless and published to the engineering wiki.",
    "Container images are scanned for vulnerabilities before promotion.", "Traffic is encrypted in transit with mutual TLS between services.",
    "Capacity planning is revisited at the start of every quarter.", "Schema changes go through an expand and contract migration process.",
    "Health checks are performed on the readiness endpoint every ten seconds.", "Cost reports are reviewed monthly with the finance partner.",
]
_FILL += [
    "Engineers rotate through the support channel one week at a time.", "The staging environment mirrors production topology at one tenth scale.",
    "Configuration changes are peer reviewed and applied through the deployment tool.", "Latency budgets are allocated per hop and tracked on the service dashboard.",
    "Retries use exponential backoff with jitter to avoid thundering herds.", "Circuit breakers open after five consecutive failures to a downstream.",
    "Idempotency keys are required on all mutating endpoints.", "Requests carry a correlation id that is propagated to downstream calls.",
    "Long running jobs checkpoint progress so they can resume after a restart.", "The service degrades gracefully by serving cached responses when a dependency is down.",
    "Access to production is granted just in time and audited.", "Data retention follows the company wide policy for personal information.",
    "Synthetic probes exercise the critical user journeys every minute.", "Error budgets are reviewed in the weekly reliability meeting.",
    "Connection pools are sized according to the downstream capacity model.", "Structured logs include the request path, status code and duration.",
    "Garbage collection pauses are tracked as a leading indicator of memory pressure.", "Graceful shutdown waits for in flight requests to complete before exiting.",
    "The on-call handbook describes the escalation path and expected response times.", "Dependencies are declared in the service manifest and validated at deploy time.",
]
def _filler(rng, n):
    return " ".join(rng.sample(_FILL, min(n, len(_FILL))))

def _timeline(rng):
    t = rng.randint(0, 23); m = rng.randint(0, 59); lines=[]
    for step in ["alert fired", "on-call acknowledged", "mitigation started", "traffic rerouted", "root cause identified", "fix deployed", "monitoring confirmed recovery"][: rng.randint(4,7)]:
        m += rng.randint(2, 25); t = (t + m // 60) % 24; m %= 60
        lines.append(f"- {t:02d}:{m:02d} {step}")
    return "\n".join(lines)

def to_json(docs, qs):
    return json.dumps(dict(docs=[asdict(d) for d in docs], questions=[asdict(q) for q in qs]), indent=1)

if __name__ == "__main__":
    import sys, tiktoken
    docs, qs, _ = build()
    enc = tiktoken.get_encoding("o200k_base")
    print(len(docs), "docs,", sum(len(enc.encode(d.text)) for d in docs), "tokens,", len(qs), "questions")
    for q in qs: print(q.id, "|", q.question, "->", q.answer, q.aliases or "")
