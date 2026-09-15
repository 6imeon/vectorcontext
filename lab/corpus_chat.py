"""Long-form chat benchmark: there are NO documents; every fact arrives through the conversation itself.

A synthetic project-planning chat (user <-> assistant) of ~90 user turns (~30k tokens; the `long` variant is ~2.5x).
Facts are stated, later revised (explicitly, by coreference such as "push that back a week", relative to another
entity such as "same budget as ingest", or by swapping two values), a module is dropped and one is added, three
tool outputs are pasted (load test, CI run, config dump), and more than half of the tokens are tangents the user
asked the assistant to explain. Probe questions at the end need early turns, latest values after revisions, the
original value before a revision, counts and sums over the current state, negation, pasted numbers and abstention
(facts that were never stated). Two hand-off tasks per chat need many facts at once (agent-to-agent delegation).
Every gold answer is computed from generator state, never from the text.
"""
from __future__ import annotations
import datetime as dt, random
from collections import defaultdict
from dataclasses import dataclass, field
from corpus import _filler, FIRST

PROJECTS = ["Atlas", "Borealis", "Cinder", "Delta", "Ember", "Fjord"]
MODULES = ["ingest", "billing-sync", "notifier", "dashboard", "auth-bridge", "exporter", "scheduler-v2", "audit-trail", "rate-limiter", "search-index",
           "webhook-relay", "feature-flags", "tenant-router", "metrics-agg", "backup-runner", "session-store"]
DATASTORES = ["Postgres", "Redis", "DynamoDB", "Cassandra", "MongoDB", "CockroachDB"]
PRIORITIES = ["P0", "P1", "P2"]
REGIONS = ["us-east-1", "eu-west-1", "ap-southeast-2", "us-west-2"]
TOPICS = ["CRDTs", "backpressure in streaming systems", "consistent hashing", "the saga pattern", "write-ahead logs", "vector clocks", "idempotent consumers",
          "blue/green deploys", "connection pooling", "the outbox pattern", "leader election", "rate limiting algorithms", "cache stampedes", "schema migrations",
          "structured logging", "distributed tracing", "feature flags", "chaos testing", "tail latency", "gRPC deadlines", "optimistic locking", "event sourcing",
          "bloom filters", "circuit breakers", "hot partitions", "compaction in LSM trees", "quorum reads", "the two generals problem", "gossip protocols", "load shedding"]
ABSENT = [("What date did we set for QA sign-off?", "qa sign-off"), ("Which CI provider did we decide on?", "ci provider"), ("Who owns the rollback plan?", "rollback owner"),
          ("What is the marketing announcement date?", "marketing date"), ("What uptime SLA did we commit to for the launch?", "sla"), ("What monthly cloud cost cap did we agree on?", "cost cap")]
ABSENT_ALIASES = ["not stated", "not mentioned", "never mentioned", "not specified", "never specified", "not discussed", "never discussed", "no information", "not set",
                  "never set", "haven't decided", "have not decided", "not decided", "wasn't decided", "not been decided", "unknown", "no mention", "never came up", "didn't come up",
                  "hasn't been", "has not been", "wasn't set", "not established", "did not establish", "didn't establish", "not defined", "did not specify", "didn't specify", "wasn't specified",
                  "not agreed", "never agreed", "wasn't agreed", "no date", "no provider", "no owner", "no sla", "no cap", "nothing in the conversation", "not in the conversation", "not yet"]
GLOBAL_ALIASES = {"P0": ["p 0", "priority 0"], "P1": ["p 1"], "P2": ["p 2"]}
DS_ALIASES = {"Postgres": ["PostgreSQL", "postgresql", "postgres db"], "MongoDB": ["Mongo"], "DynamoDB": ["Dynamo"], "CockroachDB": ["Cockroach"], "Redis": [], "Cassandra": []}

@dataclass
class ChatQuestion:
    id: str
    qtype: str
    question: str
    answer: str | int | list
    aliases: list = field(default_factory=list)
    distractors: list = field(default_factory=list)   # superseded values / other entities: if present in the answer, the judge decides

@dataclass
class Task:
    """Agent-to-agent hand-off: a worker must produce a note containing every `required` fact (label, value, aliases)."""
    id: str
    task: str
    required: list
    stale: list = field(default_factory=list)  # superseded values that a correct note would not present as current

@dataclass
class Chat:
    id: str
    project: str
    turns: list           # [{i, user, assistant, kind}]
    questions: list
    tasks: list
    state: dict           # ground truth: entity -> {attr: value}
    history: dict         # (entity, attr) -> [(turn, value)]
    meta: dict

def _fmt(d): return f"{d:%b} {d.day}"
def _date_aliases(d): return [f"{d:%B} {d.day}", f"{d.month}/{d.day}", f"{d:%Y-%m-%d}", f"{d.day} {d:%b}", f"{d.day} {d:%B}", f"{d:%b} {d.day:02d}"]
def _money(k): return f"${k}k"

LOAD_TEST = "requests_total: {n}\nrps: {rps}\np50_ms: {p50}\np95_ms: {p95}\np99_ms: {p99}\nerror_rate: {err}%\nduration_s: 600"
CI_LOG = "==== test session ====\ncollected {tot} items\n{lines}\n==== {passed} passed, {failed} failed, {skipped} skipped in {secs}s ===="
CONFIG = "[service]\nmax_workers = {mw}\ndb_pool_size = {pool}\ncache_ttl_s = {ttl}\nrequest_timeout_ms = {to}\nlog_level = info\nregion = {region}"

def build_chat(seed: int = 1, n_filler: int = 68, filler_paras: int = 4, cid: str | None = None, n_modules: int = 6) -> Chat:
    """n_modules initial modules (6 -> ~70 facts; 14 -> ~120 facts, more than a ~700-word summary comfortably holds)."""
    rng = random.Random(seed * 13 + 7); trng = random.Random(seed * 29 + 1)
    cid = cid or f"chat{seed}"
    project = PROJECTS[seed % len(PROJECTS)]
    pool = MODULES[:10] if n_modules + 1 <= 10 else MODULES   # the first 10 names keep the 6-module chats identical to the original runs
    mods = rng.sample(pool, n_modules + 1); initial, added = mods[:n_modules], mods[n_modules]
    people = rng.sample(FIRST, n_modules + 3)
    base = dt.date(2026, 10, 1)
    launch = base + dt.timedelta(days=rng.randint(31, 55))
    region = rng.choice(REGIONS); repo = f"{project.lower()}-launch"
    state: dict = defaultdict(dict); hist = defaultdict(list); turns = []
    def add(user, assistant, kind):
        turns.append(dict(i=len(turns) + 1, user=user, assistant=assistant, kind=kind)); return len(turns)
    def setf(ent, attr, val, i):
        state[ent][attr] = val; hist[(ent, attr)].append((i, val))
    # ---- initial facts per module (one module gets no datastore: negation question)
    m1, m2, m3, m4, m5, m6 = initial[:6]   # roles: m1 deadline changed twice + owner swap with m5; m2 budget by coreference + datastore change; m3 relative deadline + priority; m4 relational budget; m6 dropped
    nods = m3
    owner = {m: p for m, p in zip(initial, people[:n_modules])}
    deadline = {m: base + dt.timedelta(days=rng.randint(5, 28)) for m in initial}
    budget = {m: rng.choice([20, 25, 30, 35, 40, 45, 50, 60]) for m in initial}
    store = {m: rng.choice(DATASTORES) for m in initial}
    prio = {m: rng.choice(PRIORITIES) for m in initial}
    prio[m3] = rng.choice(["P1", "P2"])
    standup_start = base + dt.timedelta(days=rng.randint(1, 6)); standup_time = rng.choice(["09:30", "10:00", "09:15", "08:45"])
    oncall = rng.sample(people[:6], 3)
    ack = lambda s: rng.choice(["Noted. ", "Got it. ", "Okay. ", "Recorded. "]) + s

    # ---- phase A blocks (each block = list of (user, assistant, kind))
    A = [[(f"We're planning the {project} launch. Target launch date is {_fmt(launch)}. Repo is {repo}, we deploy to {region}. In-scope modules: {', '.join(initial)}.",
           ack(f"{project} launches {_fmt(launch)} from {repo} in {region}, with {len(initial)} modules in scope."), "kickoff")]]
    OWN = ["Let's have {p} own {m}.", "{p} will take {m}.", "Assign {m} to {p}.", "{m} goes to {p}."]
    DL = ["{m} needs to be done by {d}.", "Deadline for {m}: {d}.", "Let's target {d} for {m}.", "{m} is due {d}."]
    BU = ["Budget for {m} is {b}.", "Give {m} a {b} budget.", "{m} gets {b}.", "Cap {m} at {b}."]
    DS = ["{m} will use {ds}.", "For {m}, store state in {ds}.", "{m} goes on {ds}.", "Let's back {m} with {ds}."]
    PR = ["{m} is {P}.", "Mark {m} as {P}.", "Treat {m} as {P}.", "{m}: {P}."]
    for m in initial:
        A.append([(rng.choice(OWN).format(p=owner[m], m=m), ack(f"{owner[m]} owns {m}."), ("set", m, "owner", owner[m]))])
        A.append([(rng.choice(DL).format(m=m, d=_fmt(deadline[m])), ack(f"{m} deadline is {_fmt(deadline[m])}."), ("set", m, "deadline", _fmt(deadline[m])))])
        A.append([(rng.choice(BU).format(m=m, b=_money(budget[m])), ack(f"{m} budget is {_money(budget[m])}."), ("set", m, "budget_k", budget[m]))])
        if m != nods: A.append([(rng.choice(DS).format(m=m, ds=store[m]), ack(f"{m} uses {store[m]}."), ("set", m, "datastore", store[m]))])
        A.append([(rng.choice(PR).format(m=m, P=prio[m]), ack(f"{m} is {prio[m]}."), ("set", m, "priority", prio[m]))])
    A.append([(f"Daily standup at {standup_time} UTC starting {_fmt(standup_start)}.", ack(f"standup {standup_time} UTC from {_fmt(standup_start)}."), ("set", "standup", "time_utc", standup_time))])
    A.append([(f"On-call order for launch week: {oncall[0]}, then {oncall[1]}, then {oncall[2]}.", ack("on-call order recorded."), ("set", "oncall", "order", oncall))])
    head, rest = A[0], A[1:]; rng.shuffle(rest); A = [head] + rest

    # ---- phase B: updates, drop/add, pastes
    d1 = deadline[m1]; d2 = d1 + dt.timedelta(days=rng.randint(5, 12)); d3 = d2 + dt.timedelta(days=rng.randint(3, 9))
    b2_new = budget[m2] + 15
    d3m3 = deadline[m3] - dt.timedelta(days=3)
    store_m2_new = rng.choice([s for s in DATASTORES if s != store[m2]])
    pt = dict(n=rng.randint(80, 200) * 1000, rps=rng.randint(400, 1200), p50=rng.randint(20, 60), p95=rng.randint(180, 620), p99=rng.randint(700, 1900), err=rng.choice(["0.4", "0.7", "1.2", "2.5"]))
    ci = dict(tot=rng.randint(180, 320), failed=rng.randint(2, 9), skipped=rng.randint(0, 6), secs=rng.randint(40, 300))
    ci["passed"] = ci["tot"] - ci["failed"] - ci["skipped"]
    ci["lines"] = "\n".join(f"tests/test_{rng.choice(mods).replace('-', '_')}.py::test_{rng.choice(['create', 'update', 'retry', 'timeout', 'auth', 'schema'])}_{k} FAILED" for k in range(ci["failed"]))
    cfg = dict(mw=rng.choice([8, 16, 24, 32]), pool=rng.choice([12, 20, 32, 48, 64]), ttl=rng.choice([300, 600, 900]), to=rng.choice([2000, 5000, 8000]), region=region)
    added_owner, added_dl, added_b, added_ds, added_p = people[n_modules], base + dt.timedelta(days=rng.randint(20, 40)), rng.choice([15, 20, 25]), rng.choice(DATASTORES), rng.choice(PRIORITIES)
    B = {
     "u1": [(f"Change of plan: {m1}'s deadline moves to {_fmt(d2)}.", ack(f"{m1} is now due {_fmt(d2)}."), ("set", m1, "deadline", _fmt(d2)))],
     "u8": [(f"One more change on {m1}: it's now due {_fmt(d3)}.", ack(f"{m1} due {_fmt(d3)}."), ("set", m1, "deadline", _fmt(d3)))],
     "u2": [(f"Let me revisit the {m2} budget: currently {_money(budget[m2])}, right?", "Yes, " + _money(budget[m2]) + " is what we have recorded.", "filler"),
            (f"Actually, make that {_money(b2_new)}.", ack(f"{m2} budget is now {_money(b2_new)}."), ("set", m2, "budget_k", b2_new))],
     "u3": [(f"Going back to the {m3} deadline: pull it in by three days.", ack(f"{m3} is now due {_fmt(d3m3)}."), ("set", m3, "deadline", _fmt(d3m3)))],
     "u4": [(f"Give {m4} the same budget as {m1}.", ack(f"{m4} budget is now {_money(budget[m1])}."), ("set", m4, "budget_k", budget[m1]))],
     "u5": [(f"Swap the owners of {m1} and {m5}.", ack(f"{owner[m5]} owns {m1} and {owner[m1]} owns {m5}."), ("swap", m1, m5)),],
     "u6": [(f"We're moving {m2} off {store[m2]} to {store_m2_new}.", ack(f"{m2} uses {store_m2_new}."), ("set", m2, "datastore", store_m2_new))],
     "u7": [(f"Bump {m3} to P0.", ack(f"{m3} is P0."), ("set", m3, "priority", "P0"))],
     "drop": [(f"Let's drop {m6} from this launch; it moves to next quarter.", ack(f"{m6} is out of scope for this launch."), ("set", m6, "in_scope", False))],
     "add": [(f"New module for the launch: {added}. {added_owner} owns it, it's due {_fmt(added_dl)}, budget {_money(added_b)}, backed by {added_ds}, priority {added_p}.",
              ack(f"{added} added with those details."), ("add", added, dict(owner=added_owner, deadline=_fmt(added_dl), budget_k=added_b, datastore=added_ds, priority=added_p)))],
     "p1": [("Here's the load test output from staging:\n```\n" + LOAD_TEST.format(**pt) + "\n```", "Thanks. Anything you want me to flag from these numbers?", ("paste", "load_test", dict(p95_ms=pt["p95"], error_rate_pct=float(pt["err"]), rps=pt["rps"])))],
     "p2": [("CI run from this morning:\n```\n" + CI_LOG.format(**ci) + "\n```", "Thanks, I have the CI output.", ("paste", "ci_run", dict(failed=ci["failed"], passed=ci["passed"])))],
     "p3": [("Current service config:\n```\n" + CONFIG.format(**cfg) + "\n```", "Got the config.", ("paste", "config", dict(db_pool_size=cfg["pool"], max_workers=cfg["mw"])))],
    }
    order = list(B); rng.shuffle(order)
    if order.index("u8") < order.index("u1"): a, b = order.index("u8"), order.index("u1"); order[a], order[b] = order[b], order[a]
    Bl = [B[k] for k in order]

    # ---- interleave with filler tangents
    def filler_block():
        t = trng.choice(TOPICS)
        q = trng.choice([f"Quick tangent: can you explain {t}?", f"Unrelated, but remind me how {t} work(s)?", f"Side question on {t}: what should I know?", f"Give me a refresher on {t}."])
        a = f"Sure, here is a short overview of {t}.\n\n" + "\n\n".join(_filler(trng, 8) for _ in range(filler_paras))
        return [(q, a, "filler")]
    blocks = A + Bl
    nA, nB = len(A), len(Bl)
    fa = int(n_filler * nA / (nA + nB)); fb = n_filler - fa
    def weave(bl, nf):
        slots = sorted(rng.choices(range(1, len(bl) + 1), k=nf))
        out = []
        for j, b in enumerate(bl):
            out.append(b); out += [filler_block() for _ in range(slots.count(j + 1))]
        return out
    seq = weave([A[0]], 0) + weave(A[1:], fa) + weave(Bl, fb)
    for blk in seq:
        for user, assistant, kind in blk:
            i = add(user, assistant, kind if isinstance(kind, str) else kind[0])
            if isinstance(kind, tuple):
                if kind[0] == "set": setf(kind[1], kind[2], kind[3], i)
                elif kind[0] == "swap":
                    a, b = kind[1], kind[2]; oa, ob = state[a]["owner"], state[b]["owner"]; setf(a, "owner", ob, i); setf(b, "owner", oa, i)
                elif kind[0] in ("add", "paste"):
                    for k, v in kind[2].items(): setf(kind[1], k, v, i)
    for m in initial + [added]: state[m].setdefault("in_scope", True)
    setf("project", "launch_date", _fmt(launch), 1); setf("project", "region", region, 1); setf("project", "repo", repo, 1)
    in_scope = [m for m in initial + [added] if state[m]["in_scope"]]
    cur = lambda m, a: state[m].get(a)
    others = lambda ms: [x for x in mods if x not in ms]
    dA = lambda d: _date_aliases(d)
    stable_owner = [m for m in initial if m not in (m1, m5, m6)][0]
    qs = [
        ChatQuestion(f"{cid}/early-0", "early", f"Who owns {stable_owner}?", owner[stable_owner], [], [p for p in people if p != owner[stable_owner]]),
        ChatQuestion(f"{cid}/early-1", "early", f"Which region are we deploying {project} to?", region, [], [r for r in REGIONS if r != region]),
        ChatQuestion(f"{cid}/early-2", "early", "At what time (UTC) is the daily standup?", standup_time, [standup_time.replace(":", "")], []),
        ChatQuestion(f"{cid}/updated-0", "updated", f"What is the current deadline for {m1}?", _fmt(d3), dA(d3), [_fmt(d1), _fmt(d2)] + dA(d1) + dA(d2)),
        ChatQuestion(f"{cid}/updated-1", "updated", f"What is {m2}'s budget now, in thousands of dollars?", b2_new, [], [budget[m2]]),
        ChatQuestion(f"{cid}/updated-2", "updated", f"What is the current deadline for {m3}?", _fmt(d3m3), dA(d3m3), [_fmt(deadline[m3])] + dA(deadline[m3])),
        ChatQuestion(f"{cid}/updated-3", "updated", f"What is {m4}'s budget, in thousands of dollars?", budget[m1], [], [budget[m4]] if budget[m4] != budget[m1] else []),
        ChatQuestion(f"{cid}/updated-4", "updated", f"Who owns {m1} now?", owner[m5], [], [owner[m1]]),
        ChatQuestion(f"{cid}/updated-5", "updated", f"Which datastore does {m2} use?", store_m2_new, DS_ALIASES.get(store_m2_new, []), [store[m2]]),
        ChatQuestion(f"{cid}/original-0", "original", f"What was the very first deadline we set for {m1}, before any changes?", _fmt(d1), dA(d1), [_fmt(d2), _fmt(d3)] + dA(d2) + dA(d3)),
        ChatQuestion(f"{cid}/count-0", "count", "How many modules are currently in scope for the launch?", len(in_scope), [], []),
        ChatQuestion(f"{cid}/sum-0", "sum", "What is the total budget, in thousands of dollars, across the modules currently in scope?", sum(cur(m, "budget_k") for m in in_scope), [], []),
        ChatQuestion(f"{cid}/neg-0", "negation", "Which in-scope module has no datastore decided yet?", nods, [], others([nods])),
        ChatQuestion(f"{cid}/list-0", "list", "Which in-scope modules are P0?", [m for m in in_scope if cur(m, "priority") == "P0"], [], others([m for m in in_scope if cur(m, "priority") == "P0"])),
        ChatQuestion(f"{cid}/paste-0", "paste", "What p95 latency, in ms, did the load test I pasted report?", pt["p95"], [], []),
        ChatQuestion(f"{cid}/paste-1", "paste", "How many tests failed in the CI output I pasted?", ci["failed"], [], []),
        ChatQuestion(f"{cid}/paste-2", "paste", "What is db_pool_size in the config I pasted?", cfg["pool"], [], []),
    ]
    for k, (qt, _) in enumerate(rng.sample(ABSENT, 2)):
        qs.append(ChatQuestion(f"{cid}/absent-{k}", "absent", qt, "not stated", ABSENT_ALIASES, []))
    # ---- hand-off tasks
    req1 = [("launch_date", _fmt(launch), dA(launch))]
    stale1 = []
    for m in in_scope:
        req1 += [(f"{m}.owner", cur(m, "owner"), []), (f"{m}.deadline", cur(m, "deadline"), dA(dt.datetime.strptime(cur(m, "deadline") + " 2026", "%b %d %Y").date())), (f"{m}.budget_k", cur(m, "budget_k"), [])]
        if cur(m, "datastore"): req1.append((f"{m}.datastore", cur(m, "datastore"), DS_ALIASES.get(cur(m, "datastore"), [])))
        for (e, a), h in hist.items():
            if e == m and len(h) > 1 and a in ("deadline", "budget_k", "owner", "datastore"):
                for _, v in h[:-1]:
                    if v != h[-1][1] and (a != "owner" or v not in [cur(x, "owner") for x in in_scope]): stale1.append((f"{m}.{a}", v))
    req2 = [("load_test.p95_ms", pt["p95"], []), ("load_test.error_rate_pct", pt["err"], [f"{pt['err']} %"]), ("ci.failed", ci["failed"], []), ("config.db_pool_size", cfg["pool"], []),
            ("oncall.order", " then ".join(oncall), [", ".join(oncall), " -> ".join(oncall), " → ".join(oncall), " > ".join(oncall)])] + [(f"p0.{m}", m, []) for m in in_scope if cur(m, "priority") == "P0"]
    tasks = [Task(f"{cid}/task-0", f"Write the launch hand-off note for the {project} launch: the launch date, and for every module currently in scope its owner, deadline, budget (in $k) and datastore (write 'undecided' if none was chosen). Modules dropped from the launch must not be listed as in scope.", req1, stale1),
             Task(f"{cid}/task-1", "Write a launch readiness summary: the load-test p95 latency (ms) and error rate, the number of failing tests in the CI run, the db_pool_size from the service config, the list of P0 modules currently in scope, and the on-call order for launch week (first, second, third).", req2, [])]
    meta = dict(seed=seed, n_turns=len(turns), n_filler=n_filler, launch=_fmt(launch), modules=initial + [added], dropped=m6, roles=dict(m1=m1, m2=m2, m3=m3, m4=m4, m5=m5, m6=m6, added=added))
    return Chat(cid, project, turns, qs, tasks, {k: dict(v) for k, v in state.items()}, {f"{e}.{a}": h for (e, a), h in hist.items()}, meta)

def build_chats(seeds=(1, 2, 3), long_seed=4, dense_seed=5):
    """Three ~29k-token chats, one ~72k-token chat (2x filler turns, 5 paragraphs per tangent) and one fact-dense chat (14 modules, ~120 facts)."""
    chats = [build_chat(s) for s in seeds]
    if long_seed: chats.append(build_chat(long_seed, n_filler=140, filler_paras=5, cid=f"long{long_seed}"))
    if dense_seed: chats.append(build_chat(dense_seed, n_modules=14, cid=f"dense{dense_seed}"))
    return chats

if __name__ == "__main__":
    import tiktoken, json
    enc = tiktoken.get_encoding("o200k_base")
    for c in build_chats():
        toks = sum(len(enc.encode(t["user"] + t["assistant"])) for t in c.turns)
        kinds = defaultdict(int)
        for t in c.turns: kinds[t["kind"]] += 1
        print(c.id, c.project, "turns", len(c.turns), "tokens", toks, dict(kinds))
        for q in c.questions: print("  ", q.id, q.question, "->", q.answer, ("distractors " + str(q.distractors[:3])) if q.distractors else "")
        for t in c.tasks: print("  ", t.id, len(t.required), "required;", t.stale)
