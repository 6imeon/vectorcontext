"""Hard benchmark on top of corpus.build():

* change records (doc type `changelog`) that supersede facts stated in service overviews, some of which are
  themselves superseded by a later record  -> temporal reasoning / latest-wins
* collateral-impact sentences in incident reports naming a *second* service      -> distractors for extraction & search
* questions on prose-only facts outside any fact schema (timeline times, weekdays) -> forces raw-document access
* negation / set-difference / reverse-join questions                              -> forces exhaustive enumeration
* numeric reasoning (averages, maxima across joins)                               -> forces code
* multi-turn conversations whose follow-ups use pronouns and depend on earlier turns

Every gold answer is computed from the generator state, never from the text.
"""
from __future__ import annotations
import random, re
from collections import Counter
from dataclasses import dataclass, field
from corpus import build, Doc, Question, _filler, _name, TEAMS, LANGS, REGIONS, DBS, SEV, CAUSES

@dataclass
class Conversation:
    id: str
    turns: list  # list[Question]
    qtype: str = "multiturn"

def _date_key(d): return (d["month"], d["day"])

def build_hard(seed: int = 7, filler_k: int = 3, n_changes: int = 16, side_effect_frac: float = 0.4, mult: int = 1):
    """mult scales the number of documents (services, incidents, decisions, runbooks) instead of the prose per document."""
    docs, base_qs, meta = build(seed=seed, filler_k=filler_k, n_services=40 * mult, n_incidents=60 * mult, n_notes=30 * mult, n_filler=30 * mult)
    rng = random.Random(seed * 101 + 5); trng = random.Random(seed * 77 + 3)
    services, incidents, teams, notes = meta["services"], meta["incidents"], meta["teams"], meta["notes"]
    by_name = {s["name"]: s for s in services}; tm = {t["name"]: t for t in teams}
    docmap = {d.id: d for d in docs}

    # ------------------------------------------------------------------ collateral-impact mentions (distractors)
    for inc in incidents:
        inc["side_effect"] = None
        if rng.random() < side_effect_frac:
            other = rng.choice([s for s in services if s["name"] != inc["service"]])
            inc["side_effect"] = other["name"]
            ref = other["name"] if trng.random() < 0.5 else other["alias"]
            sent = trng.choice([
                f"As a side effect, {ref} also saw degraded latency for part of the window.",
                f"Downstream, {ref} experienced elevated error rates while the incident was ongoing, but was not the source.",
                f"Collateral impact was observed on {ref}, which recovered on its own once the primary issue was mitigated."])
            d = docmap[f"inc-{inc['id']}"]
            d.text = d.text.replace("This report was written by", sent + " This report was written by", 1)

    # ------------------------------------------------------------------ change records (supersession)
    kinds = ["team"] * 5 + ["oncall"] * 4 + ["region"] * 3
    rng.shuffle(kinds)
    chg_services = rng.sample(services, 12)
    state = {s["name"]: dict(team=s["team"], oncall=s["oncall"], region=s["region"]) for s in services}
    changes = []
    used_dates = set()
    for i in range(n_changes):
        s = chg_services[i % 12]; kind = kinds[i % 12]
        prev = next((c for c in changes if c["service"] == s["name"] and c["field"] == kind), None)
        while True:
            month, day = rng.randint(1, 12), rng.randint(1, 28)
            if (month, day) not in used_dates and (not prev or (month, day) > (prev["month"], prev["day"])): break
        used_dates.add((month, day))
        old = state[s["name"]][kind]
        orig = by_name[s["name"]][kind]  # a second change must not revert to the overview's value, or the overview alone would be right
        if kind == "team": new = rng.choice([t["name"] for t in teams if t["name"] != old and t["name"] != orig])
        elif kind == "region": new = rng.choice([r for r in REGIONS if r != old and r != orig])
        else: new = _name(rng) if (old is None or rng.random() > 0.25) else None
        state[s["name"]][kind] = new
        changes.append(dict(id=f"CHG-{300 + i * 3 + rng.randint(0, 2)}", service=s["name"], field=kind, old=old, new=new, month=month, day=day, year=2025,
                            supersedes=prev["id"] if prev else None))
    for c in changes:
        s = by_name[c["service"]]; ref = s["name"] if trng.random() < 0.5 else s["alias"]; date = f"2025-{c['month']:02d}-{c['day']:02d}"
        if c["field"] == "team":
            body = trng.choice([f"Effective {date}, ownership of {ref} transferred from the {c['old']} team to the {c['new']} team.",
                                f"{ref} now belongs to the {c['new']} team (previously {c['old']}) as of {date}.",
                                f"On {date} the {c['new']} team took over {ref} from the {c['old']} team."])
        elif c["field"] == "region":
            body = trng.choice([f"On {date}, {ref} was migrated from {c['old']} to {c['new']}.",
                                f"Effective {date}, the deployment of {ref} moved out of {c['old']} and now runs in {c['new']}."])
        else:
            if c["new"] is None: body = f"As of {date}, {c['old']} no longer carries the pager for {ref}; the on-call rotation is unassigned until further notice."
            elif c["old"] is None: body = f"Starting {date}, {c['new']} becomes the designated on-call engineer for {ref}, which previously had no on-call."
            else: body = trng.choice([f"Starting {date}, {c['new']} takes over primary on-call for {ref} from {c['old']}.",
                                      f"Effective {date}, pages for {ref} go to {c['new']} instead of {c['old']}."])
        sup = f" This record supersedes {c['supersedes']}." if c["supersedes"] else ""
        text = (f"# Change record {c['id']}\n\nDate: {date}\n\n{body}{sup} The service overview page has not been updated to reflect this change. "
                + _filler(trng, 2 * filler_k))
        docs.append(Doc(f"chg-{c['id']}", "changelog", f"Change record {c['id']}", text))
    trng.shuffle(docs)

    def team_at(svc, month, day):
        t = by_name[svc]["team"]
        for c in sorted((c for c in changes if c["service"] == svc and c["field"] == "team"), key=_date_key):
            if (c["month"], c["day"]) <= (month, day): t = c["new"]
        return t
    cur = state  # current (latest-wins) state

    # ------------------------------------------------------------------ helpers over generator state
    primary_cnt = Counter(inc["service"] for inc in incidents)
    side_cnt = Counter(inc["side_effect"] for inc in incidents if inc["side_effect"])
    dependents = Counter(d for s in services for d in s["deps"])
    def timeline(inc_id):
        txt = docmap[f"inc-{inc_id}"].text
        return [(m.group(2), int(m.group(1)[:2]) * 60 + int(m.group(1)[3:])) for m in re.finditer(r"- (\d\d:\d\d) (.+)", txt)]

    qs: list[Question] = []
    # ---- temporal (latest-wins, history, as-of-date joins)
    team_chg = [c for c in changes if c["field"] == "team"]
    twice = [s for s in sorted(set(c["service"] for c in team_chg)) if sum(1 for c in team_chg if c["service"] == s) > 1]
    once = [s for s in sorted(set(c["service"] for c in team_chg)) if s not in twice]
    control = rng.choice([s["name"] for s in services if s["name"] not in state or all(c["service"] != s["name"] for c in changes)])
    for i, svc in enumerate([once[0], (twice or once)[-1 if not twice else 0], control]):
        s = by_name[svc]; ref = s["alias"] if i == 1 else s["name"]
        qs.append(Question(f"temporal-{i}", "temporal", f"As of today, which team owns {ref}? Take any change records into account.", cur[svc]["team"],
                           notes="control (no change)" if svc == control else ("two changes" if svc in twice else "one change")))
    oc_chg = [c for c in changes if c["field"] == "oncall"]
    c = next(c for c in oc_chg if c["new"])
    qs.append(Question("temporal-3", "temporal", f"Who is currently on call for {by_name[c['service']]['name']}?", cur[c["service"]]["oncall"]))
    c = next(c for c in oc_chg if c["old"])
    qs.append(Question("temporal-4", "temporal", f"Who carried the pager for {by_name[c['service']]['alias']} before {c['id']} took effect?", c["old"]))
    t = by_name[team_chg[0]["service"]]["team"] if team_chg[0]["new"] else teams[0]["name"]
    t = team_chg[0]["new"]
    qs.append(Question("temporal-5", "temporal", f"How many services does the {t} team own today, after all change records are applied?",
                       sum(1 for s in services if cur[s["name"]]["team"] == t)))
    # as-of-date joins: incidents on services whose team changed, one before and one after the change
    cand = [(inc, c) for inc in incidents for c in team_chg if inc["service"] == c["service"] and not c["supersedes"]]
    before = [x for x in cand if _date_key(x[0]) < _date_key(x[1])]
    after = [x for x in cand if _date_key(x[0]) > _date_key(x[1]) and team_at(x[0]["service"], x[0]["month"], x[0]["day"]) != by_name[x[0]["service"]]["team"]]
    for j, pool in enumerate([before, after]):
        if pool:
            inc, c = rng.choice(pool)
            qs.append(Question(f"temporal-{6+j}", "temporal", f"Which team owned the service affected by {inc['id']} on the date of that incident?",
                               team_at(inc["service"], inc["month"], inc["day"]), notes="incident before change" if j == 0 else "incident after change"))
    c = next(c for c in changes if c["field"] == "region")
    qs.append(Question("temporal-8", "temporal", f"Which region is {by_name[c['service']]['name']} deployed in now?", cur[c["service"]]["region"]))

    # ---- prose-only facts (outside any schema)
    with_fix = [inc for inc in incidents if any(st == "fix deployed" for st, _ in timeline(inc["id"]))]
    for i in range(2):
        inc = rng.choice(with_fix); tl = timeline(inc["id"]); step = rng.choice(["mitigation started", "fix deployed", "on-call acknowledged"])
        tmin = next(t for st, t in tl if st == step)
        qs.append(Question(f"prose-{i}", "prose", f"In {inc['id']}, at what time (HH:MM) does the timeline record '{step}'?", f"{tmin//60:02d}:{tmin%60:02d}"))
    inc = rng.choice(with_fix); tl = timeline(inc["id"])
    t0 = next(t for st, t in tl if st == "alert fired"); t1 = next(t for st, t in tl if st == "fix deployed")
    qs.append(Question("prose-2", "prose", f"In {inc['id']}, how many minutes passed between the alert firing and the fix being deployed?", (t1 - t0) % 1440))
    t = rng.choice(teams); wd = re.search(r"happens on (\w+) mornings", docmap[f"team-{t['name'].lower()}"].text).group(1)
    qs.append(Question("prose-3", "prose", f"On which weekday does the {t['name']} team hold its weekly planning?", wd))
    inc = rng.choice(incidents); imp = re.search(r"Summary: (.+?) affecting", docmap[f"inc-{inc['id']}"].text).group(1)
    qs.append(Question("prose-4", "prose", f"What impact did {inc['id']} report: elevated error rates, latency spike, partial outage, or complete outage?", imp.lower()))
    inc = rng.choice(incidents); tl = timeline(inc["id"])
    qs.append(Question("prose-5", "prose", f"How many timeline entries does the report for {inc['id']} contain?", len(tl)))

    # ---- negation / set difference / reverse join
    # NOTE: at mult=1 every pick below is the original rng.choice over a filtered candidate list (this is the question set all mult=1 runs used);
    # at mult>1 the candidate lists can be empty, so a deterministic nearest-size pick is used instead (this is the set all wide-corpus runs used).
    tcands = [t for t in teams if 1 <= sum(1 for s in services if s["team"] == t["name"] and primary_cnt[s["name"]] == 0) <= 3]
    t = rng.choice(tcands) if (mult == 1 or tcands) else min(teams, key=lambda t: abs(3 - sum(1 for s in services if s["team"] == t["name"] and primary_cnt[s["name"]] == 0)))
    qs.append(Question("neg-0", "negation", f"Which services owned by the {t['name']} team (per the service overviews) have never been the primary affected service in any incident report?",
                       sorted(s["name"] for s in services if s["team"] == t["name"] and primary_cnt[s["name"]] == 0)))
    lang = rng.choice([l for l in LANGS if 1 <= sum(1 for t in teams if not any(s["team"] == t["name"] and s["lang"] == l for s in services)) <= 4] or LANGS)
    qs.append(Question("neg-1", "negation", f"Per the service overviews, which teams own no services written in {lang}?", sorted(t["name"] for t in teams if not any(s["team"] == t["name"] and s["lang"] == lang for s in services))))
    qs.append(Question("neg-2", "negation", "How many services are not a runtime dependency of any other service?", sum(1 for s in services if dependents[s["name"]] == 0)))
    db = rng.choice([d for d in DBS if 1 <= sum(1 for s in services if s["db"] == d and primary_cnt[s["name"]] == 0) <= 4]) if mult == 1 else min(DBS, key=lambda d: abs(3 - sum(1 for s in services if s["db"] == d and primary_cnt[s["name"]] == 0)))
    qs.append(Question("neg-3", "negation", f"Which services use {db} as their primary datastore but have never appeared as the primary affected service in an incident report?",
                       sorted(s["name"] for s in services if s["db"] == db and primary_cnt[s["name"]] == 0)))
    adr_svcs = {n["service"] for n in notes}
    qs.append(Question("neg-4", "negation", "How many services have an approved architecture decision to migrate datastore but no incident report as the primary affected service?",
                       sum(1 for s in services if s["name"] in adr_svcs and primary_cnt[s["name"]] == 0)))
    reg = rng.choice([r for r in REGIONS if 1 <= sum(1 for s in services if s["region"] == r and not s["oncall"]) <= 4]) if mult == 1 else min(REGIONS, key=lambda r: abs(2 - sum(1 for s in services if s["region"] == r and not s["oncall"])))
    qs.append(Question("neg-5", "negation", f"Which services deployed in {reg} have no designated on-call engineer according to their service overview?",
                       sorted(s["name"] for s in services if s["region"] == reg and not s["oncall"])))

    # ---- numeric reasoning
    sev = rng.choice(SEV); durs = [inc["minutes"] for inc in incidents if inc["sev"] == sev]
    qs.append(Question("num-0", "numeric", f"What is the average duration in minutes of {sev} incidents, rounded to the nearest whole minute?", round(sum(durs) / len(durs))))
    reps = Counter(); [reps.update({s["region"]: s["replicas"]}) for s in services]
    top = reps.most_common(1)[0][0]
    qs.append(Question("num-1", "numeric", "Which region has the largest total number of replicas across the services deployed there (per the service overviews)?", top))
    per_team = Counter(by_name[inc["service"]]["team"] for inc in incidents)
    a, b = sorted(teams, key=lambda t: -per_team[t["name"]])[0], sorted(teams, key=lambda t: -per_team[t["name"]])[-2]
    qs.append(Question("num-2", "numeric", f"Per the service overviews, how many more incident reports affected services owned by the {a['name']} team than services owned by the {b['name']} team?", per_team[a["name"]] - per_team[b["name"]]))
    topdep, n = dependents.most_common(1)[0]; ties = [k for k, v in dependents.items() if v == n]
    qs.append(Question("num-3", "numeric", "Which service is listed as a runtime dependency by the largest number of other services?", topdep, aliases=ties + [by_name[t]["alias"] for t in ties]))
    db = rng.choice([d for d in DBS if any(by_name[inc["service"]]["db"] == d for inc in incidents)])
    qs.append(Question("num-4", "numeric", f"What is the duration in minutes of the longest single incident affecting a service whose primary datastore is {db}?",
                       max(inc["minutes"] for inc in incidents if by_name[inc["service"]]["db"] == db)))
    t = rng.choice(teams)
    qs.append(Question("num-5", "numeric", f"What is the combined replica count of all services owned by the {t['name']} team (per the service overviews)?", sum(s["replicas"] for s in services if s["team"] == t["name"])))

    # ---- distractors: collateral mentions vs primary
    trap = [s for s in services if side_cnt[s["name"]] >= 1]
    for i, s in enumerate(rng.sample(trap, 2)):
        ref = s["name"] if i == 0 else s["alias"]
        qs.append(Question(f"distract-{i}", "distractor", f"How many incident reports have {ref} as the primary affected service?", primary_cnt[s["name"]]))
    s = rng.choice([s for s in trap if primary_cnt[s["name"]] >= 1])
    qs.append(Question("distract-2", "distractor", f"In how many incident reports is {s['name']} mentioned only as collateral impact rather than as the primary affected service?", side_cnt[s["name"]]))
    inc = rng.choice([inc for inc in incidents if inc["side_effect"]])
    qs.append(Question("distract-3", "distractor", f"Which service was the primary affected service in {inc['id']}?", inc["service"], aliases=[by_name[inc["service"]]["alias"]]))
    s = rng.choice([s for s in trap if primary_cnt[s["name"]] >= 1])
    qs.append(Question("distract-4", "distractor", f"What is the total duration in minutes of incidents where {s['alias']} was the primary affected service (exclude reports where it was only collateral)?",
                       sum(inc["minutes"] for inc in incidents if inc["service"] == s["name"])))
    cause = rng.choice([c for c in CAUSES if 1 <= sum(1 for inc in incidents if inc["cause"] == c and inc["side_effect"]) <= 3]) if mult == 1 else min(CAUSES, key=lambda c: abs(2 - sum(1 for inc in incidents if inc["cause"] == c and inc["side_effect"])))
    qs.append(Question("distract-5", "distractor", f"Which services were named as collateral impact in incident reports whose root cause was '{cause}'?",
                       sorted({inc["side_effect"] for inc in incidents if inc["cause"] == cause and inc["side_effect"]})))

    # ---- multi-turn conversations
    convos: list[Conversation] = []
    def Q(cid, k, text, ans, aliases=None): return Question(f"{cid}/t{k}", "multiturn", text, ans, aliases or [])
    s = rng.choice([s for s in services if all(c["service"] != s["name"] for c in changes)])
    sev1 = sum(1 for inc in incidents if by_name[inc["service"]]["team"] == s["team"] and inc["sev"] == "SEV1")
    convos.append(Conversation("conv-0", [Q("conv-0", 1, f"What datastore does {s['name']} use?", s["db"]), Q("conv-0", 2, "Which team owns it?", s["team"]),
                                         Q("conv-0", 3, "Who leads that team?", tm[s["team"]]["lead"]), Q("conv-0", 4, "How many SEV1 incidents have affected services owned by that team (per the service overviews)?", sev1)]))
    inc = rng.choice([inc for inc in incidents if sum(1 for x in incidents if x["cause"] == inc["cause"]) >= 3])
    same = [x for x in incidents if x["cause"] == inc["cause"] and x is not inc]; longest = max(same, key=lambda x: x["minutes"])
    convos.append(Conversation("conv-1", [Q("conv-1", 1, f"Which service was primarily affected by {inc['id']}?", inc["service"], [by_name[inc["service"]]["alias"]]),
                                         Q("conv-1", 2, "What was its root cause?", inc["cause"]),
                                         Q("conv-1", 3, "How many other incident reports list the same root cause?", len(same)),
                                         Q("conv-1", 4, "Of those other reports, which incident lasted the longest?", longest["id"], [x["id"] for x in same if x["minutes"] == longest["minutes"]])]))
    reg = rng.choice(REGIONS); in_reg = [s for s in services if s["region"] == reg]
    lang = Counter(s["lang"] for s in in_reg).most_common(1)[0][0]; sub = [s for s in in_reg if s["lang"] == lang]; most = max(sub, key=lambda s: s["replicas"])
    convos.append(Conversation("conv-2", [Q("conv-2", 1, f"How many services are deployed in {reg} according to their service overviews?", len(in_reg)), Q("conv-2", 2, f"How many of those are written in {lang}?", len(sub)),
                                         Q("conv-2", 3, "List them.", sorted(s["name"] for s in sub)), Q("conv-2", 4, "Which of them has the most replicas?", most["name"], [s["name"] for s in sub if s["replicas"] == most["replicas"]] + [most["alias"]])]))
    c = [c for c in team_chg if c["service"] not in twice][0]; s = by_name[c["service"]]
    convos.append(Conversation("conv-3", [Q("conv-3", 1, f"Which team is listed as the owner on the service overview for {s['name']}?", s["team"]),
                                         Q("conv-3", 2, "Has that ownership changed according to any change record? Answer yes or no.", "yes"),
                                         Q("conv-3", 3, "Which team owns it now?", cur[s["name"]]["team"]), Q("conv-3", 4, "What Slack channel reaches that team?", tm[cur[s["name"]]["team"]]["slack"])]))
    s = rng.choice(services); dep = by_name[s["deps"][0]]
    convos.append(Conversation("conv-4", [Q("conv-4", 1, f"What port does {s['alias']} listen on?", s["port"]), Q("conv-4", 2, "What are its runtime dependencies?", sorted(s["deps"])),
                                         Q("conv-4", 3, "Which team owns the first listed one, per its service overview?", dep["team"]),
                                         Q("conv-4", 4, "Who is the on-call engineer for that dependency according to its service overview?", dep["oncall"] or "none", ["no designated on-call", "unassigned", "no on-call"])]))
    n = primary_cnt.most_common(1)[0][1]; ties = sorted(k for k, v in primary_cnt.items() if v == n); topsvc = ties[0]; s = by_name[topsvc]
    convos.append(Conversation("conv-5", [Q("conv-5", 1, "Which service appears as the primary affected service in the most incident reports? If several are tied, pick the one whose canonical name comes first alphabetically.", topsvc, [s["alias"]]),
                                         Q("conv-5", 2, "What is its total incident duration in minutes?", sum(inc["minutes"] for inc in incidents if inc["service"] == topsvc)),
                                         Q("conv-5", 3, "Which region is it deployed in according to its service overview?", s["region"]),
                                         Q("conv-5", 4, "And which team owns it according to that overview?", s["team"])]))
    meta = dict(meta, changes=changes, current=cur)
    return docs, qs, convos, meta

if __name__ == "__main__":
    import sys, tiktoken
    docs, qs, convos, meta = build_hard(filler_k=int(sys.argv[1]) if len(sys.argv) > 1 else 3, mult=int(sys.argv[2]) if len(sys.argv) > 2 else 1)
    enc = tiktoken.get_encoding("o200k_base")
    print(len(docs), "docs,", sum(len(enc.encode(d.text)) for d in docs), "tokens,", len(qs), "questions,", len(convos), "conversations")
    for q in qs: print(q.id, "|", q.question, "->", q.answer, q.aliases or "", q.notes or "")
    for cv in convos:
        print(cv.id)
        for q in cv.turns: print("   ", q.id, "|", q.question, "->", q.answer, q.aliases or "")
    print("changes:"); [print("  ", c) for c in meta["changes"]]
