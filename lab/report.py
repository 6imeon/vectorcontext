"""Build the Excel workbook from results/*.jsonl."""
from __future__ import annotations
import glob, json, os, sys, statistics as st
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter

ROOT = os.path.join(os.path.dirname(__file__), "..")
def load(patterns):
    recs=[]
    for pat in patterns:
        for f in glob.glob(os.path.join(ROOT, "results", pat)):
            if "pilot" in f: continue
            for line in open(f):
                if line.strip(): recs.append(json.loads(line))
    # keep the last record per (condition, scale, tag, qid)
    last={}
    for r in recs:
        r.setdefault("corpus", "base"); r.setdefault("rep", 0); r.setdefault("model", "openai/gpt-5.6-luna")
        last[(r["corpus"], r["condition"], r["scale"], r.get("tag",""), r["model"], r["rep"], r["qid"])] = r
    return list(last.values())

def label(r):
    return r["condition"] + (f" [{r['tag']}]" if r.get("tag") else "") + (f" @{r['model'].split('/')[-1]}" if r.get("model","openai/gpt-5.6-luna")!="openai/gpt-5.6-luna" else "")
QTYPES = ["needle","multihop","aggregation","global","temporal","prose","negation","numeric","distractor","multiturn"]

def summarize(recs):
    groups=defaultdict(list)
    for r in recs: groups[(r["corpus"], label(r), r["scale"])].append(r)
    rows=[]
    for (corp, cond, scale), rs in sorted(groups.items(), key=lambda x:(x[0][0], x[0][2], x[0][1])):
        n=len(rs); acc=sum(r["correct"] for r in rs)/n
        by_t={t: [r for r in rs if r["qtype"]==t] for t in QTYPES}
        row=dict(corpus=corp, condition=cond, scale=scale, n=n, accuracy=acc,
            **{f"acc_{t}": (sum(r["correct"] for r in v)/len(v) if v else None) for t,v in by_t.items()},
            mean_prompt_tokens=st.mean(r["prompt_tokens"] for r in rs), mean_peak_prompt_tokens=st.mean(r.get("peak_prompt_tokens",r["prompt_tokens"]) for r in rs),
            mean_completion_tokens=st.mean(r["completion_tokens"] for r in rs), mean_reasoning_tokens=st.mean(r.get("reasoning_tokens",0) for r in rs),
            mean_cached_tokens=st.mean(r.get("cached_tokens",0) for r in rs), cached_fraction=(sum(r.get("cached_tokens",0) for r in rs)/max(1,sum(r["prompt_tokens"] for r in rs))),
            mean_cost_usd=st.mean(r["cost_usd"]+r.get("embed_cost_usd",0) for r in rs), total_cost_usd=sum(r["cost_usd"]+r.get("embed_cost_usd",0) for r in rs),
            mean_llm_calls=st.mean(r["llm_calls"] for r in rs), mean_tool_calls=st.mean(r["tool_calls"] for r in rs),
            mean_latency_s=st.mean(r["latency_s"] for r in rs), p90_prompt_tokens=sorted(r["prompt_tokens"] for r in rs)[int(0.9*(n-1))],
            errors=sum(1 for r in rs if r.get("error")))
        rows.append(row)
    # relative to baseline at same scale
    base={(r["corpus"], r["scale"]): r for r in rows if r["condition"]=="baseline"}
    for r in rows:
        b=base.get((r["corpus"], r["scale"]))
        r["prompt_tokens_vs_baseline"] = r["mean_prompt_tokens"]/b["mean_prompt_tokens"] if b else None
        r["cost_vs_baseline"] = r["mean_cost_usd"]/b["mean_cost_usd"] if b and b["mean_cost_usd"] else None
        r["accuracy_per_dollar"] = r["accuracy"]/r["mean_cost_usd"] if r["mean_cost_usd"] else None
    return rows

def write(recs, out):
    wb=Workbook(); hdr=Font(bold=True, color="FFFFFF"); fill=PatternFill("solid", fgColor="1F4E78")
    def sheet(ws, rows, cols, widths=None):
        ws.append(cols)
        for c in ws[1]: c.font=hdr; c.fill=fill; c.alignment=Alignment(wrap_text=True, vertical="top")
        for r in rows: ws.append([r.get(c) for c in cols])
        for i,c in enumerate(cols,1): ws.column_dimensions[get_column_letter(i)].width = (widths or {}).get(c, max(10, min(40, len(c)+2)))
        ws.freeze_panes="A2"
    # Summary
    rows=summarize(recs)
    ws=wb.active; ws.title="Summary"
    cols=["corpus","condition","scale","n","accuracy"]+[f"acc_{t}" for t in QTYPES]+["mean_prompt_tokens","prompt_tokens_vs_baseline","mean_peak_prompt_tokens","p90_prompt_tokens","mean_completion_tokens","mean_reasoning_tokens","mean_cached_tokens","cached_fraction","mean_llm_calls","mean_tool_calls","mean_latency_s","mean_cost_usd","cost_vs_baseline","accuracy_per_dollar","total_cost_usd","errors"]
    sheet(ws, rows, cols, {"condition":30})
    for row in ws.iter_rows(min_row=2):
        for c in row:
            if isinstance(c.value, float):
                c.number_format = "0.0%" if cols[c.column-1].startswith("acc") and "per" not in cols[c.column-1] else ("$0.0000" if "cost" in cols[c.column-1] and "vs" not in cols[c.column-1] else "0.00")
    # charts per scale
    scales=sorted({(r["corpus"], r["scale"]) for r in rows}); anchor_row=len(rows)+4
    ci = {c: cols.index(c)+1 for c in cols}
    for corp, sc in scales:
        sub=[i for i,r in enumerate(rows) if r["scale"]==sc and r["corpus"]==corp]
        if not sub: continue
        first, last = sub[0]+2, sub[-1]+2
        for j,(col,title) in enumerate([(ci["accuracy"],"Accuracy"),(ci["mean_prompt_tokens"],"Mean prompt tokens per question"),(ci["mean_cost_usd"],"Mean cost per question (USD)")]):
            ch=BarChart(); ch.type="col"; ch.title=f"{title} — {corp} corpus, scale {sc}"; ch.height=7; ch.width=14
            ch.add_data(Reference(ws, min_col=col, min_row=first, max_row=last), titles_from_data=False)
            ch.set_categories(Reference(ws, min_col=2, min_row=first, max_row=last)); ch.legend=None
            ws.add_chart(ch, f"{get_column_letter(1+j*8)}{anchor_row}")
        anchor_row += 16
    # By question type
    ws2=wb.create_sheet("ByQuestionType")
    g=defaultdict(list)
    for r in recs: g[(r["corpus"], label(r), r["scale"], r["qtype"])].append(r)
    rows2=[dict(corpus=k[0], condition=k[1], scale=k[2], qtype=k[3], n=len(v), accuracy=sum(r["correct"] for r in v)/len(v), mean_prompt_tokens=st.mean(r["prompt_tokens"] for r in v),
                mean_cost_usd=st.mean(r["cost_usd"] for r in v), mean_llm_calls=st.mean(r["llm_calls"] for r in v), mean_latency_s=st.mean(r["latency_s"] for r in v)) for k,v in sorted(g.items(), key=lambda x:(x[0][0],x[0][2],x[0][1],x[0][3]))]
    sheet(ws2, rows2, ["corpus","condition","scale","qtype","n","accuracy","mean_prompt_tokens","mean_cost_usd","mean_llm_calls","mean_latency_s"], {"condition":30})
    # Raw
    ws3=wb.create_sheet("Runs")
    cols3=["corpus","condition","tag","model","rep","scale","qid","qtype","convo","turn","convo_prompt_tokens_cum","question","gold","answer","correct","grader","prompt_tokens","peak_prompt_tokens","completion_tokens","reasoning_tokens","cached_tokens","llm_calls","tool_calls","latency_s","cost_usd","embed_cost_usd","error"]
    rows3=[{**r, "gold": json.dumps(r["gold"]) if isinstance(r["gold"], list) else r["gold"]} for r in sorted(recs, key=lambda r:(r["scale"], label(r), r["qid"]))]
    sheet(ws3, rows3, cols3, {"question":60,"answer":40,"gold":25,"error":40})
    # Conversations: cumulative context per multi-turn dialogue
    wsc=wb.create_sheet("Conversations")
    gc=defaultdict(list)
    for r in recs:
        if r.get("convo"): gc[(r["corpus"], label(r), r["scale"], r["convo"])].append(r)
    rowsc=[]
    for (corp, cond, sc, cv), v in sorted(gc.items()):
        v=sorted(v, key=lambda r: r["turn"])
        rowsc.append(dict(corpus=corp, condition=cond, scale=sc, convo=cv, turns=len(v), turns_correct=sum(r["correct"] for r in v), all_correct=all(r["correct"] for r in v),
                          total_prompt_tokens=sum(r["prompt_tokens"] for r in v), total_cost_usd=sum(r["cost_usd"] for r in v), total_llm_calls=sum(r["llm_calls"] for r in v),
                          last_turn_peak_prompt_tokens=v[-1].get("peak_prompt_tokens"), cached_tokens=sum(r.get("cached_tokens",0) for r in v),
                          per_turn=" | ".join(f"t{r['turn']}:{'ok' if r['correct'] else 'X'} {r['prompt_tokens']}tok" for r in v)))
    sheet(wsc, rowsc, ["corpus","condition","scale","convo","turns","turns_correct","all_correct","total_prompt_tokens","total_cost_usd","total_llm_calls","last_turn_peak_prompt_tokens","cached_tokens","per_turn"], {"condition":30,"per_turn":60})
    # Traces (tool calls) for agentic conditions
    ws4=wb.create_sheet("Traces")
    ws4.append(["corpus","condition","scale","qid","turn","tool","args","output (truncated)"])
    for c in ws4[1]: c.font=hdr; c.fill=fill
    for r in sorted(recs, key=lambda r:(r["scale"], label(r), r["qid"])):
        for t in r.get("trace", []):
            if "tool" in t: ws4.append([r["corpus"], label(r), r["scale"], r["qid"], t["turn"], t["tool"], json.dumps(t["args"])[:500], t["out"][:800]])
    for col,w in zip("ABCDEFGH",[8,28,8,12,6,10,60,80]): ws4.column_dimensions[col].width=w
    # Metrics definitions
    ws5=wb.create_sheet("Definitions")
    for row in [["metric","definition"],
        ["accuracy","fraction of questions graded correct (rule-based normalization; LLM judge for ambiguous cases)"],
        ["prompt_tokens","cumulative input tokens over every LLM call in the episode (what you pay for)"],
        ["peak_prompt_tokens","largest single-call input size in the episode (context pressure / how close to the window limit)"],
        ["cached_tokens / cached_fraction","input tokens served from the provider prompt cache (billed at a fraction of the price). The full-context baseline reaches ~93% cache hits when questions run back to back against the same corpus prefix, which cuts its warm cost to ~1/6 of its cold cost; token counts are the cache-independent measure"],
        ["reasoning_tokens","hidden reasoning tokens billed as output"],
        ["llm_calls / tool_calls","number of model calls and tool invocations per question"],
        ["cost_usd","OpenRouter-reported cost for the episode (embedding query cost added in Summary)"],
        ["prompt_tokens_vs_baseline","mean cumulative prompt tokens divided by the full-context baseline at the same corpus scale"],
        ["scale","corpus size knob: 1≈23k tokens, 3≈45k, 6≈78k, 12≈145k (same 39 questions at every scale)"],
        ["conditions","baseline = whole corpus in prompt; repl = corpus as python variable, model greps via tool; vector_rag = single-shot top-8 dense retrieval; vectorcontext_v1 = agentic hybrid (dense+BM25 RRF) search + open_doc tools; v3 = +python over workspace with search() callable from code; v4 = per-document fact cards (schema-first LLM extraction at index time, values canonicalized) searched by hybrid retrieval and aggregated in python"],
        ["tags","[compact] = 220-char sentence chunks; [mask] = observation masking, only last tool output kept verbatim; [schema] = schema-first extraction; [schema2] = schema-first + parse retry + value canonicalization; [v5] = v4 + raw-passage search_text tool + guided strategy prompt + index-time supersession resolution (current/history on cards); [v5min] = v5 with a minimal prompt (prompt-sensitivity ablation); [v4r] = v4 tools/prompt with the resolver only; [maskprev] = tool outputs of earlier conversation turns elided before each new turn; @model = agent model swapped (fact layer and judge stay on gpt-5.6-luna)"],
        ["corpus","base = 168 docs, 39 questions (needle/multihop/aggregation/global); hard = base + 16 change records that supersede overview facts (some superseded again) + collateral-impact mentions in 40% of incident reports; 33 hard questions (temporal, prose-only, negation, numeric, distractor) + 6 four-turn conversations (24 graded turns; follow-ups use pronouns)"],
        ["convo_prompt_tokens_cum","running total of prompt tokens over the turns of one conversation (context growth across turns)"],
        ["resolve_patches_applied","index-time supersession: number of change-record patches the resolver applied to fact cards (verified against ground truth by lab/check_resolve.py)"]]:
        ws5.append(row)
    for row in [["Chat sheet", "no-documents benchmark: the context to reduce is the conversation itself (lab/corpus_chat.py). chat_full = whole transcript in context; chat_window = last 10 exchanges; chat_summary = rolling LLM summary (compaction) + last 10; chat_repl = transcript in a Python variable + last 10; chat_state = online fact ledger (one extraction call per exchange) in the prompt + last 10; chat_vc = ledger + tools (search over facts, search over raw exchanges, open_turn, python)"],
                ["Chat question types", "early = fact from the first third, never revised; updated = latest value after explicit/coreferent/relational/swap revisions; original = value before the first revision; count/sum = over modules currently in scope (one dropped, one added); negation = module with no datastore; list = P0 modules; paste = number inside pasted tool output; absent = fact never stated (gold 'not stated')"],
                ["Handoff sheet", "agent-to-agent delegation: a worker agent must write a note containing every required fact (24 facts for the module table, 7-9 for the readiness summary). worker_full = worker gets the whole transcript; worker_brief = orchestrator (full context) writes a brief, worker gets the brief; worker_summary = worker gets the rolling summary; worker_state = worker gets the ledger; worker_vc = ledger + tools, no transcript. fact_recall = required facts present in the note (rule-graded, order-aware for the on-call list); stale = superseded values mentioned (not penalized)"],
                ["fold_model / batch", "model used for online folding (extraction + summaries; agent model may differ) and exchanges per extraction call"],
                ["ChatFolding sheet", "one row per (chat, folder model, variant): variant 'audit' = the folder re-read the raw exchanges in chunks of 20 against its finished ledger and patched omissions (audit_fields_before -> ledger_fields_correct, audit_patches applied); 'repeat:rep1' = folding redone from scratch with the cache bypassed; ledger_fields_checked counts per-module/paste fields plus the membership list (48 for 6-module chats, 88 dense)"],
                ["chat ids ending in m", "messy variant (lab/chat_messy.py): user turns paraphrased by the LLM into informal chat with every name/date/number kept verbatim (checked), assistant replies that do not restate the fact; same questions, tasks and ground truth as chat1-3"],
                ["condition suffixes", "[tag] = run tag (messy, foldglm, foldglm_audit); @model = agent model; fold@model = folder model; +audit = audited ledger; n counts pooled repeat runs (--rep)"]]:
        ws5.append(row)
    ws5.column_dimensions["A"].width=28; ws5.column_dimensions["B"].width=120
    # Index costs (one-time, amortized)
    ws6=wb.create_sheet("IndexCosts")
    ws6.append(["corpus","scale","n_docs","docs_extracted","prompt_tokens","completion_tokens","cost_usd","mode","resolve_patches_applied","note"])
    for c in ws6[1]: c.font=hdr; c.fill=fill
    ic=os.path.join(ROOT,"results","index_costs.jsonl")
    if os.path.exists(ic):
        seen=set()
        for line in open(ic):
            r=json.loads(line)
            if r.get("docs_extracted",0)==0: continue
            key=(r.get("corpus","base"), r["scale"], r.get("mode"))
            if key in seen: continue
            seen.add(key)
            ws6.append([r.get("corpus","base"), r["scale"], r["n_docs"], r["docs_extracted"], r["prompt_tokens"], r["completion_tokens"], r["cost_usd"], r.get("mode","free"), (r.get("resolve") or {}).get("applied"), "one-time fact-card extraction for vectorcontext_v4/v5; amortized over all future queries"])
    for col,w in zip("ABCDEFGHIJ",[8,8,8,14,14,16,10,10,12,80]): ws6.column_dimensions[col].width=w
    ea=os.path.join(ROOT,"results","extraction_accuracy.jsonl")
    if os.path.exists(ea):
        ws6.append([]); ws6.append(["scale","service fields checked","service field errors","service field accuracy","incident fields checked","incident field errors","incident field accuracy"])
        for c in ws6[ws6.max_row]: c.font=hdr; c.fill=fill
        last={}
        for line in open(ea):
            r=json.loads(line); last[r["scale"]]=r
        for sc,r in sorted(last.items()):
            ws6.append([sc, r["service_fields_checked"], r["service_field_errors"], 1-r["service_field_errors"]/r["service_fields_checked"], r["incident_fields_checked"], r["incident_field_errors"], 1-r["incident_field_errors"]/r["incident_fields_checked"]])
    # Literature summary
    ws7=wb.create_sheet("Literature")
    ws7.append(["family","reference","key result","implication for this lab"])
    for c in ws7[1]: c.font=hdr; c.fill=fill
    for row in [
        ["REPL","Recursive Language Models, Zhang/Kraska/Khattab 2025, arXiv:2512.24601","Corpus as a Python variable; RLM 91.3 vs 0.0 (overflow) on BrowseComp-Plus 1K docs; OOLONG-Pairs F1 58 vs 0.1 base","REPL wins aggregation; cost heavy-tailed"],
        ["REPL","Direct Corpus Interaction (DCI), Li et al. 2026, arXiv:2605.05242","grep/read agent 80.0 vs 69.0 dense-retrieval agent, -29% cost; but ~35 tool calls/q and collapse at 400K docs","grep misses paraphrase; needs index at scale"],
        ["REPL","Anthropic, Code execution with MCP (2025)","150,000 -> 2,000 tokens (98.7%) by keeping tool results in the sandbox","motivates the REPL arm"],
        ["Vector","Self-Route, Li et al. EMNLP 2024, arXiv:2407.16833","Long-context beats RAG on average; Self-Route matches LC with 38-61% of tokens; RAG fails on multi-step, vague, implicit queries","predicts vector_rag failure on multihop/aggregation"],
        ["Vector","Anthropic Contextual Retrieval (2024)","contextual headers + hybrid BM25 + rerank: retrieval failure 5.7% -> 1.9%","we use contextual headers + hybrid RRF"],
        ["Vector","BrowseComp-Plus, Chen et al. 2025, arXiv:2508.06600","retriever quality dominates: BM25 55.9% vs dense 70.1% (GPT-5)","dense search as an agent tool is worth +14 pts"],
        ["Vector","OOLONG, Bertsch et al. 2025, arXiv:2511.02817","every frontier model <50% at 128K; bottleneck is aggregation","aggregation must be done in code, not in context"],
        ["Hybrid","Context-Folding, Sun et al. ICML 2026, arXiv:2510.11967","fold sub-trajectories; 10x smaller active context at equal/better accuracy","masking/folding history; our naive masking hurt accuracy"],
        ["Hybrid","IRCoT, arXiv:2212.10509","interleaved retrieval + reasoning: +15 QA pts on multi-hop","agentic multi-round search (vectorcontext_v1)"],
        ["Hybrid","RAPTOR, ICLR 2024, arXiv:2401.18059","retrievable summary nodes above leaves","our fact cards are a per-doc structured summary layer"],
        ["Temporal","Zep / Graphiti, Rasmussen et al. 2025, arXiv:2501.13956","temporal knowledge graph for agent memory: facts carry validity intervals and are invalidated by later facts; 94.8% on DMR vs 93.4% MemGPT, up to 18.5% on LongMemEval","our v5 resolver: change records patch a `current` view on cards and keep `history`, source fields untouched (provenance)"],
        ["Multi-turn","LongMemEval, Wu et al. ICLR 2025, arXiv:2410.10813","commercial assistants drop ~30% accuracy over long multi-session histories; session decomposition + fact-augmented keys help","our conversation set: pronoun follow-ups; measures context growth per turn and cross-turn masking"],
    ]: ws7.append(row)
    for col,w in zip("ABCD",[10,60,80,50]): ws7.column_dimensions[col].width=w
    for row in ws7.iter_rows(min_row=2):
        for c in row: c.alignment=Alignment(wrap_text=True, vertical="top")
    chat_sheets(wb, hdr, fill, sheet)
    wb.save(out); print("wrote", out, "from", len(recs), "records")

def chat_sheets(wb, hdr, fill, sheet):
    """No-documents benchmark: chat probes, agent-to-agent hand-offs and folding costs (results/chat_*.jsonl, handoff_*.jsonl)."""
    sys.path.insert(0, os.path.dirname(__file__))
    import summary_chat as sc
    from corpus_chat import build_chats
    from run_chat import grade_task
    QT = sc.QT
    recs = sc.regrade(sc.load("chat_*.jsonl"))
    if not recs: return
    size = sc.size
    g = defaultdict(list)
    for r in recs: g[(size(r), sc.lab(r))].append(r)
    ws = wb.create_sheet("Chat")
    rows = []
    for (sz, cond), v in sorted(g.items()):
        n = len(v)
        rows.append(dict(chats=sz, condition=cond, n=n, accuracy=sum(r["correct"] for r in v) / n,
                         **{f"acc_{t}": (sum(r["correct"] for r in v if r["qtype"] == t) / max(1, sum(1 for r in v if r["qtype"] == t)) if any(r["qtype"] == t for r in v) else None) for t in QT},
                         mean_prompt_tokens=st.mean(r["prompt_tokens"] for r in v), mean_peak_prompt_tokens=st.mean(r["peak_prompt_tokens"] for r in v), mean_llm_calls=st.mean(r["llm_calls"] for r in v),
                         mean_cost_usd=st.mean(r["cost_usd"] for r in v), cached_fraction=sum(r.get("cached_tokens", 0) for r in v) / max(1, sum(r["prompt_tokens"] for r in v)), errors=sum(1 for r in v if r.get("error"))))
    cols = ["chats", "condition", "n", "accuracy"] + [f"acc_{t}" for t in QT] + ["mean_prompt_tokens", "mean_peak_prompt_tokens", "mean_llm_calls", "mean_cost_usd", "cached_fraction", "errors"]
    sheet(ws, rows, cols); ws.column_dimensions["B"].width = 34
    for r in ws.iter_rows(min_row=2, min_col=4, max_col=4 + len(QT)):
        for c in r: c.number_format = "0%"
    # charts: accuracy and prompt tokens per condition, normal chats
    normal = [i for i, r in enumerate(rows, start=2) if r["chats"] == "normal"]
    if normal:
        lo, hi = min(normal), max(normal)
        ch = BarChart(); ch.title = "No-documents chats (3 x ~29k tokens): accuracy"; ch.y_axis.title = "accuracy"; ch.height = 8; ch.width = 18
        ch.add_data(Reference(ws, min_col=4, min_row=lo, max_row=hi)); ch.set_categories(Reference(ws, min_col=2, min_row=lo, max_row=hi)); ch.legend = None
        ws.add_chart(ch, f"B{len(rows) + 4}")
        ch2 = BarChart(); ch2.title = "No-documents chats: mean prompt tokens per question"; ch2.height = 8; ch2.width = 18
        ch2.add_data(Reference(ws, min_col=5 + len(QT), min_row=lo, max_row=hi)); ch2.set_categories(Reference(ws, min_col=2, min_row=lo, max_row=hi)); ch2.legend = None
        ws.add_chart(ch2, f"L{len(rows) + 4}")
    ws2 = wb.create_sheet("ChatRuns")
    cols2 = ["condition", "tag", "model", "fold_model", "batch", "style", "audit", "fold_salt", "rep", "chat", "n_turns", "qid", "qtype", "question", "gold", "answer", "correct", "grader", "prompt_tokens", "peak_prompt_tokens", "completion_tokens", "cached_tokens", "llm_calls", "tool_calls", "latency_s", "cost_usd", "error"]
    sheet(ws2, [{**r, "gold": json.dumps(r["gold"]) if isinstance(r["gold"], list) else r["gold"]} for r in sorted(recs, key=lambda r: (r["chat"], sc.lab(r), r["qid"]))], cols2)
    ws2.column_dimensions["K"].width = 50; ws2.column_dimensions["M"].width = 40
    # hand-off
    hrecs = sc.load("handoff_*.jsonl")
    if hrecs:
        tasks = {t.id: t for c in build_chats() for t in c.tasks}
        for r in hrecs:
            if r["task"] in tasks and not r.get("error") and len(r["answer"]) < 1500:
                r.update(grade_task(r["answer"], tasks[r["task"]])); r["full_credit"] = r["found"] == r["required"]
        ws3 = wb.create_sheet("Handoff")
        gh = defaultdict(list)
        for r in hrecs: gh[(size(r), sc.lab(r))].append(r)
        orch = lambda r: (r.get("extra") or {}).get("orchestrator") or {}
        rows3 = [dict(chats=sz, worker_condition=cond, tasks=len(v), facts_required=sum(r["required"] for r in v), facts_found=sum(r["found"] for r in v), fact_recall=sum(r["found"] for r in v) / sum(r["required"] for r in v),
                      tasks_fully_correct=sum(r["full_credit"] for r in v), stale_values_presented=sum(len(r["stale_present"]) for r in v), mean_tokens_to_worker=st.mean(r["prompt_tokens"] for r in v),
                      mean_orchestrator_tokens=st.mean(orch(r).get("prompt_tokens", 0) for r in v), mean_total_cost_usd=st.mean(r["cost_usd"] + orch(r).get("cost_usd", 0) for r in v)) for (sz, cond), v in sorted(gh.items())]
        sheet(ws3, rows3, ["chats", "worker_condition", "tasks", "facts_required", "facts_found", "fact_recall", "tasks_fully_correct", "stale_values_presented", "mean_tokens_to_worker", "mean_orchestrator_tokens", "mean_total_cost_usd"])
        ws3.column_dimensions["B"].width = 34
        for r in ws3.iter_rows(min_row=2, min_col=6, max_col=6):
            for c in r: c.number_format = "0%"
        ws3.append([]); ws3.append(["condition", "tag", "model", "fold_model", "audit", "rep", "chat", "task", "required", "found", "missing", "stale_present", "prompt_tokens", "cost_usd", "note (truncated)"])
        for c in ws3[ws3.max_row]: c.font = hdr; c.fill = fill
        for r in sorted(hrecs, key=lambda r: (r["chat"], sc.lab(r), r["task"])):
            ws3.append([r["condition"], r["tag"], r["model"], r.get("fold_model"), bool(r.get("audit")), r.get("rep", 0), r["chat"], r["task"], r["required"], r["found"], json.dumps(r["missing"]), json.dumps(r["stale_present"]), r["prompt_tokens"], r["cost_usd"], r["answer"][:1200]])
    # folding costs
    last = sc.fold_merge()
    if last:
        ws4 = wb.create_sheet("ChatFolding")
        ws4.append(["chat", "fold_model", "variant", "exchanges", "chat_tokens", "ledger_facts", "ledger_tokens", "ledger_fields_correct", "ledger_fields_checked", "ledger_errors", "extraction_calls", "extraction_cost_usd", "audit_fields_before", "audit_patches", "audit_cost_usd", "summary_tokens", "summary_calls", "summary_cost_usd", "note"])
        for c in ws4[1]: c.font = hdr; c.fill = fill
        for (c, fm, b, au, salt), r in sorted(last.items()):
            L, S = r.get("ledger", {}), r.get("summary", {}); A = L.get("audit") or {}
            var = ", ".join(x for x in [f"batch{b}" if b != 1 else "", "audit" if au else "", f"repeat:{salt}" if salt else ""] if x) or "-"
            ws4.append([c, fm, var, r["n_turns"], r["chat_tokens"], L.get("n_facts"), L.get("ledger_tokens"), L.get("check", {}).get("ok"), L.get("check", {}).get("fields"), json.dumps(L.get("errors", []))[:500], L.get("llm_calls", 0) + L.get("cached_calls", 0), L.get("cost_usd"),
                        (A.get("check_before") or {}).get("ok"), len(A.get("applied", [])) if A else None, A.get("cost_usd") if A else None, S.get("summary_tokens"), S.get("llm_calls", 0) + S.get("cached_calls", 0), S.get("cost_usd"),
                        "online folding paid once per conversation; ledger checked against generator state (per-field values + membership list); cost 0 means every call was served from cache in this run"])
        for col, w in zip("ABCDEFGHIJKLMNOPQRS", [8, 22, 14, 9, 10, 10, 10, 10, 10, 40, 10, 12, 10, 10, 10, 10, 10, 12, 60]): ws4.column_dimensions[col].width = w

if __name__ == "__main__":
    pats = sys.argv[2:] or ["run_*.jsonl"]
    write(load(pats), sys.argv[1] if len(sys.argv)>1 else os.path.join(ROOT, "results", "vectorcontext_results.xlsx"))
