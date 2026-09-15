# Findings: can vectors replace a REPL workspace for context reduction?

Part 1 (2026-09-14) is the original benchmark. Part 2 (2026-09-15) is the hard benchmark: temporal supersession, distractors, prose-only facts, negation, numeric joins, multi-turn conversations, a 504-document corpus, prompt ablations, and two cheaper agent models.

# Part 1: original benchmark

Model: `openai/gpt-5.6-luna` via OpenRouter ($0.20/M input, $1.20/M output). Same 39 questions at every corpus scale
(needle lookup, multi-hop, aggregation, global list/negation). Full per-question data, traces, and charts are in
`results/vectorcontext_results.xlsx`. Total spend: about $3.20 for Part 1, $5.50 for Part 2 and $4.35 for Part 3 and $1.95 for Part 4 (all runs, pilots, index building, and reruns included).

## Headline (corpus ≈ 45k tokens, 39 questions)

| condition | accuracy | needle | multi-hop | aggregation | global | prompt tokens / q | vs baseline | peak context | LLM calls | cost / q |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (full context) | 95% | 100% | 100% | 90% | 89% | 46,531 | 100% | 46,531 | 1.0 | $0.0107 |
| REPL workspace | 87% | 100% | 100% | 80% | 67% | 5,047 | 11% | 1,870 | 3.8 | $0.0011 |
| vector RAG (top-8, one shot) | 38% | 100% | 10% | 30% | 11% | 889 | 2% | 889 | 1.0 | $0.0003 |
| vectorcontext v1 (agentic hybrid search) | 97% | 100% | 100% | 90% | 100% | 18,512 | 40% | 9,391 | 3.4 | $0.0030 |
| **vectorcontext v4 (fact cards, final)** | **97%** | 100% | 100% | 90% | 100% | **4,715** | **10%** | 2,495 | 2.9 | **$0.0010** |

## Scaling with corpus size (same facts, longer documents)

| scale | corpus tokens | baseline acc / tokens / cost | REPL acc / tokens / cost | v4 acc / tokens / cost |
|---|---|---|---|---|
| 3 (39 q) | 45k | 95% / 46.5k / $0.0107 | 87% / 5.0k / $0.0011 | 97% / 4.7k / $0.0010 |
| 6 (20 q) | 78k | 100% / 79.3k / $0.0200 | 95% / 6.3k / $0.0012 | 100% / 4.9k / $0.0010 |
| 12 (12 q) | 145k | 100% / 144.7k / $0.0363 | 92% / 4.8k / $0.0011 | 100% / 6.1k / $0.0012 |

Baseline cost grows linearly with the corpus. REPL and v4 are flat: their per-question cost depends on the number of
facts touched, not on how much prose surrounds them. Agentic chunk search (v1) is not flat: at 78k tokens it used
41k prompt tokens per question because retrieved chunks carry the extra prose along.

One-time index cost for v4 (schema induction + one extraction call per document + value canonicalization):
$0.11 at 45k tokens, $0.12 at 78k, $0.17 at 145k. Fact-card extraction accuracy against ground truth was 100% on
420 checked fields at every scale after the fixes below. The index pays for itself after roughly 12 questions versus
the baseline at 45k tokens, and after 5 questions at 145k.

## What the iterations taught

1. **Single-shot vector RAG is not a context-reduction strategy for agents.** It is cheapest but fails every question
   that needs more than one passage (multi-hop 10%, aggregation 30%, global 11%). This matches the Self-Route and
   OOLONG findings in the literature review.
2. **The REPL workspace is cheap and strong on structure, weak on paraphrase.** Every REPL miss came from a regex that
   matched one phrasing ("owned by the Identity team") and missed the others ("Ownership: Identity",
   "the Identity team maintains"). That is exactly the failure mode the DCI paper reports for grep-based agents.
3. **Agentic vector search (v1) fixes paraphrase but pays in tokens.** Multi-round hybrid search reached 97% but
   cumulative context was 3.7x REPL, and it scaled with document length.
4. **Compact sentence chunks and observation masking both hurt.** Smaller chunks fragmented evidence (aggregation fell
   to 70%). Masking older tool outputs (keep last 1) dropped accuracy to 69% and doubled calls because the model
   re-ran searches. Naive folding is not free; the Context-Folding paper trains for it.
5. **Fact cards were the breakthrough, but only after three fixes.** Free-form extraction produced inconsistent keys
   (owner vs owner_team vs maintaining_team), so Python filters missed rows (85%). Schema-first extraction fixed keys
   (90%). Retrying failed JSON, canonicalizing values ("Platform team" to "Platform"), and mapping null-like strings
   ("unassigned") to null pushed it to 95%. Requiring atomic single-fact schema fields with more sample documents
   fixed a composite field the scale-6 schema had invented (region + replicas) and gave 97% and 100%.
6. **Remaining v4 misses are model slips, not retrieval failures**: one multi-hop question where the model answered
   from the wrong team, and one tie question where it named both tied services instead of one.

## Answer to the original question

Vectors alone do not replace a REPL workspace. A vector index gives paraphrase-robust lookup; the REPL gives exact
aggregation. The combination that won is a **semantic layer built once at index time (per-document fact cards with
consistent keys), searched with hybrid dense + BM25 retrieval, and aggregated in a Python REPL**. On this benchmark it
matched or beat full context on accuracy at about a tenth of the tokens and cost, matched the REPL workspace's cost,
and beat the REPL on accuracy by 10 points at 45k tokens and 5 to 8 points at larger scales.

## Caveats

- Synthetic corpus with templated prose and a single model; real documents are messier and extraction accuracy will be
  lower than 100%. Extraction quality is the load-bearing assumption of v4 and should be measured on real data.
- Scaling here adds prose, not documents. Scaling by document count would stress retrieval recall and fact-layer size.
- Small samples at scale 6 (20 questions) and 12 (12 questions); single run per condition, so differences under
  about 5 points are within noise. Repeated runs of v4 at scale 3 landed at 90%, 95%, 92%, 97% across iterations.
- The grader is strict on tie answers and the LLM judge is the same model family as the system under test.


# Part 2: hard benchmark

Corpus `hard` = the 45k-token base corpus plus 16 change records that supersede overview facts (4 of them superseded again
by a later record) and collateral-impact sentences naming a second service in 40% of incident reports (184 docs, 47k tokens).
Corpus `hardx3` = the same generator with 3x the entities (120 services, 180 incidents, 504 docs, 71k tokens) and short prose.
33 single questions in five families plus 6 four-turn conversations (24 graded turns) per corpus. Every gold answer is computed
from generator state; the code that generated each question set is pinned so every stored record is reproducible.
Full data: `results/vectorcontext_results.xlsx` (Summary, ByQuestionType, Conversations, Runs, Traces, IndexCosts).

## Headline (corpus `hard`, 57 graded items, agent model gpt-5.6-luna unless marked)

| condition | accuracy | temporal | prose | negation | numeric | distractor | multi-turn | prompt tok / item | peak | cache hits | calls | cost / item |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (full context) | 86% | 100% | 100% | 100% | 33% | 100% | 83% | 49,287 | 49,287 | 91% | 1.0 | $0.0022 |
| REPL workspace | 82% | 78% | 100% | 83% | 50% | 67% | 92% | 5,328 | 2,222 | 62% | 3.1 | $0.0010 |
| vector RAG (top-8, one shot) | 40% | 78% | 67% | 0% | 17% | 50% | 33% | 1,538 | 1,538 | 60% | 1.0 | $0.0003 |
| vectorcontext v1 (agentic chunk search) | 93% | 78% | 100% | 100% | 83% | 83% | 100% | 26,822 | 15,433 | 57% | 2.8 | $0.0037 |
| vectorcontext v4 (fact cards, linked) | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 4,934 | 2,459 | 51% | 2.5 | $0.0008 |
| **vectorcontext v5 (v4 + resolver + text search)** | **100%** | 100% | 100% | 100% | 100% | 100% | 100% | **5,262** | 2,751 | 78% | 2.4 | **$0.0006** |
| v5, minimal prompt | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 5,676 | 3,076 | 71% | 2.5 | $0.0007 |
| baseline @ glm-5.3-flash | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 49,651 | 49,651 | 97% | 1.0 | $0.0027 |
| REPL @ glm-5.3-flash | 89% | 89% | 100% | 83% | 67% | 83% | 96% | 6,278 | 1,906 | 59% | 4.1 | $0.0006 |
| v5 @ glm-5.3-flash | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 7,500 | 3,155 | 69% | 2.9 | $0.0007 |
| baseline @ deepseek-v4.1-flash | 96% | 100% | 100% | 100% | 67% | 100% | 100% | 50,614 | 50,614 | 96% | 1.0 | $0.0022 |
| REPL @ deepseek-v4.1-flash | 86% | 78% | 100% | 50% | 67% | 100% | 96% | 11,043 | 2,824 | 68% | 4.7 | $0.0019 |
| v5 @ deepseek-v4.1-flash | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 11,499 | 4,036 | 68% | 3.4 | $0.0018 |

Cost per item is OpenRouter-reported and includes prompt-cache discounts: the baseline ran with 91-97% cache hits because
57 items share one 47k-token prefix, which cuts its warm cost to roughly a sixth of its cold cost ($0.013 per item when the
cache was cold). Tokens are the cache-independent number: v5 uses 11% of the baseline's prompt tokens and 6% of its peak
context.

## Scaling by document count (corpus `hardx3`, 504 docs, 71k tokens)

| condition | accuracy | temporal | negation | numeric | multi-turn | prompt tok / item | peak | cost / item |
|---|---|---|---|---|---|---|---|---|
| baseline (full context) | 70% | 89% | 50% | 50% | 62% | 76,159 | 76,159 | $0.0142 |
| REPL workspace | 82% | 67% | 67% | 83% | 92% | 6,339 | 2,314 | $0.0011 |
| vectorcontext v4 | 98% | 100% | 83% | 100% | 100% | 5,589 | 2,559 | $0.0009 |
| **vectorcontext v5** | **100%** | 100% | 100% | 100% | 100% | 6,598 | 2,863 | $0.0007 |
| baseline @ glm-5.3-flash | 84% | 100% | 67% | 67% | 83% | 77,168 | 77,168 | $0.0064 |
| REPL @ glm-5.3-flash | 84% | 67% | 83% | 50% | 96% | 7,031 | 2,069 | $0.0016 |
| v5 @ glm-5.3-flash | 100% | 100% | 100% | 100% | 100% | 8,331 | 3,110 | $0.0007 |
| baseline @ deepseek-v4.1-flash | 86% | 100% | 100% | 67% | 79% | 78,515 | 78,515 | $0.0088 |
| REPL @ deepseek-v4.1-flash | 77% | 89% | 50% | 67% | 79% | 13,795 | 3,252 | $0.0031 |
| v5 @ deepseek-v4.1-flash | 98% | 100% | 83% | 100% | 100% | 11,817 | 3,954 | $0.0016 |

The baseline's failure mode at 504 documents is not the context window (71k fits easily): it miscounts and mis-sums when
40 to 120 entities are involved (16 of its 17 misses are counts, sums, averages, or list questions). This is the OOLONG
finding reproduced: aggregation over long context degrades with the number of items, not the number of tokens. REPL and
fact-card costs are flat across both corpora because they depend on the facts touched, not the corpus.

One-time index cost for v5: $0.10 for `hard` (76 docs re-extracted, rest cached) and $0.25 for `hardx3` (504 docs extracted
plus resolver). Extraction accuracy on `hard`: 612 of 612 checked fields correct (service, incident incl. primary vs
collateral service, change record). Resolver accuracy after the fixes below: 16 of 16 change records applied and 0 errors on
120 (`hard`) and 360 (`hardx3`) current-state fields.

## Multi-turn conversations (6 conversations x 4 turns, pronoun follow-ups)

| condition | conversations fully correct | prompt tokens per conversation | peak single call |
|---|---|---|---|
| baseline | 4 / 6 | 197,293 | 49,396 |
| REPL | 4 / 6 | 18,291 | 3,021 |
| REPL, earlier turns' tool output elided | 4 / 6 | 14,872 | 2,739 |
| vectorcontext v1 | 6 / 6 | 91,958 | 19,662 |
| vectorcontext v5 | 6 / 6 | 17,365 | 3,251 |
| v5, earlier turns' tool output elided | 6 / 6 | 14,765 | 2,266 |
| v5 @ glm-5.3-flash / @ deepseek-v4.1-flash | 6 / 6 each | 21,906 / 34,944 | 3,903 / 5,560 |

The baseline re-sends the corpus every turn (cache makes it cheap, not small). Eliding earlier turns' tool outputs
(cross-turn folding) cut v5's conversation tokens by 15% with no accuracy loss, because the answers already written in the
assistant turns carry the state; the within-turn masking that hurt in Part 1 stays off.

## Bottlenecks found and fixed (each one is a stored run in the workbook)

1. **Value canonicalization destroyed identifiers.** The canonicalizer chose "the shortest clean form", so `thumbnail-api` /
   "the Thumbnail API" collapsed to "Thumbnail" and joins on service name became fuzzy; answers like "Edge" were graded wrong.
   Fix: deterministic entity linking (any value equal to a card's alias becomes that card's canonical subject, alias entries
   win over a change record's own subject) plus a canonicalizer rule that a machine identifier is always the canonical form.
   Result: 41 values linked on `hard`, 133 on `hardx3`; zero unlinked service references.
2. **The first resolver overwrote source fields.** Applying change records in place made "per the service overview" questions
   wrong (v5 pilot 82% vs v4 95%). Fix: cards keep what their document says; the resolver adds a `current` view and a
   `history` list (Zep/Graphiti-style invalidation rather than overwrite). Several of my own questions were also ambiguous about
   which state they meant and were reworded; the pilot runs are kept in `results/*_pilot.jsonl`.
3. **Resolver recall dropped with corpus size.** One 67k-token call over all 504 cards missed 2 of 16 change records.
   Fix: batches of 60 full cards plus a compact directory of every card (id, subject, aliases, fields) so a batch can target
   any card. Result: 16 of 16 on both corpora.
4. **Typed fields.** The `hardx3` schema named the on-call field `pager_contact`, and "pages fall through to the team lead"
   was extracted as the person "team lead", so a negation question ("no designated on-call") missed two services (v4 and
   v5pre 98%). Fix: a deterministic type check from the schema descriptions: person-typed fields must hold a proper name.
   Result: v5 100% on `hardx3`.
5. **Grader.** Word-boundary matching (so "no" does not match inside "now"), tied answers accepted via aliases, and a
   tie-break instruction on the one conversation opener that had a genuine tie.

## What did not matter

- **Prompt wording (for luna).** The minimal prompt ("use the tools, then answer") scored the same as the guided strategy
  prompt (100% vs 100%, 5.7k vs 5.3k tokens). The fact layer, not the prompt, carries the capability. The resolver alone
  (v4 tools + `current`/`history`, tag v4r) also scored 100%.
- **Raw-passage search.** v5's `search_text` tool was rarely needed: prose-only questions were answered by `open_doc` on the
  card's document in every arm that had the card.

## Model comparison

- **glm-5.3-flash and deepseek-v4.1-flash reach 100% with v5**, same as luna, at 1.4x and 2.2x the tokens (more tool
  rounds). Under the REPL both are worse and noisier: DeepSeek emitted pseudo tool calls as plain text in 5 of its 8 REPL
  misses and hit the 12-call limit; GLM did so once. The fact layer makes the cheap models' agent loop shorter and more
  robust because the code they write is a filter over normalized keys instead of regexes over prose.
- **Full context favours the cheaper models on arithmetic at 47k tokens**: GLM 100%, DeepSeek 96%, luna 86% (luna's misses are
  all averages, sums, and counts; it does not use tools in that arm). At 504 documents the same two models fall to 84% and 86% with full context (7 of GLM's 9 misses and 3 of DeepSeek's 8 are counts,
  sums, or lists; DeepSeek also returned 3 empty or malformed answers), while v5 holds at 100% and 98%. Every model, cheap or
  not, loses the ability to aggregate in-context once enough entities are involved; the fact layer plus code does not.

## Answer, updated

On problems built to defeat each approach, the semantic fact layer (schema-first extraction, entity linking, typed validation,
index-time supersession) searched by hybrid retrieval and aggregated in a Python REPL scored 100% on both corpora with three
different agent models, at 6 to 11% of the full-context tokens and at flat cost as the corpus grows. Vectors alone
(single-shot RAG) stayed at 40%; the REPL workspace alone stayed at 82 to 89% because grep misses paraphrase, supersession, and
primary-vs-collateral distinctions; full context degraded to 70% at 504 documents. The load-bearing component is the index-time
extraction and its verification, which costs $0.10 to $0.25 per corpus and is amortized over every query.

## Caveats

- Synthetic corpus and synthetic change records; extraction was 100% here and will not be on real documents. The checks in
  `lab/check_facts.py` and `lab/check_resolve.py` are the template for measuring that.
- Single run per condition (57 items each); differences under about 5 points are within noise. v5 was run four times on
  `hard` across the iteration (tags v5pre, v5, v5min, v4r) and scored 100% each time.
- The two cheaper models were only swapped in as the agent; extraction and the judge stayed on luna.

# Part 3: no documents — long chats and agent-to-agent hand-off

The question here was whether the approach helps when there is no corpus at all: the context to reduce is the conversation
itself (or an agent's own history). Benchmark: `lab/corpus_chat.py` generates project-planning chats in which every fact arrives
through the dialogue: 39 fact-setting exchanges, revisions that are explicit ("moves to Oct 22"), coreferent ("actually, make
that $65k"), relative ("pull it in by three days"), relational ("same budget as ingest") or a swap of two owners, one module
dropped and one added, three pasted tool outputs (load test, CI run, config), and 69 tangents that make up more than half of the
tokens. Chats: three of ~29k tokens (114 exchanges, ~72 facts), one of ~72k tokens (186 exchanges), one fact-dense chat with 14
modules (113 facts). 19 probe questions per chat (early / updated / original / count / sum / negation / list / paste / absent)
and two hand-off tasks per chat in which a worker agent must write a note containing 24 (module table) or 7 to 9 (readiness
summary) facts. Gold answers come from generator state; the ledger produced by online folding is checked field by field against
that state. Data: `results/chat_*.jsonl`, `results/handoff_*.jsonl`; sheets Chat, ChatRuns, Handoff, ChatFolding in the workbook;
tables via `uv run python lab/summary_chat.py`.

Conditions (all keep the last 10 exchanges verbatim except `chat_full` and `chat_window`):

| condition | what the model sees at question time | folding cost (paid once, online) |
|---|---|---|
| `chat_full` | the whole transcript | none |
| `chat_window` | the last 10 exchanges | none |
| `chat_summary` | rolling LLM summary (compaction every 20 exchanges, ~700 words) + last 10 | 6 to 9 calls, $0.014 to $0.028 per chat |
| `chat_repl` | last 10 + a `python` tool over the full transcript (`turns`) | none |
| `chat_state` | a **fact ledger** (entity.attribute = value with turn numbers and superseded values) + last 10 | one extraction call per exchange conditioned on the ledger so far (coreference resolves): 114 to 186 calls, $0.05 to $0.10 per chat |
| `chat_vc` | ledger + tools: hybrid search over ledger entries, hybrid search over raw exchanges, `open_turn`, `python` over `state`/`facts`/`turns` | same ledger |

## Probes (agent gpt-5.6-luna, folding by gpt-5.6-luna)

| chats | condition | accuracy | weak spots | prompt tok / q | calls | cost / q |
|---|---|---|---|---|---|---|
| 3 x 29k | chat_full | 100% | | 30,333 | 1.0 | $0.0077 |
| | chat_window | 25% | everything before the window | 2,954 | 1.0 | $0.0008 |
| | chat_summary | 100% | | 3,961 | 1.0 | $0.0011 |
| | chat_repl | 95% | sums 33% (misses revisions while grepping) | 9,588 | 2.6 | $0.0014 |
| | chat_state | 100% | | 4,039 | 1.0 | $0.0011 |
| | chat_vc | 100% | | 5,363 | 1.2 | $0.0012 |
| 1 x 72k | chat_full | 100% | | 74,367 | 1.0 | $0.0187 |
| | chat_summary | 100% | | 4,872 | 1.0 | $0.0013 |
| | chat_repl | 89% | sum 0%, one invented answer to an absent fact | 13,232 | 2.9 | $0.0017 |
| | chat_state | 100% | | 4,908 | 1.0 | $0.0013 |
| | chat_vc | 100% | | 6,538 | 1.2 | $0.0015 |
| dense (113 facts) | chat_full | 95% | sum over 14 modules wrong | 31,458 | 1.0 | $0.0080 |
| | chat_summary | 100% | | 3,727 | 1.0 | $0.0010 |
| | chat_repl | 95% | | 9,541 | 2.8 | $0.0013 |
| | chat_state | 100% | | 4,294 | 1.0 | $0.0012 |
| | chat_vc | 100% | | 5,767 | 1.2 | $0.0013 |

Folding cuts tokens per question by 7x (29k chats) to 15x (72k chat) with no accuracy loss for luna. On spot questions the
rolling summary and the fact ledger are indistinguishable; the difference shows up below. Full context is the arm that fails
on the fact-dense chat, in the same way it failed on the 504-document corpus: an in-context sum over 14 entities. The REPL over
the transcript is the weakest reduced arm: grep finds the first statement of a budget and misses the later revision.

## Agent-to-agent hand-off (10 tasks, 195 required facts; agent gpt-5.6-luna)

| worker receives | facts found | tasks fully correct | tokens to worker | orchestrator tokens | cost / task |
|---|---|---|---|---|---|
| the whole transcript (`worker_full`) | 194/195 (99%) | 9/10 | 39,382 | 0 | $0.0102 |
| a brief written by the orchestrator from its full context (`worker_brief`) | 195/195 (100%) | 10/10 | 331 | 39,394 | $0.0106 |
| the rolling summary (`worker_summary`) | 176/195 (90%) | 1/10 | 1,167 | 0 | $0.0006 |
| the fact ledger (`worker_state`) | 195/195 (100%) | 10/10 | 1,336 | 0 | $0.0006 |
| the ledger + tools, no transcript (`worker_vc`) | 195/195 (100%) | 10/10 | 2,470 | 0 | $0.0008 |

This is where the summary breaks: it answered every probe, but a hand-off needs all facts at once and the summary is missing
about 10% of them (an added module's owner and deadline, a datastore, the on-call order, a P0 flag); only 1 of 10 hand-off
notes was complete. The ledger is complete (47/47 fields on every chat, 87/87 on the dense one) at the same size (~1.0 to 1.7k
tokens). Writing a brief works but costs the orchestrator a full-context call each time; passing the transcript costs 30x the
tokens and still missed one owner on the 72k chat.

## Folder and agent model swaps

| variant | probes (29k / 72k / dense) | hand-off facts | note |
|---|---|---|---|
| folding by glm-5.3-flash, ledger only (`chat_state[foldglm]`) | 89% / 100% / 100% | 175/195, 6/10 tasks | GLM's ledger lost 4 of 47 fields on one chat (kickoff facts, scope) |
| folding by glm-5.3-flash, ledger + tools (`chat_vc[foldglm]`) | 96% / 100% / 100% | 195/195, 10/10 tasks | raw-exchange search repairs the ledger's gaps; 2.3 to 4.2k tokens to the worker |
| folding by glm-5.3-flash, summary (`chat_summary[foldglm]`) | 100% / 100% / 100% | 176/195, 1/10 | same 10% loss as luna's summary |
| agent glm-5.3-flash on luna folds: full / summary / state / vc | 100-95-100 / 96-100-100 / 100-100-95 / 100-100-100 | full 195, summary 175, state 194, vc 195 | vc with GLM uses 1.8 calls and ~9k tokens; GLM handles the 30k transcript at $0.0016/q |
| folding 5 exchanges per call (`--batch 5`) | not run | | ledger 42 to 45 of 47 fields: pasted numbers dropped; 2.6x cheaper folding, rejected |

## Bottlenecks found and fixed

1. **Tangents mined for facts.** The first extraction rule pulled numbers out of the assistant's explanations ("canary 5%",
   "logs kept 30 days"), about ten junk entries per tangent, which would have grown the ledger toward the size of the transcript.
   Rule changed to project facts the user states or pastes; explanations, questions and chit-chat yield nothing. Ledger: ~70 real
   facts, 1k tokens, zero junk.
2. **Stale membership lists.** The kickoff turn produced `project.in_scope_modules = [six names]`; adding a module later set
   `exporter.in_scope = true` but left the list unchanged, and the agent trusted the list (chat3 count and sum wrong in
   `chat_state` and `chat_vc`). Rule changed: when an exchange adds or removes a member of a list already in the ledger, emit the
   updated list too. Reran: 100%.
3. **Batched folding loses pasted numbers** (above). Kept one call per exchange.
4. **A slicing bug** (`turns[-0:]` is the whole list) had given `worker_vc` the full transcript; caught because its token count
   matched `worker_full`. Fixed and rerun.
5. **Grading the on-call order**: workers wrote it as a numbered list; the grader now checks the three names appear in order.
6. **Cache writes from threads** collided on a temp file name; serialized with a lock.

## What did not matter

- Summary vs ledger on single-fact probes (both 100% with luna, both at ~4k tokens).
- The agent model, once the context is folded: glm-5.3-flash matches luna on `chat_state` and `chat_vc` (100% / 100% / 95%)
  and on hand-offs (194 to 195 of 195). Folding quality is what moves the numbers, and a strong folder costs $0.05 per 30k chat.

## Answer for the no-documents case

Yes, the same mechanism helps, and the corpus is the history. For a long chat, folding each exchange into a fact ledger as it
happens cuts the per-question context by 7 to 15x at no accuracy loss, the ledger is checkable against ground truth (47/47),
and pairing it with search over the raw exchanges makes it robust to a weak folder. Rolling summaries do just as well on
spot questions but silently drop about 10% of the facts, which surfaces the moment the context is handed to another agent.
For agent-to-agent communication, hand over the ledger (1.3k tokens, every fact present) instead of the transcript (39k) or an
orchestrator-written brief (a full-context call per delegation). A REPL over the transcript is the wrong tool here: it misses
revisions and invents answers when grep finds nothing. For a short prompt with no history there is nothing to fold and the
approach does not apply.

## Caveats

- Synthetic chats with 70 to 113 crisp facts; real conversations have fuzzier facts and the extraction rule will need tuning
  per domain. `check_ledger` in `lab/chat_facts.py` is the template for measuring that.
- The summary's compaction budget (~700 words) is typical but arbitrary; a larger budget would hold more facts and cost more per
  turn, and it would still be unverifiable.
- Single run per condition; differences under about 5 points are noise. The folding is sequential and non-deterministic, so a
  re-fold can produce a slightly different but equally correct ledger.

---

# Part 4: reproducibility, a self-audit for cheap folders, and messier chats (2026-09-15, later)

Follow-up to Part 3 answering its own caveats: is the folding reproducible, can a weak folder repair its own ledger, and does
any of it survive chats that do not read like a form? Everything below is a stored run in `results/chat_*.jsonl`,
`results/handoff_*.jsonl` and `results/chat_index_costs.jsonl`; tables from `uv run python lab/summary_chat.py`.

## Reproducibility (repeat runs)

- **Folding repeated from scratch** (`--fold_salt rep1`, cache bypassed) on chats 1 to 3: 48/48 ledger fields on every chat,
  $0.050 to $0.051 per chat, 73 to 74 facts. The ledgers differ in wording, not in values.
- **Probes repeated** (`--rep 1`, 285 questions): `chat_state` 133/133, `chat_vc` 132/133 on the first pass and 133/133 after
  the ledger fix below, `chat_summary` 130/133 (one 'original value' miss, one 'no datastore' miss, one PostgreSQL/Postgres
  alias that the grader now accepts). Pooled over both runs: ledger 100%, ledger + tools 100%, summary 98 to 100%.
- **Hand-off repeated**: ledger 195/195 both times, ledger + tools 195/195 both times, summary 176/195 then 175/195.
  The summary's ~10% fact loss is systematic, not noise.

## A ledger defect the field checker missed

The second `chat_vc` run failed one list question on chat1: the agent read `state['Borealis launch']['in_scope_modules']` and got
five modules instead of six. The luna folder had re-emitted the updated list under a drifted attribute name
(`in_scope_modules.value`) and left the original list stale; on the dense chat the list still contained the dropped module.
Part 3's checker only compared per-module fields, so both ledgers had passed as 47/47.

Fix, in `apply_items` in `lab/chat_facts.py`: a deterministic invariant after every folding step. A dropped entity leaves every
membership list; an entity whose `in_scope=true` flag is newer than the list joins it, but only if it shares attributes with the
list's members (a spurious `standup.in_scope=true` from the GLM folder would otherwise have joined the module list, and did, in
an intermediate version). `ci_run.failed_modules` style lists are excluded by name. The checker now scores the membership list
as a field (48 fields, dense 88). After the fix: luna 48/48 on all five chats, and `chat_state`/`chat_vc` 76/76 on chat1 and
dense5 in both repeats. Three more checker false alarms were fixed on the way (numeric strings such as '0.4%', lists serialized
as JSON strings, `repository.name` for the repo).

## Audit pass for a weak folder (`--audit`)

After online folding, the folder re-reads the raw exchanges in chunks of 20 against the finished ledger and reports facts that
are missing or contradicted; a reported item is applied only if its key is absent or no later exchange already revised it.
Same cheap model as the folder, 6 to 10 calls, $0.005 to $0.012 per chat.

| chat | glm-5.3-flash online | after GLM audit | patches |
|---|---|---|---|
| chat1 | 47/48 (on-call order missing) | 48/48 | 1 |
| chat2 | 43/47 (kickoff facts, on-call order missing) | 48/48 | 5 |
| chat3 | 46/48 (config paste missing) | 48/48 | 6 |
| long4 | 48/48 | 48/48 | 0 |
| dense5 | 87/88 (error rate stored as 0.025 for 2.5%) | 87/88 | 0 |

The audit finds omissions, not wrong values that look plausible. Downstream, with the GLM folder plus audit and luna as agent:
`chat_state` 100% / 100% / 100% (was 89% / 100% / 100% without the audit), hand-off `worker_state` 195/195 (was 175/195),
`worker_vc` 195/195. A $0.03 folder plus a $0.01 audit now matches the $0.05 folder. The luna ledgers gained nothing from an
audit (48/48 before and after); on the messy chats it added a few junk CI lines and no real facts.

## Messier chats (`--style messy`)

Same facts, questions, tasks and ground truth as chats 1 to 3, but every fact-bearing user turn was rewritten by luna into an
informal team-chat message (asides about lunch and weather, lowercase, the fact buried mid-message; every name, date and number
verified to survive verbatim, 135/135 rewrites accepted) and the assistant's reply no longer restates the fact ("noted." instead
of "billing-sync is now due Oct 7"), so "pull it in by three days", "same budget as X" and "swap the owners" must be resolved by
whoever reads the chat later.

| condition | accuracy (57 q) | prompt tok/q | notes |
|---|---|---|---|
| chat_full | 96% | 30,784 | misses "same budget as X" and the sum that depends on it |
| chat_repl | 91% | 11,190 | misses "make that $65k" twice and every sum |
| chat_summary | 96% | 4,077 | one sum, one P0 list |
| **chat_state** | **100%** | 4,087 | ledger 48/48 on all three chats, online, no audit |
| **chat_vc** | **100%** | 5,618 | |

Hand-off on the messy chats (6 tasks, 96 facts): ledger 96/96, ledger + tools 96/96, orchestrator brief 96/96, full transcript
94/96 (5/6 tasks), summary 86/96 (1/6 tasks). Folding the messy chats cost the same as the clean ones ($0.051 to $0.052).

## What changed in the conclusion

Nothing reversed; two things sharpened. Resolving references at fold time, while the previous exchanges are still in view, is
worth more than the ledger's size advantage: on the messy chats the full transcript loses to a 4k-token ledger because the
reader has to resolve "same budget as notifier" from 30k tokens of chat at question time. And the ledger's checkability has to
include structure, not just values: a stale list is an error the agent will faithfully reproduce, and only a structural check
(or a deterministic invariant) catches it.

## Caveats

- The "messy" chats are still synthetic: an LLM paraphrase of template sentences. Real transcripts have facts that are implied,
  contradicted without acknowledgement, or split across people; extraction there will be below 100% and the audit pass is the
  cheap way to measure and recover part of that gap.
- Two repeats per condition; differences of one or two questions are within run-to-run variation.
- The audit cannot fix a plausible wrong value (0.025 for 2.5%); typed validation, as in the document pipeline's schema-typed
  fields, is the missing piece for that.

---

# Part 5: wild ideas for shrinking the context further (2026-09-15, night)

Part 3's best chat conditions still spend ~4k tokens per question, and most of that is not the ledger: the ledger is
~1.1k tokens and the ten raw exchanges kept "for recency" are 2.5 to 3.8k. This part throws ten unconventional
reductions at the same benchmark (five clean chats, 95 probes; the same five chats in the messy style; 10 hand-off
tasks with 195 required facts) and keeps what survives. None of the new conditions keeps a raw window. Code:
`lab/conditions_wild.py`; runs: tags `wild`, `wild2` (terse format v2), `wildfix` (race-free rerun of the retriever
conditions), `wildk4`, `wildaudit`, `wildref`; tables from `uv run python lab/summary_wild.py <tag>`.

| condition | what the model sees | fold cost per chat |
|---|---|---|
| `chat_ledger0` | the ledger, nothing else (chat_state with keep=0) | $0.05 (ledger) |
| `chat_terse` | the ledger as one line per entity, `attr=value`, history in brackets | $0.05 |
| `chat_slice` | hybrid top-12 ledger entries for the question, plus every entry whose attribute name is in the question (so sums, counts and lists get all of them), plus the list of entity names | $0.05 |
| `chat_tools0` | nothing in the prompt; ledger and transcript reachable through tools | $0.05 |
| `chat_shorthand` | notes the model wrote for itself in one call over the whole transcript ("fewest tokens you can, any shorthand you can decode") | $0.008 to $0.02 |
| `chat_rag` | single-shot hybrid top-6 raw exchanges (control) | $0 |
| `chat_useronly` | the user's turns only, numbered; assistant replies deleted | $0, no LLM |
| `chat_asstclip` | user turns plus the first sentence of each reply | $0, no LLM |
| `chat_regex` | user sentences containing a number, money, percent, hyphenated identifier or capitalised name; pastes kept whole | $0, no LLM |
| `chat_prune` | user turns with function words removed (a LLMLingua-style token diet done with a stop-word list) | $0, no LLM |

## Probes on the clean chats (5 chats, 95 questions, agent gpt-5.6-luna)

Reference rows from Parts 3 and 4: `chat_full` 99% at 30k to 74k tokens per question, `chat_state` 100% at 4.0k,
`chat_summary` 100% at 4.1k, `chat_vc` 100% at 5.5k.

| condition | accuracy | prompt tok/q | LLM calls/q | notes |
|---|---|---|---|---|
| **`chat_slice`** | **100%** (190/190 over two runs) | **544** | 1 | 1/55 of the transcript on the 29k chats, 1/150 on the 72k chat |
| `chat_slice` @glm-5.3-flash | 100% | 520 | 1 | the cheap agent model is perfect on the sliced ledger ($0.0001/q) |
| `chat_slice` top-4 (`wildk4`) | 96% | 509 | 1 | the four misses are P0 lists: entries not covered by the attribute expansion fall out |
| `chat_terse` | 100% | 799 (v1), 838 (v2) | 1 | |
| `chat_terse` @glm-5.3-flash | 99% | 798 | 1 | one 'original value' miss |
| `chat_ledger0` | 100% | 1,375 | 1 | the raw window was pure cost on these probes |
| `chat_shorthand` | 99% | 940 | 1 | one sum on the 72k chat: the notes dropped a budget change |
| `chat_tools0` | 99% | 2,112 | 2.4 | same sum miss; tool traffic costs more than the sliced ledger |
| `chat_prune` | 100% | 1,949 | 1 | |
| `chat_useronly` | 100% | 2,222 | 1 | |
| `chat_asstclip` | 100% | 4,024 | 1 | |
| `chat_regex` | 89% | 1,019 | 1 | drops "pull it in by three days" and "Lucas will take notifier" |
| `chat_rag` | 83 to 86% | 533 to 823 | 1 | sums, negation and lists need entries the top-6 does not contain |

## Probes on the messy chats (5 chats incl. messy long and dense variants, 95 questions)

The messy long (72k) and dense (113 facts) chats are new in this part. Folding them cost $0.082 and $0.063; the long
one came out with one wrong field (audit-trail's deadline stayed at Oct 16 after a paraphrased "pull it in by three
days" at exchange 161). `--audit` found exactly that field and patched it (47/48 to 48/48, $0.024), the first time the
audit repaired a luna ledger; on the other four messy chats it changed nothing that a probe touches.

| condition | accuracy | prompt tok/q | notes |
|---|---|---|---|
| `chat_state` (ledger + window, reference) | 99% | 4,354 | 57 q from Part 4 plus the two new chats; the miss is the fold error above |
| `chat_summary` (reference) | 98% | 4,192 | one sum, one P0 list |
| **`chat_useronly`** | **100%** | 3,009 | no LLM before question time |
| **`chat_slice`** | 99%, **100% on the audited ledgers** | 514 | the one miss is the fold error above |
| `chat_slice` @glm-5.3-flash | 99% | 519 | the same fold error; the cheap agent matches luna on the sliced ledger |
| `chat_useronly` @glm-5.3-flash | 99% | 3,013 | one P0 list on the 72k chat: the cheap model reading 186 raw turns misses a priority change |
| `chat_ledger0` | 99%, 100% audited | 1,369 | same single miss |
| `chat_terse` v2 | 99% | 836 | same single miss; v1 also lost two "original value" questions because its history notation was ambiguous |
| `chat_asstclip` | 99% | 4,902 | one P0 list |
| `chat_shorthand` | 97% | 985 | a standup time, a P0 flag and a budget change vanished from the notes |
| `chat_prune` | 95% | 2,585 | stop-word removal mangles paraphrased sentences ("make that $65k" style edits) |
| `chat_regex` | 80% | 1,307 | 12 'updated' misses: relative changes and plain-name assignments carry no regex-visible token |

## Hand-off (10 tasks, 195 required facts)

| worker | clean: facts found | tok/task | messy: facts found | tok/task |
|---|---|---|---|---|
| `worker_state` (reference, Part 3/4) | 195/195 | 1,340 | 96/96 on 6 tasks | |
| `worker_terse` | 195/195 | 816 | 194/195 | 814 |
| `worker_slice` (task-sliced ledger, top-30) | 195/195 | 914 | 194/195 | 901 |
| `worker_shorthand` | 195/195 | 950 | 194/195 | 996 |
| `worker_useronly` | 195/195 | 2,230 | 195/195 | 3,017 |
| `worker_regex` | 185/195 (5/10 notes complete) | 1,031 | 181/195 | 1,319 |

The messy misses for terse and slice are the unaudited long4m deadline; the shorthand miss is a P0 flag its notes
never recorded.

## What bit

- **The recency window was dead weight.** Every ledger condition is at 100% without it. Dropping it takes the ledger
  condition from 4.0k to 1.4k tokens per question, and a terse serialisation to 0.8k, with no change in accuracy.
- **Slicing the ledger by question is the new best condition**: 100% on clean and audited messy chats at ~520 tokens
  per question, 55 to 150 times below the transcript and 7 times below the previous best. Two details make it work
  where single-shot RAG fails: a deterministic expansion that pulls in every entry whose attribute the question names
  (sums, counts, lists) and an entity directory so "not stated" questions stay answerable. Dense retrieval alone
  (top-4, no expansion) loses list questions; single-shot RAG over raw exchanges loses sums and negations at 83 to 86%.
- **The cheap agent model becomes perfect on it**: glm-5.3-flash, 100% on the clean sliced ledger at $0.0001 per
  question and 99% on the messy one (the fold error), where its REPL runs in Part 2 were at 77 to 89%. On the raw user-only
  transcript the same model loses a list question that luna gets: folding moves the hard reading out of the cheap model's path.
- **Deleting the assistant is free and lossless on this benchmark.** `chat_useronly` is 100% on clean and messy
  probes and 195/195 on both hand-off sets with no folding at all, at 2 to 3.6k tokens. It works because every fact in
  these chats is user-stated and the assistant only explains; it is a property of the corpus as much as of the idea.
  It also means the honest comparison for the ledger is not the 30k transcript but the 2.2k user-only transcript:
  the ledger's advantage there is 4x on tokens (sliced) plus checkability, not 50x.
- **Surface filters are not extraction.** The regex filter and stop-word pruning are the same idea as LLMLingua-style
  token dropping, and they fail on exactly the facts that matter: relative changes ("three days earlier", "same as X"),
  swaps, and plain-name assignments. 80 to 95% on the messy chats.
- **Self-shorthand is a cheaper summary, not a cheaper ledger.** One call, $0.008 per 30k chat (vs $0.05 for online
  folding), 4x shorter than the rolling summary, and it loses facts the same way a summary does (97 to 99%). Nothing in
  it can be checked against the transcript field by field.
- **A harness bug, found by the cheap model.** Probes of one chat ran in parallel on a shared session object; with a
  retriever call between resetting and appending, two questions could land in one prompt. It only ever surfaced in
  the GLM slice run (4 of 95 answers contained both questions); luna answers were unaffected, and the retriever
  conditions were rerun on the fixed wrapper (`wildfix`, 100% again). A second bug: `--model` also changed the
  paraphrasing model for `--style messy`, so a cheap agent would have been tested on different chats; fixed.

## Caveats

- Single run per condition unless stated; one question is 1% of a table.
- The zero-LLM conditions exploit a corpus where facts live only in user turns. In agent transcripts the assistant's
  decisions and tool outputs carry facts, and user-only would drop them; the ledger fold does not care who said it.
- Slicing was tuned once (k=12 plus attribute expansion) on the same questions it is evaluated on. Question types
  that name no attribute and need many entries ("what is still undecided?") would need a larger k or the expansion
  extended to entity names.
- Spend for this part: ~$1.9, of which ~$0.5 went to folding the two new messy chats and their references.
