# vectorcontext lab

Can vector retrieval do what a "REPL workspace" does for LLM context reduction, and do it more efficiently?
This repo is a reproducible testing lab that measures four families of approaches on the same questions
with the same model (`openai/gpt-5.6-luna` via OpenRouter):

| condition | mechanism |
|---|---|
| `baseline` | whole corpus in the prompt, one call |
| `repl` | corpus is a Python variable in a sandbox; the model greps/slices via a `python` tool and only sees stdout |
| `vector_rag` | classic single-shot RAG: top-8 dense chunks (contextual headers) in the prompt |
| `vectorcontext_v1` | agentic hybrid search (dense + BM25, reciprocal-rank fusion) + `open_doc`, multi-round |
| `vectorcontext_v3` | v1 + a `python` tool where `search()` is callable from code |
| `vectorcontext_v4` | **fact cards**: one schema-first LLM extraction per document at index time (cached, amortized), values canonicalized and entity-linked (aliases to canonical ids); the agent searches cards semantically and aggregates over them in `python` |
| `vectorcontext_v5` | v4 + `search_text` over raw passages (for facts no card carries) + guided strategy prompt + **index-time supersession**: one LLM pass turns change records into patches; cards get a `current` view and `history`, source fields stay untouched (`--resolve`) |

Metrics per question: cumulative prompt tokens (what you pay), peak single-call prompt tokens (context pressure),
completion/reasoning tokens, LLM calls, tool calls, latency, OpenRouter-reported cost, and correctness
(rule-based grading, LLM judge only for ambiguous phrasing).

## Corpus and questions

`lab/corpus.py` generates a synthetic engineering knowledge base (service overviews, incident reports, team pages,
architecture decisions, runbooks) with exact ground truth. Facts are phrased with varied templates and entity aliases
("billing-service" vs "the Billing API") so lexical grep and vector search each have realistic failure modes.
39 questions in four types: needle lookup, multi-hop, aggregation (count/sum), global (list/negation).
The `--scale` knob pads documents with more prose (1≈23k, 3≈45k, 6≈78k, 12≈145k tokens) while keeping the same
facts and questions, so cost scaling can be measured directly.

## Hard benchmark (`--corpus hard`)

`lab/corpus_hard.py` extends the base corpus with 16 change records that supersede overview facts (four are superseded
again by a later record), collateral-impact sentences naming a second service in 40% of incident reports, and question
families built to break each arm: temporal (latest-wins, as-of-date joins, history), prose-only facts (timeline times,
weekdays), negation / set difference / reverse joins, numeric reasoning across joins, distractors (primary vs collateral),
and six four-turn conversations whose follow-ups use pronouns. Ground truth is computed from generator state.
`lab/check_resolve.py` verifies the resolver's `current` view against that state; `lab/check_facts.py` verifies extraction.

## No-documents benchmark (long chats, agent-to-agent hand-off)

`lab/corpus_chat.py` generates project-planning chats in which every fact arrives through the dialogue (revisions by
coreference, a dropped and an added module, pasted tool outputs, tangents as filler), with probe questions and hand-off tasks.

| condition | what the model sees |
|---|---|
| `chat_full` / `chat_window` | whole transcript / last 10 exchanges |
| `chat_summary` | rolling LLM summary (compaction) + last 10 |
| `chat_repl` | last 10 + `python` over the transcript |
| `chat_state` | online **fact ledger** (one extraction call per exchange, conditioned on the ledger so far) + last 10 |
| `chat_vc` | ledger + search over ledger entries and raw exchanges, `open_turn`, `python` |
| `worker_full` / `worker_brief` / `worker_summary` / `worker_state` / `worker_vc` | what a delegated worker agent receives: transcript / orchestrator-written brief / summary / ledger / ledger + tools |

## Run

```bash
cp .env.example .env   # put OPENROUTER_API_KEY=... in .env
uv sync
uv run python lab/run.py --conditions baseline,repl,vector_rag,vectorcontext_v1,vectorcontext_v4 --scale 3
uv run python lab/report.py            # -> results/vectorcontext_results.xlsx
uv run python lab/check_facts.py 3     # fact-card extraction accuracy vs ground truth
```

Hard set and other models:

```bash
uv run python lab/run.py --corpus hard --conditions baseline,repl,vectorcontext_v4 --qtypes temporal,prose,negation,numeric,distractor,multiturn
uv run python lab/run.py --corpus hard --conditions vectorcontext_v5 --resolve --tag v5
uv run python lab/run.py --corpus hard --conditions baseline,repl,vectorcontext_v5 --resolve --model z-ai/glm-5.3-flash
uv run python lab/check_resolve.py 3
```

No-documents benchmark:

```bash
uv run python lab/run_chat.py --prepare_only                      # fold every chat (ledger + summary), check ledgers vs ground truth
uv run python lab/run_chat.py                                     # probes under chat_full,chat_window,chat_summary,chat_repl,chat_state,chat_vc
uv run python lab/run_chat.py --handoff --conditions worker_full,worker_brief,worker_summary,worker_state,worker_vc
uv run python lab/run_chat.py --fold_model z-ai/glm-5.3-flash --tag foldglm --conditions chat_state,chat_vc   # cheap folder
uv run python lab/run_chat.py --model z-ai/glm-5.3-flash --conditions chat_state,chat_vc                     # cheap agent
uv run python lab/summary_chat.py                                 # markdown tables
uv run python lab/run_chat.py --fold_model z-ai/glm-5.3-flash --audit --tag foldglm_audit --conditions chat_state   # cheap folder + self-audit pass
uv run python lab/run_chat.py --style messy --chats chat1,chat2,chat3 --tag messy                                  # LLM-paraphrased chats, non-echoing assistant
uv run python lab/run_chat.py --prepare_only --fold_salt rep1 --chats chat1,chat2,chat3                           # re-fold from scratch (reproducibility)
```

Flags: `--chats chat1,long4,dense5`, `--qtypes early,updated,...`, `--batch 5` (exchanges per folding call), `--fold_model`, `--model`, `--tag`,
`--audit` (ledger audit pass after folding, same model as the folder), `--style messy` (paraphrased user turns, see `lab/chat_messy.py`),
`--fold_salt X` (bypass the folding cache for an independent repeat), `--rep k`.

Useful flags: `--limit N` (questions per type), `--qtypes needle,aggregation` (`multiturn` = conversations), `--qids id,...`,
`--kw '{"mask_keep":1}'` / `'{"prompt":"minimal"}'` / `'{"mask_prev_turns":true}'`, `--chunk_chars 220`, `--tag name`,
`--rep k` (repeat runs are kept separately), `--model` (agent model; extraction and judge stay on gpt-5.6-luna). Every LLM call is metered; `VC_BUDGET_USD` (default 4) aborts a run past that spend.

## Layout

- `lab/corpus.py` synthetic corpus + questions; `lab/corpus_hard.py` change records, distractors, hard questions, conversations
- `lab/llm.py` OpenRouter client, per-episode meter, budget guard
- `lab/sandbox.py` persistent Python REPL with output truncation and timeout
- `lab/index.py` chunking, embedding cache, dense/BM25/hybrid retrieval, fact-card index
- `lab/facts.py` schema induction, per-document extraction, value canonicalization
- `lab/conditions.py` the experimental conditions and the agent loop
- `lab/grade.py` grading; `lab/run.py` runner (single questions and conversations); `lab/report.py` Excel builder
- `lab/check_facts.py`, `lab/check_resolve.py` extraction and supersession accuracy against ground truth
- `lab/corpus_chat.py` chat generator (probes + hand-off tasks); `lab/chat_facts.py` online ledger folding, rolling summaries, ledger check
- `lab/chat_messy.py` messy variant of the chats (LLM paraphrase with verbatim-value check, non-echoing assistant replies)
- `lab/conditions_chat.py` chat conditions; `lab/run_chat.py` chat/hand-off runner; `lab/summary_chat.py` markdown tables
- `docs/literature.md` literature review with verified arXiv ids
- `results/` raw `run_*.jsonl`, `chat_*.jsonl`, `handoff_*.jsonl`, caches, `vectorcontext_results.xlsx`

## Findings

See `docs/findings.md` (written after the final runs) and the Excel workbook.
