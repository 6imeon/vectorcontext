# Literature review: full-context vs REPL workspace vs vector retrieval vs hybrid

Compiled 2026-09-14 for the vectorcontext lab. All arXiv ids were resolved against arxiv.org/abs pages.

## A. REPL / code-workspace approaches

**Core mechanism.** The corpus is never placed in the prompt. It is bound to a variable in a sandboxed interpreter; the model emits code that slices, greps, filters, aggregates, or dispatches sub-LLM calls over pieces, and only stdout re-enters context.

**Key references.**
- *Recursive Language Models*, Zhang, Kraska, Khattab (MIT), Dec 2025, arXiv:2512.24601. Prompt stored as a Python variable; root model can "peek, partition, grep, and launch recursive sub-queries." Table 1 (GPT-5 backbone, depth=1), score / mean cost per query:
  - S-NIAH: base GPT-5 24.0, compaction 58.0, CodeAct+BM25 22.0, Claude Code 12.0, RLM 62.0.
  - BrowseComp-Plus, 1K docs: base 0.0 (overflow), compaction 70.5 ($0.57), CodeAct+BM25 51.0 ($0.71), RLM 91.3 ($0.99).
  - OOLONG (trec_coarse): base 44.0 ($0.14), compaction 46.0, CodeAct+BM25 38.0 ($0.61), Claude Code 40.2 ($3.43), RLM 56.0 ($0.43).
  - OOLONG-Pairs (pairwise aggregation, F1): base 0.1, compaction 0.1, CodeAct+BM25 24.7, RLM 58.0 ($0.33).
  - LongBench-v2 CodeQA: base 24.0, compaction 58.0 ($1.31), Claude Code 62.0 ($1.25), RLM 62.0 ($0.11).
  - Abstract: median +26% vs compaction, +130% vs CodeAct-with-sub-calls, +13% vs Claude Code, "comparable cost."
- *Recursive Agent Harnesses*, Lumer et al., 2026, arXiv:2606.13643: parent agent writes a script that spawns parallel sub-harnesses; Oolong-synth (up to 4M tokens): GPT-5 Codex 71.75 -> RAH 81.36.
- *Recursive LMs Meet Uncertainty (SRLM)*, Alizadeh et al. (Apple), 2026, arXiv:2603.15653: up to +22% over RLM at equal time budget. Negative finding: RLM recursion often degrades accuracy relative to the base model when the input already fits in the window; RLMs are "less effective in tasks with semantically intensive nature."
- *Beyond Semantic Similarity: Direct Corpus Interaction (DCI)*, Li et al., 2026, arXiv:2605.05242: agent uses grep/read/shell over the raw corpus, no index. BrowseComp-Plus: DCI-Agent (Claude Sonnet 4.6) 80.0 vs 69.0 for the same model with Qwen3-Embedding-8B retrieval, at 29% lower cost. Failure modes: ~35 tool calls/question vs 17-19 for retrieval; gold-doc coverage 28% vs 57%; at 400K docs accuracy fell to 37.5% with budget exhaustion.
- *CodeAct*, Wang et al., ICML 2024, arXiv:2402.01030: code actions vs JSON: up to +20% success, up to 30% fewer actions. *ReAct*, Yao et al. 2022, arXiv:2210.03629.
- Anthropic, *Code execution with MCP* (Nov 2025): tool results stay in the sandbox; worked example 150,000 -> 2,000 tokens (98.7%). Anthropic, *Effective context engineering* (Sept 2025): "context rot"; just-in-time loading via grep/glob; compaction; sub-agents returning 1-2K-token summaries. Manus (2025): file system as restorable context; ~100:1 input:output ratio. Cognition (2025, "Don't build multi-agents"): share full traces.

**Failure modes.** Extra tool-call turns and latency; heavy-tailed cost (RLM sd >= mean on BrowseComp+/OOLONG); recursion hurts when input fits; weak on semantic/fuzzy queries where lexical grep misses paraphrase; corpus-scale collapse without an index.

## B. Vector / embedding retrieval and compression

**Core mechanism.** Chunk, embed, top-k by similarity (optionally hybrid BM25 + dense, ColBERT arXiv:2004.12832, or cross-encoder rerank), stuff k chunks into the prompt. Memory systems: MemGPT arXiv:2310.08560; Mem0 arXiv:2504.19413; A-MEM arXiv:2502.12110; HippoRAG arXiv:2405.14831 / HippoRAG 2 arXiv:2502.14802; Zep arXiv:2501.13956. Compression: LLMLingua arXiv:2310.05736; LongLLMLingua arXiv:2310.06839; LLMLingua-2 arXiv:2403.12968; Selective Context arXiv:2310.06201; gist tokens arXiv:2304.08467.

**Quantitative results.**
- Li et al., *RAG or Long-Context LLMs?* (Self-Route), EMNLP 2024, arXiv:2407.16833: LC beats RAG on average (Gemini-1.5-Pro LC 49.70 vs RAG 37.33; GPT-4o 48.67 vs 32.60); Self-Route ~= LC using 38-61% of LC tokens. RAG failure taxonomy: multi-step reasoning, general/vague queries, long complex queries, implicit queries.
- Yu, Xu, Akkiraju, *In Defense of RAG* (OP-RAG), 2024, arXiv:2409.01666: order-preserving retrieval on InfiniteBench: EN.QA F1 47.25 at 48K tokens vs 34.26 for full 117K context. Inverted-U in accuracy vs number of chunks.
- LaRA, ICML 2025, arXiv:2502.09977: "no silver bullet"; winner depends on model size, context length, task type. arXiv:2501.01880: LC wins QA; summary-based retrieval ~= LC, chunk-based lags.
- Anthropic *Contextual Retrieval* (Sept 2024): top-20 retrieval failure 5.7% -> 3.7% (contextual embeddings) -> 2.9% (+contextual BM25) -> 1.9% (+rerank, -67%).
- RAPTOR, ICLR 2024, arXiv:2401.18059: summary tree; QuALITY 82.6% with GPT-4 vs prior 62.3%. LongRAG, arXiv:2406.15319: 4K-token units, NQ EM 62.7, HotpotQA 64.3. Late chunking, arXiv:2409.04701.
- HippoRAG 2 vs NV-Embed-v2: MuSiQue F1 44.8 -> 51.9. Mem0: >90% fewer tokens than full-context on LOCOMO. Zep: LongMemEval up to +18.5% accuracy, -90% latency.
- Compression: LLMLingua up to 20x with small loss; LongLLMLingua +17.1% on NQ at 4x compression; Selective Context 50% token cut for -0.023 BERTScore.
- BrowseComp-Plus, Chen et al. 2025, arXiv:2508.06600: retriever quality dominates: GPT-5 55.9% with BM25 vs 70.1% with Qwen3-Embed-8B; Recall@5 BM25 1.4% vs dense 18.5%; oracle docs give gpt-4.1 93.5%.
- Lost in the Middle, Liu et al. 2023, arXiv:2307.03172: U-shaped position curve. RULER, arXiv:2404.06654: only 4 of 10 claimed-32K models held up. LongBench arXiv:2308.14508; InfiniteBench arXiv:2402.13718; OOLONG, Bertsch et al. 2025, arXiv:2511.02817: at 128K every frontier model <50%; bottleneck is aggregation, not per-item classification.

**Failure modes.** Multi-hop (bridge entity unknown at query time); aggregation/counting/"all X" queries (top-k caps recall by construction); global summarization; exact-lexical constraints; chunk-boundary context loss; distractor-induced inverted-U as k grows; stale index on mutable data.

## C. Hybrids

- *Context-Folding*, Sun et al. (ByteDance/CMU), ICML 2026, arXiv:2510.11967: branch/return ops fold sub-trajectories into summaries. BrowseComp-Plus: fold 62.0 (32K active) vs ReAct-32K 28.6, ReAct-327K 47.8. Active context 10x smaller. FoldAct arXiv:2512.22733.
- AgentFold arXiv:2510.24699; ReSum arXiv:2509.13313 (+4.5 over ReAct); MemAgent arXiv:2507.02259 (>95% on 512K RULER); Sculptor arXiv:2508.04664; Dynamic Cheatsheet arXiv:2504.07952; Masking stale observations arXiv:2606.00408; ContextBudget arXiv:2604.01664.
- Retrieval-as-tool inside a REPL is measured in the RLM paper's CodeAct+BM25 baseline: 51.0 BrowseComp+, 38.0 OOLONG, 24.7 OOLONG-Pairs. No published paper isolates dense search as a REPL tool; DCI is the closest (grep-as-tool); BrowseComp-Plus quantifies dense vs BM25 as an agent tool (+14 pts). Iterative retrieval: IRCoT arXiv:2212.10509 (+15 QA on multi-hop), Search-R1 arXiv:2503.09516.

## D. Temporal facts and multi-turn memory (added for the hard benchmark)

- *Zep: A Temporal Knowledge Graph Architecture for Agent Memory*, Rasmussen et al. 2025, arXiv:2501.13956 (verified): facts in the Graphiti graph carry validity intervals and are invalidated by later contradicting facts rather than overwritten. 94.8% vs 93.4% (MemGPT) on DMR; up to +18.5% on LongMemEval. Our v5 resolver borrows the pattern: change records patch a `current` view on the target fact card and append `history`, while the card's own fields keep what the source document states (provenance).
- *LongMemEval*, Wu et al. ICLR 2025, arXiv:2410.10813 (verified): commercial assistants and long-context models lose about 30% accuracy across sustained multi-session interaction; decomposing sessions and fact-augmented keys help. Our conversation set (pronoun follow-ups, four turns) measures per-turn accuracy and context growth for each arm, and tests eliding earlier turns' tool outputs.

## Synthesis: predictions for the four arms

| Query type | Full context | REPL workspace | Vector top-k | Hybrid (vector tool in REPL) |
|---|---|---|---|---|
| Needle lookup | Good until context rot; fails if corpus > window | Strong if a lexical handle exists; weak on paraphrase | Strong; contextual+hybrid+rerank ~98% top-20 recall | Semantic recall + exact verification |
| Multi-hop | Best when it fits | Good but many turns | Weakest single-shot; needs iteration | Iterative search from code loop recovers most of the gap |
| Aggregation/counting | Fails as N grows | Clear winner | Fails structurally | Works only if the tool supports exhaustive scans, then code aggregates |
| Global summarization | Best if it fits | Map-reduce via sub-calls, costly | Poor; RAPTOR partially fixes | RAPTOR summaries + REPL reduce |
| Prompt tokens | 100% | 1-5% per turn x turns; high variance | 10-40% | Between; DCI -29% cost vs retrieval |
| Cost tail | Predictable | Heavy-tailed | Predictable | Moderate; index bounds turn count |

## Algorithmic ideas to make vector retrieval competitive

1. Iterative, reasoning-interleaved retrieval (IRCoT, Search-R1): retrieve, reason, re-query with the bridge entity.
2. Query decomposition + parallel sub-queries; for count/all questions convert to exhaustive retrieval via metadata/structured filters, then aggregate in code (retrieval-then-REPL).
3. Contextual chunk headers + hybrid BM25/dense + reranker; parent-child chunking; order-preserving assembly.
4. RAPTOR summary trees indexed alongside leaves; LongRAG-style large units.
5. HyDE (arXiv:2212.10496) for implicit queries; HippoRAG-style entity graph for hops.
6. Adaptive k and self-routing: cheap "is the retrieved set sufficient?" check, escalate to full scan or larger k.
