# EASE — Experience-aware, Adaptive, Search with Efficient budgeting

A budget-aware retrieval agent that models search as an **economically rational process**:
a skill store guides where to search, a budget ledger constrains every step, and the system
makes an explicit trade-off between **answer quality** and **retrieval cost**.

> Full realism principle: **no mocks, no fake data, no simulated pipelines.** Every answer comes
> from real LLM API calls + real retrieval over an offline HotpotQA corpus (or Tavily web search).

## Results (n=50, same 50 HotpotQA questions, real API)

| method     | EM    | F1    | searches/q | cost/q |
|------------|-------|-------|-----------|--------|
| **EASE(warm)** | **0.640** | **0.763** | 3.46  | $0.0080 |
| EASE(cold) | 0.580 | 0.739 | 3.22      | $0.0074 |
| CoT        | 0.460 | 0.602 | 0.00      | $0.0021 |
| IRCoT      | 0.440 | 0.580 | 1.84      | $0.0025 |
| RAG-Once   | 0.420 | 0.565 | 1.00      | $0.0014 |
| ReAct      | 0.400 | 0.563 | 1.16      | $0.0012 |

Key findings:

- **EASE(warm) significantly beats all four baselines** (McNemar with continuity correction:
  vs ReAct χ²=7.56, vs RAG-Once χ²=6.67, vs CoT χ²=6.23, vs IRCoT χ²=5.06, all p<0.03).
- **Naive retrieval is a net liability in the distractor setting**: RAG-Once / IRCoT / ReAct
  all fall *below* pure CoT (0.46). EASE is the only retrieval system that beats pure
  reasoning — the gain comes from the **control loop** (slot-driven queries, budget/stopping,
  evidence-gated answer commit), not from retrieval itself.
- **28% hard ceiling**: 14/50 questions are missed by all six methods (corpus/format/depth limit).

## Architecture

```
                    ┌─────────────────────────────────────────┐
  Experience        │ SkillStore: skill retrieval surface (vector) + payloads  │
                    │  decomposition templates / budget ratios / query        │
                    │  templates / stopping params; ExpeL lifecycle            │
                    ├─────────────────────────────────────────┤
  Control           │ MetacognitiveManager: coverage graph / gaps / confidence │
                    │ BudgetPlanner: allocation + ledger injection (BATS)      │
                    │ Router: RETRIEVE / REASON / GENERATE / STOP              │
                    │ StopChecker: compound stopping + safe-stop               │
                    ├─────────────────────────────────────────┤
  Execution         │ QueryGenerator → SearchTool → UtilityEvaluator           │
                    │ → EvidenceCompressor → WorkingMemory (dedup)             │
                    └─────────────────────────────────────────┘
```

**Asymmetric model split** (the cost-saving core):

| step | model | thinking | role |
|------|-------|----------|------|
| routing / scoring / compression | small (fast) | off | cheap per-step decisions |
| final answer | large | on | one high-quality generation |

## Key mechanisms

- **Budget-ledger injection**: cost is explicitly visible and drives every decision.
- **Gap-bound retrieval**: no retrieval without a gap — prevents "search a bit more".
- **Three retrieval intents**: `fast_lookup` (low-token) / `evidence_search` (adaptive-k) / `deep_scrape` (deep read).
- **Utility triple-filter**: `score = 0.5·relevance + 0.3·novelty − 0.2·redundancy`; redundant facts (cos>0.95) dropped.
- **Compound stopping + safe-stop**: marginal-gain early stop + coverage/confidence thresholds + safety re-check.
- **Answer-finalization reserve**: search tokens never touch the reserved final-answer budget.

## Quick start

```bash
pip install -r requirements.txt

# 0) Configure keys (project-root .env)
#    DEEPSEEK_API_KEY=sk-...           DeepSeek
#    EASE_SEARCH_BACKEND=offline       offline | web
#    TAVILY_API_KEY=tvly-...           (web backend only)

# 1) Download embedding model (bge-small-en-v1.5, via ModelScope)
python scripts/download_embedding_model.py

# 2) Build offline corpus (HotpotQA dev, 66,581 paragraphs + dense index)
python scripts/build_corpus.py

# 3) Smoke-test each layer (real API + real retrieval)
python scripts/smoke_llm.py          # LLM layer
python scripts/smoke_search.py       # search backends
python scripts/smoke_skillstore.py   # experience layer
python scripts/smoke_control.py      # control layer
python scripts/smoke_execution.py    # execution layer

# 4) End-to-end single-question demo (cold/warm comparison + skill solidification)
python scripts/demo_single.py

# 5) Eval: EASE(cold/warm) vs CoT / RAG-Once / IRCoT / ReAct
python scripts/run_eval.py --n 10                                   # smoke
python scripts/run_eval.py --n 100 --out data/runs/full             # full run (resumable)
```

Run evaluation is resumable: per-question results append to `data/runs/<name>/rows-*.jsonl`
and finished qids are skipped on re-run.

## Eval reports

Each run writes `data/runs/<name>/summary.md` (EM/F1 × efficiency dual-axis) plus per-question
detail in `rows-*.jsonl`. The `scripts/compare_baselines.py` script assembles the 6-way
same-question comparison table (aggregate + paired wins + per-question matrix).

## Repository layout

```
ease/
  llm/          LLM client / cost table / prompts
  embeddings/   local embedder
  search/       search backends (offline corpus / web Tavily)
  experience/   SkillStore + skill extraction + contrast-rule distillation
  control/      metacognition / budget / routing / stopping
  execution/    query generation / utility filtering / compression / working memory
  agent/        EASE main loop / answer generation / traces
  eval/         metrics / data loading / harness / reports
  baselines/    CoT / RAG-Once / IRCoT / ReAct
scripts/        build, smoke, verify, and eval scripts
config/         centralized config (secrets via ${ENV_VAR} placeholders)
```

## Notes

- Evaluation uses the offline-corpus backend (reproducible); web search is for demos only (real but not reproducible).
- Cost is computed from DeepSeek official per-token pricing (`ease/llm/costs.py`).
- All keys live in `.env` only — never committed.
