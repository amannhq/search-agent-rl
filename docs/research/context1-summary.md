# Chroma Context-1: Paper Summary

> **Source:** https://www.trychroma.com/research/context-1  
> **Authors:** Hammad Bashir, Kelly Hong (Chroma), Patrick Jiang, Zhiyi Shi (UIUC)  
> **Date:** March 26, 2026  
> **Note:** This is a summary for research reference. See original for complete details.

---

## Executive Summary

Chroma Context-1 is a **20B parameter agentic search model** trained to perform multi-hop document retrieval. It achieves performance comparable to frontier LLMs (GPT-5.x, Claude Opus/Sonnet) at a fraction of the cost (~10x faster inference). The model operates as a **retrieval subagent** that returns ranked documents to a downstream reasoning model.

**Key Innovation:** The model is trained to **self-edit its context** by selectively pruning irrelevant documents during search, enabling efficient long-horizon retrieval within bounded context windows.

---

## 1. Problem Statement

### Limitations of Single-Shot Retrieval
- Traditional RAG assumes information can be retrieved in one pass
- Many real-world queries require **multi-hop retrieval** where outputs of one search inform the next
- Frontier models for agentic search are expensive and have high latency

### Context Window Challenge
- As agents gather information, context fills with tangential/redundant documents
- "Bloated context" increases cost and degrades performance
- Phenomenon known as **"context rot"**

---

## 2. Key Techniques

### 2.1 Staged Training Curriculum
- First optimizes for **recall** (find all relevant docs)
- Then shifts toward **precision** (be selective)
- Uses F-beta score with β annealing: starts at β=4 (recall-heavy), ends at β=1

### 2.2 Self-Editing Context Management
- Agent has `prune_chunks(chunk_ids)` tool to remove irrelevant passages
- Frees context capacity for further exploration
- Reduces effects of context rot
- **Prune accuracy:** 0.941 (vs 0.824 for base model)

### 2.3 Synthetic Task Generation Pipeline
- Generates multi-hop questions across 4 domains
- Uses LLM judge for verification (>80% alignment with human labelers)
- Publicly released: https://github.com/chroma-core/context-1-data-gen

---

## 3. Agent Architecture

### 3.1 Observe-Reason-Act Loop
The agent operates in a loop:
1. **Observe:** Receive tool result or initial prompt
2. **Infer:** Model produces tool call or final response  
3. **Act:** Execute tool, get observation

### 3.2 Tools Available

| Tool | Description |
|------|-------------|
| `search_corpus(query)` | Hybrid BM25 + dense vector search via RRF. Top 50 → reranked → token budget |
| `grep_corpus(pattern)` | Regex search over corpus. Returns up to 5 matching chunks |
| `read_document(doc_id)` | Read full document. Chunks reranked and truncated to fit budget |
| `prune_chunks(chunk_ids)` | Removes specified chunks from conversation context |

### 3.3 Token Budget Management

- **Continuous visibility:** Token usage appended after every turn (e.g., `[Token usage: 14,203/32,768]`)
- **Soft threshold:** At 75% usage, suggests pruning or concluding
- **Hard cutoff:** Between 90-95%, only prune or conclude allowed
- **Deduplication:** Tracks all chunk IDs seen, excludes from future searches

---

## 4. Synthetic Task Generation Pipeline

### 4.1 Pipeline Structure (Explore → Verify → Extend)

```
1. GATHER DOCUMENTS
   - Given seed topic, agent explores and collects docs with unique facts
   
2. GENERATE TASK  
   - Create clues (obfuscated references to facts)
   - Generate question combining clues
   - Generate answer
   
3. VERIFY
   - Extract document quotes and clue quotes
   - Fuzzy match to confirm quotes exist in source
   - Filter tasks where quotes don't match
   
4. ADD DISTRACTORS (optional)
   - Find docs that satisfy some criteria but wrong answer
   - Verify distractors don't contain the answer
   
5. CHAIN (optional)
   - Bridge answer to new task with new final answer
   - Controls number of hops required
```

### 4.2 Four Domains

| Domain | Source | Characteristics |
|--------|--------|-----------------|
| **Web** | Web pages via search | BrowseComp-style, up to 4 hops |
| **Finance** | SEC filings (2025) | Recent data to minimize contamination, up to 3 hops |
| **Legal** | USPTO patent office actions | Natural multi-doc reasoning (prior art citations) |
| **Email** | Epstein files + Enron corpus | Informal language, abbreviations, implicit context |

### 4.3 Verification via Quote Extraction

For each supporting document:
1. LLM extracts **document quotes** (verbatim spans from source)
2. LLM extracts **clue quotes** (corresponding spans from generated clues)
3. Normalize both (lowercase, strip whitespace)
4. Confirm document quotes actually appear in source
5. Filter tasks where any supporting doc lacks matching quotes

**Result:** >80% alignment accuracy with human labelers

---

## 5. Training Methodology

### 5.1 Base Model
- **gpt-oss-20B** (open source)
- Selected for: fast inference under MXFP4 quantization, strong oracle retrieval performance
- LoRA adaptation

### 5.2 Supervised Fine-Tuning (SFT)

**Purpose:** Learn well-formed tool calls, prompt format, behavior priors (parallel tool calling, query decomposition)

**Data Generation:**
- Run agent loop with large models (Kimi K2.5) as inference backend
- Filter by recall metrics:
  - High recall (>50% trajectory, >40% output): kept in full
  - Lower recall: included at diminishing rate
  - Zero recall: up to 5% as negative examples
  - High trajectory but low output recall: excluded (bad selection behavior)

### 5.3 Reinforcement Learning

**Algorithm:** CISPO (Clipped Importance-Sampled Policy Optimization)
- Variant of GRPO
- Clips importance sampling weights rather than surrogate objective
- Critical for preventing entropy collapse

**Training Setup:**
- 128 queries per step, 8 rollouts each = 1,024 trajectories per step
- 4 substeps of gradient descent per training step
- ~300 total steps, convergence around step 230
- Trained on legal, patent, and web domains only

### 5.4 Reward Function

```
R = R_trajectory + R_answer + R_efficiency - penalties

Where:
- R_trajectory = F_β(precision, recall)  
- R_answer = +1.0 if chunk contains final answer
- Penalties:
  - Repeated pruning: 0.1 per excess call beyond 3 consecutive (cap 0.5)
  - Turn count: Linear from 0 at 64 turns to 0.5 at 128 turns
```

### 5.5 Curriculum Learning

**Difficulty Curriculum:**
- Phase 1: Skewed toward lower-difficulty questions
- Phase 2: Shift toward higher-difficulty multi-hop tasks

**Reward Curriculum (β annealing):**
- Start: β=4 (recall 16x more than precision) - encourages broad exploration
- End: β=1 (balanced) - encourages selectivity

---

## 6. Evaluation Metrics

### Output-Level Metrics
| Metric | Description |
|--------|-------------|
| **Final Answer Found** | Document containing answer appeared in output |
| **Recall** | Fraction of positive docs in output / total positive docs |
| **Precision** | Fraction of output docs that are relevant |
| **F1** | Harmonic mean of recall and precision |

### Trajectory-Level Metric
| Metric | Description |
|--------|-------------|
| **Trajectory Recall** | Fraction of target docs encountered at any point during search |

**Key Insight:** Comparing trajectory recall to output recall reveals whether agent found relevant docs but failed to include them, or missed them entirely.

---

## 7. Results Summary

### Model Comparison (Final Answer Found)

| Model | Web | Finance | Legal | Email |
|-------|-----|---------|-------|-------|
| **Context-1 (4x)** | 0.97 | 0.82 | 0.95 | 0.98 |
| **Context-1 (1x)** | 0.88 | 0.64 | 0.89 | 0.92 |
| gpt-oss-20b (base) | 0.58 | 0.42 | 0.58 | 0.75 |
| gpt-5.4 | 0.97 | 0.67 | 0.95 | 0.97 |
| opus-4.5 | 0.99 | 0.82 | 0.90 | 0.98 |

### Context-1 vs Base Model

| Metric | gpt-oss-20b | Context-1 | Improvement |
|--------|-------------|-----------|-------------|
| Trajectory Recall | 0.640 | 0.739 | +15% |
| Output Recall | 0.361 | 0.641 | +78% |
| F1 | 0.307 | 0.487 | +59% |
| Final Answer Found | 0.541 | 0.798 | +47% |

### Behavioral Improvements

| Behavior | Base Model | Context-1 |
|----------|------------|-----------|
| Tool calls/turn | 1.52 | 2.56 |
| Turns/trajectory | 6.7 | 5.2 |
| Prune accuracy | 0.824 | 0.941 |

### Cost & Latency

- Context-1 achieves **10x faster inference** than frontier models
- Running 4x parallel rollouts with RRF fusion still cheaper than larger models
- Inference: 400-500 tok/s on Nvidia B200 with MXFP4 quantization

---

## 8. Key Insights for RL Environment Design

### From This Paper

1. **Trajectory recall as training signal:** Reward exploration even if docs are later pruned
2. **β-curriculum:** Start recall-heavy, anneal toward precision
3. **Token budget as constraint:** Soft/hard thresholds create action space narrowing
4. **Prune as a learnable skill:** Explicit tool for context management
5. **Deduplication:** Prevent re-retrieval of same chunks

### Reward Formula to Implement

```python
def reward(retrieved, gold, answer_found, steps, max_steps, beta=1.0):
    # F-beta score
    precision = len(retrieved & gold) / len(retrieved) if retrieved else 0
    recall = len(retrieved & gold) / len(gold) if gold else 0
    
    if precision + recall == 0:
        f_beta = 0
    else:
        beta_sq = beta ** 2
        f_beta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
    
    # Answer bonus
    answer_bonus = 1.0 if answer_found else 0.0
    
    # Turn penalty (linear from 64 to 128)
    turn_penalty = max(0, min(0.5, (steps - 64) / 128 * 0.5))
    
    return f_beta + answer_bonus - turn_penalty
```

---

## 9. Resources

- **Model Weights:** https://huggingface.co/chromadb/context-1 (Apache 2.0)
- **Data Gen Code:** https://github.com/chroma-core/context-1-data-gen
- **Chroma Cloud:** Used for scalable search infrastructure during training

---

## 10. Future Directions (from paper)

1. **Scaling model size** - Current 20B, room to scale
2. **Cross-domain transfer** - Train on more domains
3. **End-to-end training** - Joint retrieval + reasoning
4. **Better context management** - Beyond simple pruning
5. **Longer horizon tasks** - More complex multi-hop chains

---

*This summary was created for research reference. For complete details, methodology, and results, please refer to the original paper at https://www.trychroma.com/research/context-1*
