# Search RL Environment Design Document

## Overview

This document describes the design of a **Search Reinforcement Learning Environment** for training agents to perform multi-hop information retrieval tasks. The design is inspired by Chroma's Context-1 data generation pipeline and the accompanying technical report on training agentic search models.

## Goals

1. **Train search agents** that can efficiently navigate large document corpora
2. **Support multi-hop reasoning** where answers require synthesizing information from multiple sources
3. **Provide configurable reward signals** that balance precision and recall
4. **Enable curriculum learning** from simple to complex retrieval tasks

---

## 1. Environment Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Search RL Environment                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │    Agent     │───▶│   Actions    │───▶│   Document Corpus    │  │
│  │   (Policy)   │    │              │    │                      │  │
│  └──────────────┘    │ • search()   │    │ • Hybrid Index       │  │
│         ▲            │ • read()     │    │   (Dense + BM25)     │  │
│         │            │ • prune()    │    │ • Reranker           │  │
│         │            │ • answer()   │    │ • Chunk Store        │  │
│  ┌──────────────┐    └──────────────┘    └──────────────────────┘  │
│  │ Observation  │                                                   │
│  │              │◀──────────────────────────────────────────────────│
│  │ • Context    │                                                   │
│  │ • Retrieved  │    ┌──────────────────────────────────────────┐  │
│  │ • Budget     │    │              Reward Function              │  │
│  └──────────────┘    │                                          │  │
│         ▲            │  R = α·F_β(trajectory) + γ·answer_bonus  │  │
│         │            │                                          │  │
│         └────────────│  β curriculum: high→low (recall→precision)│  │
│                      └──────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. State Space

The **state** represents everything the agent knows at a given timestep.

### 2.1 State Components

```python
@dataclass
class SearchState:
    # Task definition
    task_id: str                          # Unique episode identifier
    question: str                         # The multi-hop question to answer
    
    # Retrieved context (agent's working memory)
    context_chunks: List[Chunk]           # Currently held document chunks
    context_token_count: int              # Total tokens in context
    
    # Search history
    queries_issued: List[str]             # All search queries made
    documents_read: List[str]             # Document IDs that have been read
    
    # Budget tracking
    step_count: int                       # Current step number
    max_steps: int                        # Maximum allowed steps
    max_context_tokens: int               # Context window budget
    
    # Hidden ground truth (for reward calculation, not visible to agent)
    _gold_chunks: List[str]               # Ground truth chunk IDs
    _gold_answer: str                     # Ground truth answer
```

### 2.2 Chunk Representation

```python
@dataclass
class Chunk:
    chunk_id: str                         # Unique identifier
    document_id: str                       # Parent document ID
    content: str                          # Text content
    token_count: int                      # Token count for budget
    metadata: Dict[str, Any]              # Source info, position, etc.
    
    # Search metadata (populated when retrieved)
    retrieval_score: Optional[float]      # Score from retrieval system
    rerank_score: Optional[float]         # Score from reranker
```

---

## 3. Action Space

The agent can take **four types of actions**:

### 3.1 Search Action

Issue a search query to the retrieval system.

```python
@dataclass
class SearchAction:
    query: str                            # Natural language search query
    top_k: int = 10                       # Number of results to retrieve
```

**Returns:** List of chunk summaries (not full content) with scores

### 3.2 Read Action

Read the full content of specific documents/chunks.

```python
@dataclass
class ReadAction:
    chunk_ids: List[str]                  # Chunks to read in full
```

**Effect:** Adds full chunk content to context (consumes token budget)

### 3.3 Prune Action

Remove chunks from context to free up token budget.

```python
@dataclass
class PruneAction:
    chunk_ids: List[str]                  # Chunks to remove from context
```

**Effect:** Removes chunks, frees token budget

### 3.4 Answer Action

Submit final answer and end episode.

```python
@dataclass
class AnswerAction:
    answer: str                           # The agent's final answer
    supporting_chunk_ids: List[str]       # Chunks used to support answer
```

**Effect:** Episode terminates, final reward calculated

### 3.5 Action Space Summary

```python
class ActionType(Enum):
    SEARCH = "search"
    READ = "read"
    PRUNE = "prune"
    ANSWER = "answer"

@dataclass
class Action:
    action_type: ActionType
    payload: Union[SearchAction, ReadAction, PruneAction, AnswerAction]
```

---

## 4. Observation Space

After each action, the agent receives an observation.

### 4.1 Observation Structure

```python
@dataclass
class SearchObservation:
    # Current context state
    context_chunks: List[ChunkSummary]    # Summaries of held chunks
    context_token_count: int              # Current token usage
    context_token_budget: int             # Remaining token budget
    
    # Action result
    action_result: ActionResult           # Result of last action
    
    # Episode progress
    step_count: int
    max_steps: int
    done: bool
    
    # Reward signal
    reward: float
    
    # The original question (always visible)
    question: str
```

### 4.2 Action Results

```python
@dataclass
class SearchResult:
    """Result of a search action"""
    query: str
    results: List[ChunkSummary]           # Title, snippet, score (not full content)
    total_found: int

@dataclass
class ReadResult:
    """Result of a read action"""
    chunks: List[Chunk]                   # Full chunk content
    tokens_added: int
    budget_exceeded: bool                 # True if couldn't fit all chunks

@dataclass
class PruneResult:
    """Result of a prune action"""
    chunks_removed: int
    tokens_freed: int

@dataclass  
class AnswerResult:
    """Result of an answer action (episode end)"""
    answer_submitted: str
    final_reward: float
    trajectory_recall: float              # % of gold chunks retrieved
    trajectory_precision: float           # % of retrieved that were gold
    answer_correct: bool                  # Answer quality metric
```

---

## 5. Reward Function

The reward function is **critical** for training effective search agents. Based on the Context-1 technical report, we use a multi-component reward.

### 5.1 Core Reward Formula

```
R_total = R_trajectory + R_answer + R_efficiency
```

### 5.2 Trajectory Reward (F-beta Score)

Measures how well the agent's retrieved context covers the gold evidence.

```python
def trajectory_reward(retrieved_chunks: Set[str], gold_chunks: Set[str], beta: float) -> float:
    """
    F-beta score for trajectory quality.
    
    beta > 1: Emphasizes recall (early training)
    beta < 1: Emphasizes precision (late training)
    beta = 1: Balanced F1
    """
    if not retrieved_chunks:
        return 0.0
    
    true_positives = len(retrieved_chunks & gold_chunks)
    precision = true_positives / len(retrieved_chunks) if retrieved_chunks else 0
    recall = true_positives / len(gold_chunks) if gold_chunks else 0
    
    if precision + recall == 0:
        return 0.0
    
    beta_sq = beta ** 2
    f_beta = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)
    
    return f_beta
```

### 5.3 Answer Reward

Bonus for correct/high-quality answers.

```python
def answer_reward(predicted: str, gold: str, method: str = "llm_judge") -> float:
    """
    Reward for answer quality.
    
    Methods:
    - "exact": Exact string match (0 or 1)
    - "fuzzy": Fuzzy string similarity (0 to 1)
    - "llm_judge": LLM-based evaluation (0 to 1)
    """
    if method == "exact":
        return 1.0 if predicted.strip().lower() == gold.strip().lower() else 0.0
    elif method == "fuzzy":
        return fuzz.ratio(predicted.lower(), gold.lower()) / 100.0
    elif method == "llm_judge":
        # Use LLM to judge answer quality
        return llm_judge_answer(predicted, gold)
```

### 5.4 Efficiency Reward

Encourages efficient use of budget.

```python
def efficiency_reward(steps_used: int, max_steps: int, tokens_used: int, max_tokens: int) -> float:
    """
    Small bonus for efficiency.
    """
    step_efficiency = 1 - (steps_used / max_steps)
    token_efficiency = 1 - (tokens_used / max_tokens)
    
    return 0.1 * (step_efficiency + token_efficiency) / 2
```

### 5.5 Curriculum via Beta Scheduling

The key insight from Context-1 is using **beta scheduling** for curriculum learning:

```python
class BetaScheduler:
    """
    Schedule beta parameter for F-beta score during training.
    
    High beta (e.g., 2.0) → Emphasizes recall → Agent learns to find relevant docs
    Low beta (e.g., 0.5) → Emphasizes precision → Agent learns to be selective
    """
    
    def __init__(self, start_beta: float = 2.0, end_beta: float = 0.5, decay_steps: int = 10000):
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.decay_steps = decay_steps
    
    def get_beta(self, step: int) -> float:
        progress = min(step / self.decay_steps, 1.0)
        return self.start_beta + (self.end_beta - self.start_beta) * progress
```

### 5.6 Step-wise Intermediate Rewards (Optional)

For denser reward signal during training:

```python
def step_reward(action: Action, state: SearchState) -> float:
    """
    Small intermediate rewards to shape behavior.
    """
    reward = 0.0
    
    # Reward for finding relevant chunks
    if action.action_type == ActionType.READ:
        new_relevant = count_relevant_chunks(action.payload.chunk_ids, state._gold_chunks)
        reward += 0.1 * new_relevant
    
    # Small penalty for redundant searches
    if action.action_type == ActionType.SEARCH:
        if is_redundant_query(action.payload.query, state.queries_issued):
            reward -= 0.05
    
    return reward
```

---

## 6. Document Corpus & Retrieval System

### 6.1 Corpus Structure

```python
@dataclass
class DocumentCorpus:
    """
    The document corpus the agent searches over.
    """
    documents: Dict[str, Document]        # doc_id -> Document
    chunks: Dict[str, Chunk]              # chunk_id -> Chunk
    
    # Indexes
    dense_index: DenseIndex               # Embedding-based search
    sparse_index: SparseIndex             # BM25 search
    
    # Optional
    reranker: Optional[Reranker]          # Cross-encoder reranker
```

### 6.2 Hybrid Retrieval (from Context-1)

```python
class HybridRetriever:
    """
    Combines dense and sparse retrieval with RRF fusion.
    """
    
    def __init__(
        self,
        dense_index: DenseIndex,
        sparse_index: SparseIndex,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
    
    def search(self, query: str, top_k: int = 10) -> List[ChunkSummary]:
        # Get results from both indexes
        dense_results = self.dense_index.search(query, top_k * 2)
        sparse_results = self.sparse_index.search(query, top_k * 2)
        
        # RRF fusion
        fused = self._rrf_fusion(dense_results, sparse_results)
        
        return fused[:top_k]
    
    def _rrf_fusion(self, dense: List, sparse: List) -> List:
        """Reciprocal Rank Fusion"""
        scores = {}
        
        for rank, result in enumerate(dense):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0)
            scores[result.chunk_id] += self.dense_weight / (self.rrf_k + rank + 1)
        
        for rank, result in enumerate(sparse):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0)
            scores[result.chunk_id] += self.sparse_weight / (self.rrf_k + rank + 1)
        
        # Sort by fused score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [self._get_summary(chunk_id, scores[chunk_id]) for chunk_id in sorted_ids]
```

### 6.3 Optional Reranker

```python
class CrossEncoderReranker:
    """
    Rerank results using a cross-encoder model.
    Context-1 uses Qwen 3 8B for this.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query: str, chunks: List[ChunkSummary], top_k: int) -> List[ChunkSummary]:
        pairs = [(query, chunk.snippet) for chunk in chunks]
        scores = self.model.predict(pairs)
        
        reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in reranked[:top_k]]
```

---

## 7. Episode Generation

### 7.1 Task Structure

```python
@dataclass
class SearchTask:
    """
    A single search task/episode.
    """
    task_id: str
    question: str                         # The multi-hop question
    gold_answer: str                      # Ground truth answer
    gold_chunks: List[str]                # Chunk IDs containing evidence
    difficulty: str                       # "easy", "medium", "hard"
    num_hops: int                         # Number of reasoning hops required
    domain: str                           # "web", "sec", "patents", "email"
    
    # Optional metadata
    reasoning_chain: Optional[List[str]]  # Step-by-step reasoning
    distractor_chunks: Optional[List[str]]  # Hard negative chunks
```

### 7.2 Task Generation Pipeline (from Context-1)

The Context-1 pipeline uses **Explore → Verify → Extend** pattern:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Task Generation Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. EXPLORE: Agent explores corpus, discovers facts                 │
│     ┌─────────┐    ┌─────────┐    ┌─────────┐                       │
│     │ Search  │───▶│  Read   │───▶│ Extract │                       │
│     │ (LLM)   │    │ (LLM)   │    │ Facts   │                       │
│     └─────────┘    └─────────┘    └─────────┘                       │
│                                                                     │
│  2. VERIFY: Validate facts against source documents                 │
│     ┌──────────────────┐    ┌─────────────────┐                     │
│     │ Extract Quotes   │───▶│ Fuzzy Match     │                     │
│     │ (LLM)            │    │ (algorithmic)   │                     │
│     └──────────────────┘    └─────────────────┘                     │
│                                                                     │
│  3. EXTEND: Create multi-hop questions from facts                   │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │
│     │ Bridge Facts │───▶│ Generate Q   │───▶│ Add          │        │
│     │ (LLM)        │    │ (LLM)        │    │ Distractors  │        │
│     └──────────────┘    └──────────────┘    └──────────────┘        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 LLM Calls in Pipeline

| Step | LLM Call | Purpose |
|------|----------|---------|
| Explore | `explore_agent.run()` | Issue searches, decide what to read |
| Explore | `extract_facts()` | Extract factual claims from documents |
| Verify | `extract_quotes()` | Extract supporting quotes for facts |
| Verify | `fuzzy_match()` | Algorithmic verification (no LLM) |
| Extend | `find_bridge()` | Find connections between facts |
| Extend | `generate_question()` | Generate multi-hop question |
| Extend | `generate_answer()` | Generate gold answer |
| Distract | `find_distractors()` | Find topically similar wrong docs |

---

## 8. Environment Configuration

### 8.1 Config Schema

```python
@dataclass
class SearchEnvConfig:
    """Configuration for the Search RL Environment."""
    
    # Corpus settings
    corpus_path: str                      # Path to document corpus
    index_type: str = "hybrid"            # "dense", "sparse", "hybrid"
    use_reranker: bool = True
    
    # Budget settings
    max_steps: int = 20                   # Max actions per episode
    max_context_tokens: int = 8192        # Context window budget
    
    # Reward settings
    beta: float = 1.0                     # F-beta parameter
    answer_reward_weight: float = 1.0
    efficiency_reward_weight: float = 0.1
    use_intermediate_rewards: bool = False
    
    # Retrieval settings
    search_top_k: int = 10                # Results per search
    dense_weight: float = 0.5             # Weight for dense retrieval
    sparse_weight: float = 0.5            # Weight for sparse retrieval
    
    # Task settings
    task_difficulty: str = "mixed"        # "easy", "medium", "hard", "mixed"
    min_hops: int = 1
    max_hops: int = 3
```

---

## 9. API Design

### 9.1 Environment Interface

```python
class SearchEnvironment(Environment):
    """
    Search RL Environment implementing OpenEnv interface.
    """
    
    def __init__(self, config: SearchEnvConfig):
        self.config = config
        self.corpus = load_corpus(config.corpus_path)
        self.retriever = create_retriever(config)
        self.task_generator = TaskGenerator(config)
    
    def reset(
        self,
        seed: Optional[int] = None,
        task: Optional[SearchTask] = None,
        **kwargs
    ) -> SearchObservation:
        """
        Reset environment for new episode.
        
        Args:
            seed: Random seed for reproducibility
            task: Specific task to use (if None, samples from task set)
        
        Returns:
            Initial observation with question and empty context
        """
        pass
    
    def step(self, action: Action) -> SearchObservation:
        """
        Execute action and return new observation.
        
        Args:
            action: One of SearchAction, ReadAction, PruneAction, AnswerAction
        
        Returns:
            Observation with action result, updated context, and reward
        """
        pass
    
    def get_state(self) -> SearchState:
        """Return current environment state."""
        pass
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state for debugging."""
        pass
```

### 9.2 Client Interface

```python
class SearchEnvClient(EnvClient):
    """
    Client for interacting with Search RL Environment.
    """
    
    def reset(self, task_id: Optional[str] = None) -> StepResult[SearchObservation]:
        """Reset and optionally specify a task."""
        pass
    
    def search(self, query: str, top_k: int = 10) -> StepResult[SearchObservation]:
        """Convenience method for search action."""
        return self.step(Action(ActionType.SEARCH, SearchAction(query, top_k)))
    
    def read(self, chunk_ids: List[str]) -> StepResult[SearchObservation]:
        """Convenience method for read action."""
        return self.step(Action(ActionType.READ, ReadAction(chunk_ids)))
    
    def prune(self, chunk_ids: List[str]) -> StepResult[SearchObservation]:
        """Convenience method for prune action."""
        return self.step(Action(ActionType.PRUNE, PruneAction(chunk_ids)))
    
    def answer(self, answer: str, supporting_chunks: List[str]) -> StepResult[SearchObservation]:
        """Convenience method for answer action."""
        return self.step(Action(ActionType.ANSWER, AnswerAction(answer, supporting_chunks)))
```

---

## 10. Implementation Plan

### Phase 1: Core Environment (MVP)
- [ ] Basic state/action/observation models
- [ ] Simple BM25-only retrieval
- [ ] F-beta reward function
- [ ] In-memory document corpus (small test set)

### Phase 2: Full Retrieval System
- [ ] Dense embeddings (sentence-transformers)
- [ ] Hybrid retrieval with RRF
- [ ] Optional reranker integration
- [ ] Persistent corpus storage (ChromaDB or similar)

### Phase 3: Task Generation
- [ ] Task loading from JSON/Parquet
- [ ] LLM-based task generation pipeline
- [ ] Difficulty curriculum

### Phase 4: Training Integration
- [ ] Beta scheduling for curriculum
- [ ] Logging and metrics
- [ ] Compatibility with RL frameworks (GRPO, PPO, etc.)

---

## 11. Example Episode

```python
# Initialize
env = SearchEnvironment(config)
obs = env.reset()

# Agent receives: question="Who founded the company that acquired Instagram?"
# Context is empty, budget is full

# Step 1: Search
obs = env.step(Action(SEARCH, SearchAction("Instagram acquisition")))
# Returns: List of chunk summaries about Instagram

# Step 2: Read relevant result
obs = env.step(Action(READ, ReadAction(["chunk_123"])))
# Returns: Full content about Facebook acquiring Instagram in 2012

# Step 3: Follow-up search
obs = env.step(Action(SEARCH, SearchAction("Facebook founder")))
# Returns: Chunks about Mark Zuckerberg

# Step 4: Read
obs = env.step(Action(READ, ReadAction(["chunk_456"])))
# Returns: Content about Zuckerberg founding Facebook

# Step 5: Answer
obs = env.step(Action(ANSWER, AnswerAction(
    answer="Mark Zuckerberg",
    supporting_chunk_ids=["chunk_123", "chunk_456"]
)))
# Episode ends, final reward calculated

# Reward breakdown:
# - Trajectory F-beta: 1.0 (both gold chunks retrieved)
# - Answer bonus: 1.0 (correct answer)
# - Efficiency: 0.08 (used 5 of 20 steps)
# Total: ~2.08
```

---

## 12. References

1. Chroma Context-1 Technical Report
2. CISPO: GRPO variant for agentic search training
3. OpenEnv framework documentation
4. BM25 and dense retrieval literature
5. Multi-hop QA datasets (HotpotQA, MuSiQue, etc.)
