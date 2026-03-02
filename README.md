# agentx-graph 🔀🧠

> Multi-agent research system with conditional graph routing and automatic retry logic.

Builds on [agentx](https://github.com/anirudhs16/agentx) by replacing the fixed sequential pipeline with a **LangGraph StateGraph** — agents now loop back automatically when confidence is too low, instead of always running A→B→C→D regardless of output quality.

![Stack](https://img.shields.io/badge/stack-LangGraph%20%2B%20Groq%20%2B%20Python-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-active-brightgreen)

---

## What's New vs agentx

| Feature | agentx | agentx-graph |
|---|---|---|
| Sequential flow | ✅ | ✅ |
| Typed shared state | ❌ raw strings | ✅ ResearchState |
| Conditional routing | ❌ | ✅ |
| Automatic retry loop | ❌ | ✅ |
| Infinite loop guard | ❌ | ✅ retry_count |

---

## Graph Architecture

```
START
  ↓
searcher       ← finds raw facts and evidence
  ↓
synthesiser    ← builds structured answer from evidence
  ↓
critic         ← aggressively challenges the answer
  ↓
verdict        ← delivers final answer with confidence score (0–100)
  ↓
should_retry() ← router: reads confidence + retry_count
  ↓ confidence < 70 AND retry_count < 3
  ↓ retry → loops back to searcher
  ↓ done
END
```

The key difference from a sequential pipeline: after `verdict` runs, the `should_retry` router reads the confidence score from state and **dynamically decides** whether to loop back or finish. Sequential agents always run A→B→C→D. Graph agents **adapt based on their own output quality.**

---

## State Object

All nodes share a single typed state object that flows through the entire graph:

```python
class ResearchState(TypedDict):
    query: str              # original question — available to every node
    searcher_output: str    # raw facts found
    synthesiser_output: str # structured answer built from evidence
    critic_output: str      # challenges and holes in the synthesis
    verdict_output: str     # final answer from supervisor
    confidence: int         # score 0–100, parsed from verdict response
    retry_count: int        # increments each loop — prevents infinite retries
```

Each node returns **only the fields it changes**. LangGraph merges them into the central state automatically. This prevents nodes from accidentally overwriting each other's outputs.

---

## Conditional Routing

```python
def should_retry(state: ResearchState) -> str:
    if state["confidence"] < 70 and state["retry_count"] < 3:
        return "retry"   # → loops back to searcher
    return "done"        # → END

graph.add_conditional_edges(
    "verdict",
    should_retry,
    {
        "retry": "searcher",
        "done": END
    }
)
```

Two guards work together:
- **confidence threshold** — retry if the verdict agent scores itself below 70
- **retry_count cap** — stop after 3 retries regardless of confidence, preventing infinite loops

---

## Model Selection

| Agent | Model | Reason |
|---|---|---|
| Searcher | llama-3.1-8b-instant | lightweight — retrieval task, speed matters |
| Synthesiser | llama-3.3-70b-versatile | reasoning-heavy — needs stronger model |
| Critic | llama-3.1-8b-instant | pattern matching — finds holes, doesn't need depth |
| Verdict | llama-3.3-70b-versatile | final judgement — needs strongest reasoning |

---

## Tech Stack

- Python 3.10+
- LangGraph — StateGraph, conditional_edges, typed state
- LangChain Groq — LLM integration
- Groq API — Llama 3.3 70B / 3.1 8B (free tier)

---

## Getting Started

### Prerequisites
- Python 3.10+
- Free [Groq API key](https://console.groq.com)

### Setup

```bash
git clone https://github.com/anirudhs16/agentx-graph.git
cd agentx-graph

python3 -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

pip install -r requirements.txt

echo "GROQ_API_KEY=your_key_here" > .env
```

### Run

```bash
python main.py
```

Watch the terminal — if confidence comes back below 70 you'll see the retry message and the loop starts again. If it retries 3 times and still hasn't hit 70, it stops and returns what it has.

---

## Project Structure

```
agentx-graph/
├── main.py       # graph definition, nodes, conditional routing
├── agents.py     # node functions + Groq LLM calls
├── state.py      # ResearchState TypedDict
└── .env          # GROQ_API_KEY (git ignored)
```

---

## Key Concept — Why Partial State Returns

In `agentx`, each function returned a raw string passed directly to the next function. In `agentx-graph`, each node returns only the fields it changed:

```python
# Searcher only touches one field
def searcher_node(state: ResearchState) -> ResearchState:
    response = llm_fast.invoke([...])
    return {"searcher_output": response.content}  # only this field

# LangGraph merges it — everything else stays untouched
```

This matters for two reasons: safety (nodes can't accidentally overwrite fields they don't own) and clarity (one glance at a node's return tells you exactly what it's responsible for).

---

## Learning Progression

```
agentx          ← sequential pipeline (done)
agentx-graph    ← conditional graph + retry loop (you are here)
agentx-memory   ← short + long term memory (next)
agentx-tools    ← tool calling + autonomous agent
```

---

## License

MIT
