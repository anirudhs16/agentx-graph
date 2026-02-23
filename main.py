# main.py
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from state import ResearchState
from agents import searcher_node, synthesiser_node, critic_node, verdict_node

# ── Router ─────────────────────────────────────────────
def should_retry(state: ResearchState) -> str:
    """
    After verdict runs, decide:
    - confidence too low AND retries left → go back to searcher
    - confidence good OR retries exhausted → end
    """
    if state["confidence"] < 70 and state["retry_count"] < 3:
        print(f"\n🔄 Confidence {state['confidence']}% too low. Retrying... ({state['retry_count']}/3)")
        return "retry"
    else:
        print(f"\n✅ Confidence {state['confidence']}%. Done.")
        return "done"

# ── Build Graph ────────────────────────────────────────
def build_graph():
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("searcher", searcher_node)
    graph.add_node("synthesiser", synthesiser_node)
    graph.add_node("critic", critic_node)
    graph.add_node("verdict", verdict_node)

    # Add edges (sequential flow)
    graph.add_edge("searcher", "synthesiser")
    graph.add_edge("synthesiser", "critic")
    graph.add_edge("critic", "verdict")

    # Conditional edge from verdict
    # This is where graph beats sequential
    graph.add_conditional_edges(
        "verdict",
        should_retry,
        {
            "retry": "searcher",  # loop back
            "done": END           # finish
        }
    )

    # Entry point
    graph.set_entry_point("searcher")

    return graph.compile()

# ── Run ────────────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    # Initial state
    initial_state = {
        "query": "Should a mid-level React developer who has some experience build rag AI bots and llm integrations, switch to AI engineering in 2026? the goal is good career growth and money. I want final Yes/No answer with confidence score and reasoning. No hedging, be decisive.",
        "searcher_output": "",
        "synthesiser_output": "",
        "critic_output": "",
        "verdict_output": "",
        "confidence": 0,
        "retry_count": 0
    }

    print("🧠 Starting AgentX Graph...\n")
    result = app.invoke(initial_state)

    print("\n" + "="*50)
    print("FINAL VERDICT:")
    print("="*50)
    print(result["verdict_output"])
    print(f"\nConfidence: {result['confidence']}%")
    print(f"Retries: {result['retry_count']}")