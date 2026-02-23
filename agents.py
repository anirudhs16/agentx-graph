# agents.py
from langchain_groq import ChatGroq
from state import ResearchState

# Initialise model
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

llm_fast = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)

# ── Searcher Node ──────────────────────────────────────
def searcher_node(state: ResearchState) -> ResearchState:
    response = llm_fast.invoke([
        ("system", """You are a Searcher agent. 
         Find raw facts and evidence only. 
         No opinions. No conclusions. Just evidence."""),
        ("human", f"Research this: {state['query']}")
    ])
    
    return {"searcher_output": response.content}

# ── Synthesiser Node ───────────────────────────────────
def synthesiser_node(state: ResearchState) -> ResearchState:
    response = llm.invoke([
        ("system", """You are a Synthesiser agent.
         Build a clear structured answer from the evidence.
         Be decisive. Draw conclusions."""),
        ("human", f"""Question: {state['query']}
         Evidence: {state['searcher_output']}""")
    ])
    
    return {"synthesiser_output": response.content}

# ── Critic Node ────────────────────────────────────────
def critic_node(state: ResearchState) -> ResearchState:
    response = llm_fast.invoke([
        ("system", """You are a Critic agent.
         Aggressively challenge the answer below.
         Find holes, assumptions, missing evidence.
         Be harsh but fair."""),
        ("human", f"""Answer to critique: {state['synthesiser_output']}""")
    ])
    
    return {"critic_output": response.content}

# ── Verdict Node ───────────────────────────────────────
def verdict_node(state: ResearchState) -> ResearchState:
    response = llm.invoke([
        ("system", """You are a Supervisor delivering a final verdict.
         You MUST end your response with exactly this format on the last line:
         CONFIDENCE: [number between 0 and 100]
         Nothing after that line."""),
        ("human", f"""Query: {state['query']}
         Searcher found: {state['searcher_output']}
         Synthesiser concluded: {state['synthesiser_output']}
         Critic challenged: {state['critic_output']}
         
         Deliver your final verdict.""")
    ])
    
    # Parse confidence from response
    content = response.content
    confidence = 75  # default if parsing fails
    
    lines = content.strip().split("\n")
    for line in lines:
        if line.startswith("CONFIDENCE:"):
            try:
                confidence = int(line.replace("CONFIDENCE:", "").strip())
            except:
                pass
    
    return {
        "verdict_output": content,
        "confidence": confidence,
        "retry_count": state.get("retry_count", 0) + 1
    }