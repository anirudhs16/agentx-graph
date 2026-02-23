from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class ResearchState(TypedDict):
    # The original question
    query: str
    # Each agent writes its output here
    searcher_output: str
    synthesiser_output: str
    critic_output: str
    verdict_output: str
    
    # Confidence score from verdict agent (0-100)
    confidence: int
    
    # How many times we've looped back
    retry_count: int