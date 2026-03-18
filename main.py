import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

# LangGraph & Logic Imports
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # For State Persistence
import arxiv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# --- 1. SETUP LANGGRAPH LOGIC ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0) # gemini-2.5-pro

class ResearchState(TypedDict):
    topic: str
    paper_summaries: List[str]
    final_idea: Optional[str]

def scout_papers(state: ResearchState):
    search = arxiv.Search(query=state['topic'], max_results=3)
    summaries = [f"Title: {r.title}\nSummary: {r.summary[:300]}" for r in search.results()]
    return {"paper_summaries": summaries}

def synthesize_idea(state: ResearchState):
    context = "\n\n".join(state['paper_summaries'])
    prompt = f"Based on these papers about {state['topic']}, give 1 meaningful idea: {context}"
    response = llm.invoke(prompt)
    return {"final_idea": response.content}

# Compile with Memory (Checkpointer)
memory = MemorySaver()
workflow = StateGraph(ResearchState)
workflow.add_node("scout", scout_papers)
workflow.add_node("analyst", synthesize_idea)
workflow.set_entry_point("scout")
workflow.add_edge("scout", "analyst")
workflow.add_edge("analyst", END)
graph_app = workflow.compile(checkpointer=memory)

# --- 2. FASTAPI APP ---
app = FastAPI(title="Agentic Research API")

class ResearchRequest(BaseModel):
    topic: str
    thread_id: str  # Unique ID for the user's session

@app.post("/research")
async def run_research(request: ResearchRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    try:
        # Run the graph synchronously for this simple example
        result = graph_app.invoke({"topic": request.topic}, config)
        return {
            "thread_id": request.thread_id,
            "final_idea": result.get("final_idea"),
            "papers_found": len(result.get("paper_summaries", []))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="0.0.0.0", port=8000)