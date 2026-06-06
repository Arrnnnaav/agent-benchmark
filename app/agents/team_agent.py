# three specialized agents connected in a pipeline using langgraph
# reasearch agent, writer agent,  editor agent 
# three seperate LLm calls with three completely different system prompts
# each node - an agent function
# each edge - handoff
# state is a shared dictionary that flows through all nodes

import os
from dotenv import load_dotenv
from typing import TypedDict # defines a dictionary with a specific keys and values type. use it to defing grapg state
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END #StateGraph - class for building a graph of nodes END special constant that marks the terminal node
from app.tools.search import get_search_tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

class TeamState(TypedDict): # this is the state dictionary that flows through the entire graph. every node reciever this dict and return an updated version of it
    topic: str
    research_notes: str
    draft_report: str
    final_report: str

def get_llm() -> ChatGroq:
    return ChatGroq(
        model = os.getenv("AGENT_MODEL", "llama-3.1-8b-instant"),
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=512
    )

#RESEARCH NODE
#a langggraph node is just an python function. it takes the current state and return a dictionary of state update.
#langgraph merges the returned dict into the state automatically
def research_node(state: TeamState) -> dict: 
    print("\n[Team]  Researcher starting...")
    topic = state["topic"]
    tools = [get_search_tool()]
    system_prompt = (
        "You are a research specialist. Your only job is to gather facts. "
        "Search for information on the given topic."
        "Return detailed research notes with key facts, statistics, and findings. "
        "Do NOT write a report. Just return raw research notes. "
    )

    agent = create_react_agent(
        model = get_llm(),
        tools = tools,
        prompt = system_prompt
    )
    result = agent.invoke({
           "messages": [HumanMessage(
               content = f"Research this topic thoroughly: {topic}\n"
                         f"Return detailed notes with the key facts anf findings"
           )] 
        })
    # result is a dictionary ususally
    messages = result.get("messages", []) # safer then below version
    # extarcting messages 
    #equivalent to
    # if "messages" in results:
    #    messages = result["messages"]
    # else:
    #    messages = []


    notes = messages[-1].content if messages else "No research gathered"
    # -1 = last content this this raises error when messages is empty       .content to extract text then results are stored in the notes

    print(f"[Team] Research comeplete ({len(notes)} chars)")
    return {"research_notes": notes}

#WRITER NODE
def writer_node(state: TeamState) -> dict:
    print("\n[Team] Writer starting...")
    #writer reade both topic and research_notes from state
    topic = state["topic"]
    research_notes = state["research_notes"]

    llm = get_llm()
    system_msg = SystemMessage(content = (
        "You are an expert report writer. "
        "Transform research notes into a clear, well-structures report. "
        "Use headings, organize logically, write in professional prose. "
        "Keep the report under 400 words. "
    ))

    human_msg = HumanMessage(content=(
        f"Topic: {topic}\n\n"
        f"Research Notes:\n{research_notes[:1500]}\n\n"
        f"Write a comprehensive, well-structures report based in these notes."
    ))

    response = llm.invoke([system_msg, human_msg])
    draft = response.content
    print(f"[Team] Draft Written ({len(draft)} chars)")
    return {"draft_report": draft}

#EDITOR NODE
def editor_node(state: TeamState) -> dict:
    print("\n[Team] Editor starting...")
    topic = state["topic"]
    draft_report = state["draft_report"]

    llm = get_llm()
    system_msg = SystemMessage(content=(
        "You are a professional editor. "
        "Polish and improve the given report draft. "
        "Fix any awkward phrasing, improve flow, ensure logical structure. "
        "Add a clear introduction and conclusion if missing. "
        "Do not add new facts — only improve what is already there. "
        "Keep the final report under 450 words."
    ))

    human_msg = HumanMessage(content=(
        f"Topic: {topic}\n\n"
        f"Draft to edit:\n{draft_report}\n\n"
        f"Polish this report. Improve clarity, flow, and structure."
    ))

    response = llm.invoke([system_msg, human_msg])
    final = response.content
    print(f"[Team] Editing complete ({len(final)} chars)")
    return {"final_report": final}


# BUILD TEAM GRAPH
#Assemble the three nodes into a connected graph called once per run
def build_team_graph():
    # create a new graph with TeamState as its state type
    graph = StateGraph(TeamState)
    #register the nodes 
    graph.add_node("researcher", research_node)
    graph.add_node("writer", writer_node)
    graph.add_node("editor", editor_node)
    #set entry point
    graph.set_entry_point("researcher")
    # create edges and define the flow
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "editor")
    graph.add_edge("editor", END)

    return graph.compile()
    #after compile we can call .invoke() on it just like any othet LangChain runnable

# entry point called by the harness
def run_team_agent(topic: str, callbacks: list = None, **kwargs) -> str:
    team = build_team_graph()
    initial_state : TeamState = {
        "topic": topic,
        "research_notes": "",
        "draft_report": "",
        "final_report": ""
    }
    #run the full graph updating state at each step. final_state contains all four fields filled in
    config = {"callbacks": callbacks or []}
    final_state = team.invoke(initial_state, config = config)
    return final_state.get("final_report", "No output generated")