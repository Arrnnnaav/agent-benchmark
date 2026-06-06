import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from app.tools.search import get_search_tool
from app.memory.chroma_store import ChromaMemoryStore
import time
load_dotenv()

#get llm
def get_llm() -> ChatGroq:
    return ChatGroq(
            model = os.getenv("AGENT_MODEL", "llama-3.1-8b-instant"),
            temperature=0.1,
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=512
    )

def format_memories(memories: list[str]) -> str:
    # take list of memory string from ChromaBD and formats them into a readable block
    if not memories:
        return "No relevant memories found"
    # start with a header line we are building a list of strings to join at the end
    formatted = ["Relevant knowledge from memory"]

    for i, memory in enumerate(memories, 1): # starts counting from 1 instead of 0
        formatted.append(f"{i}. {memory[:200]}") # number each memory and trim to 200 character 
    return "\n".join(formatted)

def save_to_memory(store: ChromaMemoryStore, topic: str, report: str) -> None:
    # after the agent writes a report, we extract and save key facts
    llm = get_llm()
    system_msg = SystemMessage( # system prompt for the extractor
        content = (
            "You are a knowledge extractor. "
            "Exatract 3-5 keys facts from the report as short sentences."
            "Each fact on its own line. No bullet points. No numbering. "
            "Just plain sentences. "
        )
    )
    human_msg = HumanMessage(content = ( #pass the first 800 chars of the report 
        f"Topic: {topic}\n\nReport:\n{report[:800]}\n\n"
    ))
    # call the llm and split the response by newlines 
    response = llm.invoke([system_msg, human_msg])
    facts = response.content.strip().split("\n")
    # save each fact thats longer than 20 character 
    for fact in facts:
        fact = fact.strip()
        if(len(fact)) > 20:
            store.save(
                text = fact,
                metadata = {"topic": topic, "source": "agent_report"}
            )

# Main agent
def run_memory_agent(topic: str, callbacks: list = None, **kwargs) -> str:
    store = ChromaMemoryStore()
    print(f"\n[Memory] Store has {store.count()} documents") # see how much the agents already knows. First run = 0

    memories = store.search(topic, n_results=3)
    memory_context = format_memories(memories)
    print(f"[Memory] Retrieved {len(memories)} relevant memories")

    tools = [get_search_tool()]
    #The system prompt tells the model it has memory — primes it to use the context we'll inject.
    system_prompt = (
        "You are a research agent with memory. "
        "You have access to relevant knowledge from past research. "
        "Use this knowledge alongside new searches. "
        "Write clear, well-structured reports under 350 words. "
    )

    agent = create_react_agent(
        model = get_llm(),
        tools = tools,
        prompt = system_prompt
    )
    #task injects memory_context directly into the prompt. The agent sees past knowledge before it even start searching
    #t can say "I already know X from memory let me search for whats new."
    task = f"""Topic: {topic}

    {memory_context}

    Use the above memory context and search for any additional current information.
    Write a comprehensive report with key findings and summary."""

    config = {"callbacks": callbacks or []}
    result = agent.invoke(
        {"messages": [HumanMessage(content = task)]},
        config = config
    )
    messages = result.get("messages", [])
    report = messages[-1].content if messages else "No output generated"

    print(f"[Memory] Saving new facts to memory...")
    try:
        time.sleep(5)
        save_to_memory(store, topic, report)
        print(f"[Memory] Store now has {store.count()} documents")
    except Exception as e:
        print(f"[Memory] Save skipped: {e}")
    return report