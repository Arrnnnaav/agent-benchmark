import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from app.tools.search import get_search_tool
from app.tools.calculator import calculator
from app.tools.scraper import scrape_webpage



load_dotenv()

def get_llm(temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(
        model=os.getenv("AGENT_MODEL", "llama-3.1-8b-instant"),
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=512
    )

def build_react_agent():
    llm = get_llm()
    tools = [
        get_search_tool(),
        calculator,
        scrape_webpage
    ]
    system_prompt = (
        "You are a research agent. Use your tools to gather information "
        "and write comprehensive, well-structured reports. "
        "Always search for current information before writing. "
        "Use the calculator for any numerical analysis."
    )
    return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt
    )

def run_react_agent(topic: str, callbacks: list = None, **kwargs) -> str:
    agent = build_react_agent()
    prompt_text = f"""Research this topic and write a concise report.

    Topic: {topic}

    Search for key facts, then write a structured report with findings and summary.
    Keep the report under 400 words."""
    
    config = {"callbacks": callbacks or []}
    result = agent.invoke(
        {"messages": [HumanMessage(content=prompt_text)]},
        config=config
    )
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return "No output generated"