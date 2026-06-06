import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool

load_dotenv()


@tool
def search_web(query: str) -> str:
    """
    Search the web for current information on a topic.
    Input should be a search query string.
    Returns a summary of the top search results.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not found"
    client = TavilyClient(api_key=api_key)
    results = client.search(query, max_results=2)
    output = []
    for r in results.get("results", []):
        title = r.get("title", "")
        content = r.get("content", "")[:300]
        output.append(f"{title}: {content}")
    return "\n\n".join(output) if output else "No results found"


def get_search_tool():
    return search_web