import httpx #instead of request it is sync but FastAPI is async so httpx is async HTTP client
from bs4 import BeautifulSoup
from langchain.tools import tool

@tool
def scrape_webpage(url : str) -> str:
    """
    Useful fro reading the full content of a webpage given its URL.
    Input should be a complete URL starting with http:// or https://.
    Returns the clean text content of the page.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; research-agent/1.0)"        
            }
        response = httpx.get(url, headers=headers, timeout = 15)
        response.raise_for_status() # raise the httpx error is occured
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]): # remove HTML tags that contains no useful information
            tag.decompose() #decompose removes the tag and everything inside it from the tree entirely
        text = soup.get_text(separator = "\n", strip = True) #extract remaining text from the cleaned HTML tree puts a newline between each block of text so paragraph stays seperatd
        lines = [line for line in text.splitlines() if line.strip()] # split into individual lines then filter out blank lines
        clean_text = "\n".join(lines)
        return clean_text[:500]

    except httpx.TimeoutException:
        return "Error: Request times out after 10 seconds"
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} for URL {url}"
    except Exception as e:
        return f"Error: {str(e)}"

