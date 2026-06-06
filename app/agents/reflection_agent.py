import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.tools.search import get_search_tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

def get_generator_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("AGENT_MODEL", "llama-3.1-8b-instant"),
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=512
    )


def get_critic_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("CRITIC_MODEL", "llama-3.1-8b-instant"),
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=256
    )


#  Generator - researches and writes 
def generate_draft(topic: str, feedback: str = None, callbacks: list = None) -> str:
    tools = [get_search_tool()]
    system_prompt = (
        "You are an expert research writer. "
        "Search for information on the given topic and write a clear, "
        "well-structured report. Be concise but comprehensive. "
        "Keep your report under 350 words."
    )
    agent = create_react_agent(
        model=get_generator_llm(),
        tools=tools,
        prompt=system_prompt
    )
    if feedback:
        task = f"""Topic: {topic}

            Previous draft was scored. Critic feedback:
            {feedback}

            Rewrite the report addressing all the feedback points."""
    else:
        task = f"""Topic: {topic}

            Search for key information and write a structured report with findings and summary."""

    config = {"callbacks": callbacks or []}
    result = agent.invoke(
        {"messages": [HumanMessage(content=task)]},
        config=config
    )
    messages = result.get("messages", [])
    if messages:
        return messages[-1].content
    return ""


#  Critic — scores and gives feedback
def critique_draft(topic: str, draft: str) -> tuple[float, str]:
    critic_llm = get_critic_llm()
    system_msg = SystemMessage(content=(
        "You are an expert editor and critic. "
        "Evaluate research reports objectively. "
        "Always respond in this exact format:\n"
        "SCORE: <number between 0 and 10>\n"
        "FEEDBACK: <specific, actionable feedback>"
    ))
    human_msg = HumanMessage(content=(
        f"Topic: {topic}\n\n"
        f"Report to evaluate:\n{draft}\n\n"
        f"Score this report 0-10 and provide specific feedback."
    ))
    response = critic_llm.invoke([system_msg, human_msg])
    response_text = response.content
    score = 5.0
    feedback = response_text
    for line in response_text.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.replace("SCORE:", "").strip())
            except ValueError:
                score = 5.0
        elif line.startswith("FEEDBACK:"):
            feedback = line.replace("FEEDBACK:", "").strip()
    score = max(0.0, min(10.0, score))
    return score, feedback


# Main reflection loop 
def run_reflection_agent(topic: str, callbacks: list = None, **kwargs) -> str:
    max_iterations = int(os.getenv("MAX_REFLECTION_ITERATIONS", 3))
    threshold = float(os.getenv("REFLECTION_QUALITY_THRESHOLD", 7.0))
    draft = ""
    feedback = None
    final_score = 0.0

    for iteration in range(max_iterations):
        print(f"\n[Reflection] Iteration {iteration + 1}/{max_iterations}")
        draft = generate_draft(
            topic=topic,
            feedback=feedback,
            callbacks=callbacks
        )
        print(f"[Reflection] Draft generated ({len(draft)} chars)")
        time.sleep(2)
        final_score, feedback = critique_draft(topic, draft)
        print(f"[Reflection] Score: {final_score}/10")
        print(f"[Reflection] Feedback: {feedback[:100]}...")
        if final_score >= threshold:
            print(f"[Reflection] Quality threshold met. Stopping.")
            break
        if iteration < max_iterations - 1:
            time.sleep(3)

    return f"{draft}\n\n---\nReflection Score: {final_score}/10"
