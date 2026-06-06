# LLM as judge it takes any agent report and scores it 0-10

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from app.models import RunResult

load_dotenv()

def get_judge_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("CRITIC_MODEL", "llama-3.1-8b-instant"),
        temperature=0.1,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=512
    )

#takes raw LLM response and extracts the numeric scores and feedback text
def parse_scores(response_text: str) -> tuple[float, str]:
    scores = {}
    feedback_lines = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if not line :
            continue
    
    if line.startswith("ACCURACY:"):
        try:
            scores["accuracy"] = float(line.replace("ACCURACY:", "").strip())
        except ValueError:
            scores["accuracy"] = 5.0
    elif line.startswith("COMPLETENESS:"):
            try:
                scores["completeness"] = float(line.replace("COMPLETENESS:", "").strip())
            except ValueError:
                scores["completeness"] = 5.0

    elif line.startswith("STRUCTURE:"):
        try:
            scores["structure"] = float(line.replace("STRUCTURE:", "").strip())
        except ValueError:
            scores["structure"] = 5.0

    elif line.startswith("CLARITY:"):
        try:
            scores["clarity"] = float(line.replace("CLARITY:", "").strip())
        except ValueError:
            scores["clarity"] = 5.0

    elif line.startswith("FEEDBACK:"):
        feedback_lines.append(line.replace("FEEDBACK:", "").strip())

    else:
        feedback_lines.append(line)
    
    if scores:
        final_score = sum(scores.values()) / len(scores)
    else:
        final_score = 5.0
    

    final_score = max(0.0, min(10.0, final_score))
    feedback = " ".join(feedback_lines) if feedback_lines else "No feedback provided"
    return round(final_score, 2), feedback



def score_result(result: RunResult) -> RunResult:
    if not result.report or result.report.startswith("Agent failed"):
        result.quality_score = 0.0
        result.quality_feedback = "Agent failed to produce a report"
        return result
    judge = get_judge_llm()
    system_msg = SystemMessage(content=(
        "You are an expert evaluator of research reports. "
        "Score the given report on four criteria, each from 0 to 10.\n\n"
        "Respond in this EXACT format with nothing else:\n"
        "ACCURACY: <score>\n"
        "COMPLETENESS: <score>\n"
        "STRUCTURE: <score>\n"
        "CLARITY: <score>\n"
        "FEEDBACK: <one sentence of specific feedback>"
    ))
    report_preview = result.report[:1000]
    human_msg = HumanMessage(content=(
        f"Topic: {result.topic}\n\n"
        f"Agent type: {result.agent_type.value}\n\n"
        f"Report to evaluate:\n{report_preview}\n\n"
        f"Score this report on all four criteria."
    ))
    try:
        response = judge.invoke([system_msg, human_msg])
        score, feedback = parse_scores(response.content)
        result.quality_score = score
        result.quality_feedback = feedback
    except Exception as e:
        result.quality_score = 5.0
        result.quality_feedback = f"Scoring failed: {str(e)}"
    return result


# ── Batch scorer ───────────────────────────────────────────────
def score_all(results: list[RunResult]) -> list[RunResult]:
    scored = []
    for result in results:
        print(f"[Scorer] Scoring {result.agent_type.value} agent...")
        scored_result = score_result(result)
        print(f"[Scorer] Score: {scored_result.quality_score}/10")
        scored.append(scored_result)
    return scored