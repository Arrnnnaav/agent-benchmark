from app.evaluation.blog_writer import generate_blog_post, save_blog_post
from app.models import (
    BenchmarkComparision, RunResult,
    TokenUsage, AgentType
)

def make_result(agent_type, latency, tokens, cost, score, report):
    return RunResult(
        agent_type=agent_type,
        topic="benefits of exercise",
        report=report,
        latency_seconds=latency,
        token_usage=TokenUsage(
            prompt_tokens=int(tokens*0.7),
            completion_tokens=int(tokens*0.3),
            total_tokens=tokens,
            estimated_cost_usd=cost
        ),
        quality_score=score,
        quality_feedback="Well structured with good coverage of the topic."
    )

results = [
    make_result(AgentType.REACT,       9.1,  1986, 0.000117, 7.5, "Exercise improves mood and reduces disease risk."),
    make_result(AgentType.REFLECTION,  15.0, 3083, 0.000184, 8.0, "Regular exercise has benefits for mental and physical health."),
    make_result(AgentType.MEMORY,      11.2, 2100, 0.000125, 7.8, "Exercise reduces chronic disease risk and improves cognition."),
    make_result(AgentType.TEAM,        7.9,  3372, 0.000199, 8.5, "Exercise provides cardiovascular, mental, and longevity benefits."),
]

comparison = BenchmarkComparision(
    topic="benefits of exercise",
    results=results,
    fastest_agent=AgentType.TEAM,
    cheapest_agent=AgentType.REACT,
    highest_quality_agent=AgentType.TEAM,
)

blog = generate_blog_post(comparison)
filename = save_blog_post(blog, "benefits of exercise")
print(f"Blog saved to: {filename}")
print("\nPreview:")
print(blog[:800])