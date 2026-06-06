from app.evaluation.scorer import score_result
from app.models import RunResult, TokenUsage, AgentType

fake_result = RunResult(
    agent_type=AgentType.REACT,
    topic="benefits of exercise",
    report="""
    Exercise has numerous benefits for physical and mental health.
    Regular physical activity reduces the risk of chronic diseases,
    improves mood, boosts energy levels, and helps maintain a healthy weight.
    Studies show that 30 minutes of moderate exercise daily significantly
    improves cardiovascular health and mental wellbeing.
    """,
    latency_seconds=10.0,
    token_usage=TokenUsage(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        estimated_cost_usd=0.0001
    )
)

scored = score_result(fake_result)
print(f"Quality Score:    {scored.quality_score}/10")
print(f"Quality Feedback: {scored.quality_feedback}")