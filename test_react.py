from app.agents.react_agent import run_react_agent
from app.evaluation.harness import run_with_harness
from app.models import AgentType
import os
from dotenv import load_dotenv
load_dotenv()

result = run_with_harness(
    agent_fn=run_react_agent,
    topic="benefits of exercise",
    agent_type=AgentType.REACT,
    model=os.getenv("AGENT_MODEL", "llama-3.1-8b-instant")
)

print("\n" + "="*50)
print("REACT AGENT RESULT")
print("="*50)
print(f"Latency:  {result.latency_seconds}s")
print(f"Tokens:   {result.token_usage.total_tokens}")
print(f"Cost:     USD {result.token_usage.estimated_cost_usd}")
print(f"Report:\n{result.report[:500]}...")