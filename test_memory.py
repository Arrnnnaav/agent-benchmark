from app.agents.memory_agent import run_memory_agent
from app.evaluation.harness import run_with_harness
from app.models import AgentType
import os
from dotenv import load_dotenv
load_dotenv()

# Run 1 — cold start, no memories
print("RUN 1 - Cold start")
result1 = run_with_harness(
    agent_fn=run_memory_agent,
    topic="benefits of exercise",
    agent_type=AgentType.MEMORY,
    model=os.getenv("AGENT_MODEL", "llama-3.1-8b-instant")
)
print(f"Latency: {result1.latency_seconds}s | Tokens: {result1.token_usage.total_tokens}")

# Run 2 — same topic, should find memories from run 1
print("\nRUN 2 - With memories from run 1")
result2 = run_with_harness(
    agent_fn=run_memory_agent,
    topic="exercise and mental health",
    agent_type=AgentType.MEMORY,
    model=os.getenv("AGENT_MODEL", "llama-3.1-8b-instant")
)
print(f"Latency: {result2.latency_seconds}s | Tokens: {result2.token_usage.total_tokens}")
print("\n" + "="*50)
print("MEMORY AGENT RESULT")
print("="*50)
print(result2.report[:600])