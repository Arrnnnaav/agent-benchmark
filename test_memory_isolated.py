from dotenv import load_dotenv
load_dotenv()

from app.agents.memory_agent import run_memory_agent

result = run_memory_agent(topic="benefits of exercise")
print("Result:", result[:200])