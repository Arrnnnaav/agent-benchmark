from app.evaluation.harness import TokenCounterCallback, calculate_cost, run_with_harness
from app.models import AgentType
from langchain_core.outputs import LLMResult, Generation

# Test 1: cost calculation
cost = calculate_cost(1000, 500, 'llama-3.1-8b-instant')
print(f'Cost for 1000 input + 500 output tokens: USD {cost}')

# Test 2: callback accumulation
cb = TokenCounterCallback()
fake_result = LLMResult(
    generations=[[Generation(text='hello')]],
    llm_output={'token_usage': {'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150}}
)
cb.on_llm_end(fake_result)
cb.on_llm_end(fake_result)
print(f'After 2 LLM calls: {cb.total_tokens} total tokens')

# Test 3: full harness with a fake agent
def fake_agent(topic, callbacks, **kwargs):
    return f'This is a report about {topic}'

result = run_with_harness(
    agent_fn=fake_agent,
    topic='artificial intelligence',
    agent_type=AgentType.REACT,
    model='llama-3.1-8b-instant'
)
print(f'Latency: {result.latency_seconds}s')
print(f'Report: {result.report}')
print(f'Tokens: {result.token_usage.total_tokens}')
print('Harness OK')