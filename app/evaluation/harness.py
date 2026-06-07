import time
from typing import Callable, Any
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from app.models import RunResult, TokenUsage, AgentType


# Token counting callback 
class TokenCounterCallback(BaseCallbackHandler):

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)


# Cost calculation 
def calculate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    PRICES = {
        "llama-3.1-8b-instant": {
            "input": 0.05 / 1_000_000,
            "output": 0.08 / 1_000_000
        },
        "llama-3.3-70b-versatile": {
            "input": 0.59 / 1_000_000,
            "output": 0.79 / 1_000_000
        },
    }
    price = PRICES.get(model, PRICES["llama-3.1-8b-instant"])
    input_cost = prompt_tokens * price["input"]
    output_cost = completion_tokens * price["output"]
    return round(input_cost + output_cost, 8)


# Main harness function
def run_with_harness(
    agent_fn: Callable,
    topic: str,
    agent_type: AgentType,
    model: str,
    **kwargs
    ) -> RunResult:

    callback = TokenCounterCallback()
    start_time = time.time()

    try:
        report = agent_fn(
            topic=topic,
            callbacks=[callback],
            **kwargs
        )
    except Exception as e:
        report = f"Agent failed with error: {str(e)}"

    end_time = time.time()
    latency = round(end_time - start_time, 3)

    token_usage = TokenUsage(
        prompt_tokens=callback.prompt_tokens,
        completion_tokens=callback.completion_tokens,
        total_tokens=callback.total_tokens,
        estimated_cost_usd=calculate_cost(
            callback.prompt_tokens,
            callback.completion_tokens,
            model
        )
    )

    return RunResult(
        agent_type=agent_type,
        topic=topic,
        report=report,
        latency_seconds=latency,
        token_usage=token_usage,
    )
