from enum import Enum # let us define a fixed set of allowed values
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

#enum - Instead of using "magic numbers" or strings throughout your code, you can give meaningful names to those values.
class AgentType(str, Enum):
    REACT = "react"
    REFLECTION = "reflection"
    MEMORY = "memory"
    TEAM = "team"

class RunRequest(BaseModel): #request model
    topic: str
    agent_type: AgentType
    max_iterations: Optional[int] = Field(default = None, ge = 1, le =10) #how many iteration to allow (mainly for the reflection agent)
    save_output: bool = Field(default = True) #Field let us attach extra metadata to a field

class BenchmarkRequest(BaseModel):
    topic: str
    save_output: bool = Field(default=True)

class TokenUsage(BaseModel): # tracks token consumption for one agent run
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

class RunResult(BaseModel): #main result model
    agent_type: AgentType
    topic: str
    report: str
    latency_seconds: float
    token_usage: TokenUsage
    quality_feedback: Optional[str] = None
    quality_score: Optional[float] = None
    iterations_used: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory= dict)
 
class BenchmarkComparision(BaseModel): #comparision model
    topic: str
    results: List[RunResult]
    fastest_agent: Optional[AgentType] = None
    cheapest_agent: Optional[AgentType] = None
    highest_quality_agent: Optional[AgentType] = None
    summary: Optional[str] = None

class HealthResponse(BaseModel): #health check model
    status: str
    version: str = '1.0.0'

