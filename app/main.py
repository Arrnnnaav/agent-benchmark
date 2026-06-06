# FastAPI application
# GET  /health      → is the server alive?
# POST /run         → run one agent on a topic
# POST /run/all     → run all four agents and compare
# GET  /results     → list saved result files
# POST /blog        → run all agents and generate blog post

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time 

from app.models import (
    RunRequest,
    RunResult,
    BenchmarkComparision,
    BenchmarkRequest,
    AgentType,
    HealthResponse
)
from app.agents.react_agent import run_react_agent
from app.agents.reflection_agent import run_reflection_agent
from app.agents.memory_agent import run_memory_agent
from app.agents.team_agent import run_team_agent
from app.evaluation.harness import run_with_harness
from app.evaluation.scorer import score_result, score_all
from app.evaluation.blog_writer import generate_blog_post, save_blog_post

load_dotenv()


# ── App setup ─────────────────────────────────────────────────
app = FastAPI(
    title="Agent Benchmark API",
    description="Compare ReAct, Reflection, Memory and Team agent architectures",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ── Agent dispatcher ───────────────────────────────────────────
AGENT_MAP = {
    AgentType.REACT: run_react_agent,
    AgentType.REFLECTION: run_reflection_agent,
    AgentType.MEMORY: run_memory_agent,
    AgentType.TEAM: run_team_agent,
}


# ── Save result helper ─────────────────────────────────────────
def save_result(result: RunResult) -> str:
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/{result.agent_type.value}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(result.model_dump(), f, indent=2)
    return filename


# ── GET /health ────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok")


# ── POST /run ─────────────────────────────────────────────────
@app.post("/run", response_model=RunResult)
async def run_agent(request: RunRequest):
    agent_fn = AGENT_MAP.get(request.agent_type)
    if not agent_fn:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown agent type: {request.agent_type}"
        )
    model = os.getenv("AGENT_MODEL", "llama-3.1-8b-instant")
    result = run_with_harness(
        agent_fn=agent_fn,
        topic=request.topic,
        agent_type=request.agent_type,
        model=model
    )
    result = score_result(result)
    if request.save_output:
        filename = save_result(result)
        print(f"Saved to {filename}")
    return result


# ── POST /run/all ──────────────────────────────────────────────
@app.post("/run/all", response_model=BenchmarkComparision)
async def run_all_agents(request: BenchmarkRequest):
    results = []                                         # fix 1: this is the list, never overwrite it
    model = os.getenv("AGENT_MODEL", "llama-3.1-8b-instant")

    for agent_type, agent_fn in AGENT_MAP.items():       # loop runs all 4 agents
        print(f"\nRunning {agent_type.value} agent...")
        result = run_with_harness(                       # fix 2: singular — one result per iteration
            agent_fn=agent_fn,
            topic=request.topic,
            agent_type=agent_type,
            model=model
        )
        results.append(result)                           # fix 3: append singular result to plural list
        if request.save_output:
            save_result(result)                          # fix 4: save singular result
        time.sleep(5)
    # fix 5: ALL of this is OUTSIDE the loop — runs once after all 4 agents finish
    results = score_all(results)
    fastest = min(results, key=lambda r: r.latency_seconds)
    cheapest = min(results, key=lambda r: r.token_usage.estimated_cost_usd)
    highest_quality = max(results, key=lambda r: r.quality_score or 0)

    comparison = BenchmarkComparision(
        topic=request.topic,                             # fix 6: topic not topvi
        results=results,
        fastest_agent=fastest.agent_type,
        cheapest_agent=cheapest.agent_type,
        highest_quality_agent=highest_quality.agent_type
    )
    return comparison                                    # fix 7: return is OUTSIDE the loop


# ── GET /results ───────────────────────────────────────────────
@app.get("/results")
def list_results():
    if not os.path.exists("outputs"):
        return {"results": []}
    files = [f for f in os.listdir("outputs") if f.endswith(".json")]
    files.sort(reverse=True)
    return {"results": files, "count": len(files)}


# ── POST /blog ─────────────────────────────────────────────────
@app.post("/blog")
async def generate_blog(request: BenchmarkRequest):
    results = []                                         # same fixes applied here too
    model = os.getenv("AGENT_MODEL", "llama-3.1-8b-instant")

    for agent_type, agent_fn in AGENT_MAP.items():
        print(f"\nRunning {agent_type.value} agent...")
        result = run_with_harness(
            agent_fn=agent_fn,
            topic=request.topic,
            agent_type=agent_type,
            model=model
        )
        results.append(result)

    results = score_all(results)
    fastest = min(results, key=lambda r: r.latency_seconds)
    cheapest = min(results, key=lambda r: r.token_usage.estimated_cost_usd)
    highest_quality = max(results, key=lambda r: r.quality_score or 0)

    comparison = BenchmarkComparision(
        topic=request.topic,
        results=results,
        fastest_agent=fastest.agent_type,
        cheapest_agent=cheapest.agent_type,
        highest_quality_agent=highest_quality.agent_type
    )
    blog_content = generate_blog_post(comparison)
    filename = save_blog_post(blog_content, request.topic)
    return {
        "blog_file": filename,
        "preview": blog_content[:500],
        "comparison": comparison
    }