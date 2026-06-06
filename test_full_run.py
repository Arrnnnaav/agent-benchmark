import requests
import json

BASE_URL = "http://localhost:8000"

# Step 1 — health check
print("Step 1: Health check")
r = requests.get(f"{BASE_URL}/health")
print(f"Status: {r.json()}")

# Step 2 — run single agent
print("\nStep 2: Single agent run (ReAct)")
r = requests.post(f"{BASE_URL}/run", json={
    "topic": "artificial intelligence in healthcare",
    "agent_type": "react",
    "save_output": True
})
result = r.json()
print(f"Latency:  {result['latency_seconds']}s")
print(f"Tokens:   {result['token_usage']['total_tokens']}")
print(f"Quality:  {result['quality_score']}/10")
print(f"Feedback: {result['quality_feedback']}")
print(f"Report:   {result['report'][:200]}...")

# Step 3 — list saved results
print("\nStep 3: List saved results")
r = requests.get(f"{BASE_URL}/results")
data = r.json()
print(f"Total saved runs: {data['count']}")
for f in data['results'][:5]:
    print(f"  {f}")


# Step 4 — run all four agents
print("\nStep 4: Full benchmark — all four agents")
r = requests.post(f"{BASE_URL}/run/all", json={
    "topic": "artificial intelligence in healthcare",
    "save_output": True
})
print("Status code:", r.status_code)
comparison = r.json()
print("Response keys:", list(comparison.keys()))
print("Full response:", json.dumps(comparison, indent=2)[:1000])

print(f"\nFastest agent:  {comparison['fastest_agent']}")
print(f"Cheapest agent: {comparison['cheapest_agent']}")
print(f"Best quality:   {comparison['highest_quality_agent']}")

print("\n--- Per agent results ---")
for result in comparison['results']:
    print(f"\n{result['agent_type'].upper()}")
    print(f"  Latency:  {result['latency_seconds']}s")
    print(f"  Tokens:   {result['token_usage']['total_tokens']}")
    print(f"  Cost:     USD {result['token_usage']['estimated_cost_usd']}")
    print(f"  Quality:  {result['quality_score']}/10")
    print(f"  Feedback: {result['quality_feedback'][:80]}...")


# Step 5 — generate blog post
print("\nStep 5: Generate blog post")
r = requests.post(f"{BASE_URL}/blog", json={
    "topic": "artificial intelligence in healthcare",
    "save_output": True
})
if r.status_code == 200:
    data = r.json()
    print(f"Blog saved to: {data['blog_file']}")
    print(f"Preview:\n{data['preview'][:400]}")
else:
    print(f"Error {r.status_code}: {r.text[:200]}")