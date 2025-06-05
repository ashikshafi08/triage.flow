import asyncio
from src.github_client import GitHubIssueClient


# Optionally, test FastAPI endpoints if httpx is available
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

REPO_URL = "https://github.com/huggingface/transformers"

async def test_github_client():
    print("Testing GitHubIssueClient.list_issues...")
    client = GitHubIssueClient()
    issues = await client.list_issues(REPO_URL, state="open", per_page=5, max_pages=1)
    print(f"Fetched {len(issues)} issues from {REPO_URL}")
    if issues:
        first = issues[0]
        print("First issue:")
        print(f"  Number: {first.number}")
        print(f"  Title: {first.title}")
        print(f"  State: {first.state}")
        print(f"  URL: {first.url}")

    if issues:
        print("\nTesting GitHubIssueClient.get_issue for first issue...")
        issue_url = f"{REPO_URL}/issues/{issues[0].number}"
        resp = await client.get_issue(issue_url)
        print(f"Status: {resp.status}")
        if resp.data:
            print(f"Title: {resp.data.title}")
            print(f"Body: {resp.data.body[:100]}...")

async def test_fastapi_endpoints():
    if not HAS_HTTPX:
        print("httpx not installed, skipping FastAPI endpoint tests.")
        return
    print("\nTesting FastAPI endpoints...")
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        r = await client.get("/api/issues", params={"repo_url": REPO_URL, "state": "open"})
        print(f"/api/issues status: {r.status_code}")
        issues = r.json()
        print(f"Fetched {len(issues)} issues from API.")
        if issues:
            first = issues[0]
            print(f"First issue from API: #{first['number']} {first['title']}")
            r2 = await client.get(f"/api/issues/{first['number']}", params={"repo_url": REPO_URL})
            print(f"/api/issues/{{number}} status: {r2.status_code}")
            detail = r2.json()
            print(f"Detail title: {detail['title']}")

if __name__ == "__main__":
    asyncio.run(test_github_client())
    if HAS_HTTPX:
        asyncio.run(test_fastapi_endpoints()) 