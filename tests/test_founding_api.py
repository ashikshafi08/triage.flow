import pytest
import pytest_asyncio
import httpx
import asyncio
import json
from typing import Dict, Any, Tuple

BASE_URL = "http://localhost:8000"

@pytest_asyncio.fixture
async def test_session() -> Tuple[str, httpx.AsyncClient]:
    """Fixture to create and cleanup a test session"""
    repo_url = "https://github.com/huggingface/smolagents"
    timeout = httpx.Timeout(160.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Create session
        resp = await client.post(f"{BASE_URL}/founder/sessions", json={"repo_url": repo_url})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        session_id = data["session_id"]
        
        yield session_id, client
        
        # Cleanup - delete session
        try:
            await client.delete(f"{BASE_URL}/assistant/sessions/{session_id}")
        except Exception as e:
            print(f"Warning: Failed to cleanup session {session_id}: {e}")

async def verify_response_structure(response, expected_fields):
    if isinstance(response, str):
        response = json.loads(response)
    for field in expected_fields:
        assert field in response, f"Missing field: {field}"
        assert response[field] is not None, f"Empty field: {field}"

@pytest.mark.asyncio
async def test_founding_member_agent_flow(test_session):
    session_id, client = test_session
    
    # 1. Verify session creation
    print("Session created:", session_id)
    assert session_id, "Session ID should not be empty"
    
    # 2. List and verify available tools
    resp = await client.get(f"{BASE_URL}/founder/sessions/{session_id}/tools")
    assert resp.status_code == 200, resp.text
    tools = resp.json()["tools"]
    print("Available tools:", tools)
    
    expected_tools = [
        "get_file_history",
        "summarize_feature_evolution",
        "who_fixed_this",
        "regression_detector",
        "who_implemented_this"
    ]
    for tool in expected_tools:
        assert tool in tools, f"Missing expected tool: {tool}"

    # 3. Test CodeAgent implementation query with retry
    query = "Who implemented the CodeAgent in smolagents repo?"
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = await client.post(
                f"{BASE_URL}/founder/sessions/{session_id}/agentic-query",
                json={"query": query}
            )
            assert resp.status_code == 200, resp.text
            result = resp.json()["result"]
            print(f"\nCodeAgent Implementation Query Result (Attempt {attempt + 1}):", result)
            await verify_response_structure(result, ["steps", "final_answer", "status"])
            break
        except AssertionError as e:
            if attempt == max_retries - 1:
                raise
            print(f"Attempt {attempt + 1} failed, retrying...")
            await asyncio.sleep(2)

    # 4. Test feature evolution with specific component
    query = "How has the CodeAgent's error handling evolved over time?"
    resp = await client.post(
        f"{BASE_URL}/founder/sessions/{session_id}/agentic-query",
        json={"query": query}
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    print("\nFeature Evolution Query Result:", result)
    await verify_response_structure(result, ["steps", "final_answer", "status"])

    # 5. Test who fixed this with specific file and line
    query = "Who last modified the error handling in CodeAgent class?"
    resp = await client.post(
        f"{BASE_URL}/founder/sessions/{session_id}/agentic-query",
        json={"query": query}
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    print("\nWho Fixed This Query Result:", result)
    await verify_response_structure(result, ["steps", "final_answer", "status"])

    # 6. Test regression detection with specific feature
    query = "Are there any regressions in the CodeAgent's error handling?"
    resp = await client.post(
        f"{BASE_URL}/founder/sessions/{session_id}/agentic-query",
        json={"query": query}
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    print("\nRegression Detection Query Result:", result)
    await verify_response_structure(result, ["steps", "final_answer", "status"])

    # 7. Test complex multi-step query with specific focus
    query = "What are the main components of the CodeAgent's error handling and how do they interact with the rest of the system?"
    resp = await client.post(
        f"{BASE_URL}/founder/sessions/{session_id}/agentic-query",
        json={"query": query}
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    print("\nComplex Multi-step Query Result:", result)
    await verify_response_structure(result, ["steps", "final_answer", "status"])

    # 8. Test error handling with invalid query
    query = ""  # Empty query should be handled gracefully
    resp = await client.post(
        f"{BASE_URL}/founder/sessions/{session_id}/agentic-query",
        json={"query": query}
    )
    assert resp.status_code == 400, "Empty query should return 400"
    
    # 9. Test session metadata
    resp = await client.get(f"{BASE_URL}/assistant/sessions/{session_id}/metadata")
    assert resp.status_code == 200, resp.text
    metadata = resp.json()
    assert metadata["type"] == "repo"
    assert metadata["metadata"]["session_subtype"] == "founding_member"
    assert metadata["has_founding_member"] == True

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling scenarios"""
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Test invalid session ID
        resp = await client.get(f"{BASE_URL}/founder/sessions/invalid-id/tools")
        assert resp.status_code == 404, "Invalid session ID should return 404"
        
        # Test invalid repo URL
        resp = await client.post(
            f"{BASE_URL}/founder/sessions",
            json={"repo_url": "invalid-url"}
        )
        assert resp.status_code == 400, "Invalid repo URL should return 400 with proper error message"
        error_data = resp.json()
        assert "detail" in error_data, "Error response should include detail message"
        assert "Invalid repository URL" in error_data["detail"], "Error message should indicate repository URL problem"
        
        # Test missing query parameter
        resp = await client.post(
            f"{BASE_URL}/founder/sessions/valid-id/agentic-query",
            json={}
        )
        assert resp.status_code == 422, "Missing query should return 422"

if __name__ == "__main__":
    asyncio.run(test_founding_member_agent_flow())