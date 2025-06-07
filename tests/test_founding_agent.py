import pytest
from unittest.mock import MagicMock
from src.founding_member_agent import FoundingMemberAgent
import os
import asyncio
import json

@pytest.fixture
def mock_code_rag():
    mock = MagicMock()
    mock.repo_info = {'repo_path': '/tmp/repo'}
    return mock

@pytest.fixture
def mock_issue_rag():
    mock = MagicMock()
    mock.indexer.patch_builder.load_patch_links.return_value = {
        1: [MagicMock(pr_number=42, pr_title="Fix bug", pr_url="url", merged_at="2023-01-01T00:00:00Z", files_changed=["foo.py"])]
    }
    mock.indexer.diff_docs = {
        42: MagicMock(diff_summary="Fixed foo.py", files_changed=["foo.py"], merged_at="2023-01-01T00:00:00Z", pr_number=42)
    }
    mock.indexer.issue_docs = {
        1: MagicMock(title="Bug in foo.py")
    }
    mock.repo_owner = "testowner"
    mock.repo_name = "testrepo"
    return mock

def test_tool_registration(mock_code_rag, mock_issue_rag):
    agent = FoundingMemberAgent("session", mock_code_rag, mock_issue_rag)
    tool_names = [tool.metadata.name for tool in agent.tools]
    assert "get_file_history" in tool_names
    assert "summarize_feature_evolution" in tool_names
    assert "who_fixed_this" in tool_names
    assert "regression_detector" in tool_names

def test_get_file_history(mock_code_rag, mock_issue_rag):
    agent = FoundingMemberAgent("session", mock_code_rag, mock_issue_rag)
    result = agent.get_file_history("foo.py")
    assert "history" in result
    assert "foo.py" in result

def test_chunking(mock_code_rag, mock_issue_rag):
    agent = FoundingMemberAgent("session", mock_code_rag, mock_issue_rag)
    long_content = "A" * 9000
    chunked = agent._chunk_large_output(long_content)
    assert "chunk_id" in chunked

def test_summarize_feature_evolution(mock_code_rag, mock_issue_rag):
    agent = FoundingMemberAgent("session", mock_code_rag, mock_issue_rag)
    # Patch issue and diff docs for feature query
    mock_issue_rag.indexer.issue_docs = {
        1: MagicMock(id=1, title="Add login", body="Implements login", created_at="2023-01-01T00:00:00Z", closed_at="2023-01-02T00:00:00Z", labels=["feature"])
    }
    mock_issue_rag.indexer.diff_docs = {
        42: MagicMock(pr_number=42, issue_id=1, files_changed=["auth.py"], diff_summary="Added login", merged_at="2023-01-02T00:00:00Z")
    }
    mock_issue_rag.repo_owner = "testowner"
    mock_issue_rag.repo_name = "testrepo"
    result = agent.summarize_feature_evolution("login")
    assert "timeline" in result
    assert "login" in result

def test_who_fixed_this_patch_linkage(monkeypatch, mock_code_rag, mock_issue_rag):
    agent = FoundingMemberAgent("session", mock_code_rag, mock_issue_rag)
    # Patch os.path.realpath to always return a path inside the repo root
    monkeypatch.setattr(os.path, "realpath", lambda path: "/tmp/repo/foo.py")
    result = agent.who_fixed_this("foo.py")
    assert "pr_number" in result or "No PR found" in result or "history" in result

def test_who_fixed_this_git_blame(monkeypatch, mock_code_rag, mock_issue_rag):
    agent = FoundingMemberAgent("session", mock_code_rag, mock_issue_rag)
    mock_issue_rag.indexer.diff_docs = {}
    # Patch os.path.realpath
    monkeypatch.setattr(os.path, "realpath", lambda path: "/tmp/repo/foo.py")
    # Patch subprocess.run to simulate git blame output
    def fake_run(cmd, capture_output, text):
        class Result:
            returncode = 0
            stdout = "abcd1234 (Alice 2023-01-01) print('hello')"
        return Result()
    monkeypatch.setattr("subprocess.run", fake_run)
    result = agent.who_fixed_this("foo.py", line_number=1)
    assert "git_blame" in result or "No PR found" in result

def test_regression_detector(mock_code_rag, mock_issue_rag):
    agent = FoundingMemberAgent("session", mock_code_rag, mock_issue_rag)
    # Patch retriever to return a closed issue with a patch
    mock_issue = MagicMock(id=1, title="Bug", patch_url="https://github.com/testowner/testrepo/pull/42", closed_at="2023-01-02T00:00:00Z")
    mock_result = MagicMock(issue=mock_issue)
    mock_issue_rag.retriever.find_related_issues.return_value = ([mock_result], None)
    mock_issue_rag.indexer.diff_docs = {
        42: MagicMock(pr_number=42, files_changed=["foo.py"], merged_at="2023-01-02T00:00:00Z")
    }
    result = agent.regression_detector("Bug")
    assert "regression_candidates" in result

@pytest.mark.asyncio
def test_agentic_answer(monkeypatch, mock_code_rag, mock_issue_rag):
    agent = FoundingMemberAgent("session", mock_code_rag, mock_issue_rag)
    # Patch agent.achat to simulate a reasoning trace
    class FakeResponse:
        def __str__(self):
            return "final answer"
    async def fake_achat(query):
        return FakeResponse()
    agent.agent.achat = fake_achat
    result = asyncio.run(agent.agentic_answer("Who changed foo.py?"))
    result_json = json.loads(result)
    assert "steps" in result_json
    assert "final_answer" in result_json
    assert "final answer" in result_json["final_answer"]

# You can add more tests for summarize_feature_evolution, who_fixed_this, regression_detector, and agentic_answer