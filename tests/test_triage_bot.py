"""
Tests for the Triage Bot functionality
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.triage_bot import TriageBot

@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing"""
    return {
        "status": "completed",
        "final_result": {
            "classification": {
                "category": "bug-code",
                "confidence": 0.87,
                "reasoning": "The issue describes a runtime error with specific stack trace"
            },
            "remediation_plan": "## Root Cause\nThe issue stems from...\n\n## Solution\n1. Fix the error handling",
            "related_files": ["src/components/App.tsx", "src/utils/errorHandler.ts"]
        },
        "steps": [
            {
                "step": "PR Detection",
                "status": "completed",
                "result": {
                    "has_existing_prs": False,
                    "message": "No existing PRs found"
                }
            },
            {
                "step": "Issue Classification",
                "status": "completed",
                "result": {
                    "category": "bug-code",
                    "confidence": 0.87
                }
            }
        ]
    }

@pytest.fixture
def mock_github_client():
    """Mock GitHub client"""
    with patch('src.triage_bot.GitHubIssueClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        yield mock_client

class TestTriageBot:
    """Test cases for TriageBot"""
    
    def test_bot_initialization(self):
        """Test bot initializes correctly"""
        with patch('src.triage_bot.GitHubIssueClient'):
            bot = TriageBot()
            assert bot is not None
            assert "Triage.Flow" in bot.bot_signature
    
    def test_format_classification(self):
        """Test classification formatting"""
        bot = TriageBot()
        classification = {
            "category": "bug-code",
            "confidence": 0.87,
            "reasoning": "Test reasoning"
        }
        
        result = bot._format_classification(classification)
        
        assert "🐛 Issue Classification" in result
        assert "bug-code" in result
        assert "87%" in result
        assert "Test reasoning" in result
    
    def test_format_pr_detection_no_prs(self):
        """Test PR detection formatting when no PRs found"""
        bot = TriageBot()
        pr_detection = {
            "has_existing_prs": False,
            "message": "No existing PRs found"
        }
        
        result = bot._format_pr_detection(pr_detection)
        
        assert "🔍 PR Detection Results" in result
        assert "✅ **No existing PRs found**" in result
    
    def test_format_pr_detection_with_prs(self):
        """Test PR detection formatting when PRs are found"""
        bot = TriageBot()
        pr_detection = {
            "has_existing_prs": True,
            "message": "Related work found",
            "related_open_prs": [
                {
                    "pr_number": 123,
                    "title": "Fix bug",
                    "author": "developer",
                    "url": "https://github.com/owner/repo/pull/123",
                    "draft": False
                }
            ]
        }
        
        result = bot._format_pr_detection(pr_detection)
        
        assert "⚠️ **Existing work detected:**" in result
        assert "Related Open PRs:" in result
        assert "#123" in result
        assert "Fix bug" in result
        assert "@developer" in result
    
    def test_format_solution_plan(self):
        """Test solution plan formatting"""
        bot = TriageBot()
        plan = "# Main Header\n## Sub Header\nSome content"
        
        result = bot._format_solution_plan(plan)
        
        assert "🎯 Solution Plan" in result
        assert "### Sub Header" in result  # Should convert ## to ###
        assert "# Main Header" not in result  # Should remove # headers
    
    def test_format_analysis_summary(self, sample_analysis_result):
        """Test complete analysis summary formatting"""
        bot = TriageBot()
        
        result = bot._format_analysis_summary(sample_analysis_result)
        
        assert "🔍 Triage Analysis Report" in result
        assert "🐛 Issue Classification" in result
        assert "🔍 PR Detection Results" in result
        assert "🎯 Solution Plan" in result
        assert "📁 Key Files Identified" in result
        assert "Triage.Flow" in result  # Bot signature
    
    def test_format_analysis_summary_error_status(self):
        """Test analysis summary formatting for error status"""
        bot = TriageBot()
        analysis_result = {
            "status": "error",
            "error": "Analysis failed due to network error"
        }
        
        result = bot._format_analysis_summary(analysis_result)
        
        assert "❌ **Analysis Failed:**" in result
        assert "network error" in result
        assert "Triage.Flow" in result  # Bot signature should still be present
    
    @pytest.mark.asyncio
    async def test_post_analysis_to_issue_success(self, mock_github_client, sample_analysis_result):
        """Test successful posting of analysis to GitHub issue"""
        # Setup mock response
        mock_github_client.post_issue_comment.return_value = {
            "id": 12345,
            "url": "https://github.com/owner/repo/issues/1#issuecomment-12345",
            "body": "Test comment",
            "created_at": "2024-01-01T00:00:00Z",
            "user": "triagebot"
        }
        
        bot = TriageBot()
        bot.github_client = mock_github_client
        
        result = await bot.post_analysis_to_issue(
            issue_url="https://github.com/owner/repo/issues/1",
            analysis_result=sample_analysis_result
        )
        
        assert result["success"] is True
        assert result["comment_id"] == 12345
        assert "github.com" in result["comment_url"]
        mock_github_client.post_issue_comment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_post_analysis_to_issue_failure(self, mock_github_client, sample_analysis_result):
        """Test failed posting of analysis to GitHub issue"""
        # Setup mock to raise exception
        mock_github_client.post_issue_comment.side_effect = Exception("Permission denied")
        
        bot = TriageBot()
        bot.github_client = mock_github_client
        
        result = await bot.post_analysis_to_issue(
            issue_url="https://github.com/owner/repo/issues/1",
            analysis_result=sample_analysis_result
        )
        
        assert result["success"] is False
        assert "Permission denied" in result["error"]
    
    @pytest.mark.asyncio
    async def test_post_simple_comment(self, mock_github_client):
        """Test posting a simple comment"""
        # Setup mock response
        mock_github_client.post_issue_comment.return_value = {
            "id": 12345,
            "url": "https://github.com/owner/repo/issues/1#issuecomment-12345",
            "body": "Test comment",
            "created_at": "2024-01-01T00:00:00Z",
            "user": "triagebot"
        }
        
        bot = TriageBot()
        bot.github_client = mock_github_client
        
        result = await bot.post_simple_comment(
            issue_url="https://github.com/owner/repo/issues/1",
            message="Hello from bot!"
        )
        
        assert result["success"] is True
        assert result["comment_id"] == 12345
        
        # Verify the comment includes the message and signature
        call_args = mock_github_client.post_issue_comment.call_args
        comment_body = call_args[1]["comment_body"]
        assert "Hello from bot!" in comment_body
        assert "Triage.Flow" in comment_body
    
    def test_get_bot_signature(self):
        """Test bot signature generation"""
        bot = TriageBot()
        signature = bot._get_bot_signature()
        
        assert "🤖" in signature
        assert "Triage.Flow" in signature
        assert "triage.flow" in signature.lower()
        assert signature.startswith("\n\n---\n")

if __name__ == "__main__":
    pytest.main([__file__]) 