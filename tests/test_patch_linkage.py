"""
Unit tests for patch linkage functionality
Fast feedback without GitHub I/O using mocked responses
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import Mock, AsyncMock, patch
from aiohttp import ClientResponse, ClientResponseError
import aiohttp

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patch_linkage import PatchLinkageBuilder, PatchLink, DiffDoc, DIFF_TRUNCATION_SENTINEL

class TestPatchLinkageBuilder:
    """Test the PatchLinkageBuilder class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Mock the GitHub token for testing
        with patch.dict('os.environ', {'GITHUB_TOKEN': 'test_token'}):
            self.builder = PatchLinkageBuilder("test-owner", "test-repo")
    
    def test_init(self):
        """Test builder initialization"""
        assert self.builder.repo_owner == "test-owner"
        assert self.builder.repo_name == "test-repo"
        assert self.builder.repo_key == "test-owner/test-repo"
        assert "Authorization" in self.builder.headers
        assert "groot-preview" in self.builder.commit_pr_headers["Accept"]
        assert "mockingbird-preview" in self.builder.timeline_headers["Accept"]
    
    def test_patch_link_dataclass(self):
        """Test PatchLink dataclass creation"""
        link = PatchLink(
            issue_id=123,
            pr_number=456,
            merged_at="2024-01-01T12:00:00Z",
            pr_title="Fix bug in component",
            pr_url="https://github.com/test/repo/pull/456",
            pr_diff_url="https://github.com/test/repo/pull/456.diff",
            files_changed=["src/component.js", "tests/component.test.js"]
        )
        
        assert link.issue_id == 123
        assert link.pr_number == 456
        assert link.merged_at == "2024-01-01T12:00:00Z"
        assert len(link.files_changed) == 2
    
    @pytest.mark.asyncio
    async def test_parse_timeline_events(self):
        """Test parsing timeline events to extract commit SHAs"""
        # Mock timeline events data (based on real GitHub API response)
        mock_timeline_events = [
            {
                "event": "referenced",
                "commit_id": "abc123def456",
                "actor": {"login": "testuser"}
            },
            {
                "event": "closed",
                "commit_id": "def456ghi789",
                "actor": {"login": "testuser"}
            },
            {
                "event": "labeled",
                "label": {"name": "bug"}
            }
        ]
        
        # Mock the session and methods
        session_mock = AsyncMock()
        
        # Mock _get_all_timeline_events to return our test data
        with patch.object(self.builder, '_get_all_timeline_events', return_value=mock_timeline_events):
            # Mock _find_pr_for_commit to return PR info
            mock_pr_info = {
                "number": 789,
                "title": "Fix critical bug",
                "html_url": "https://github.com/test/repo/pull/789",
                "diff_url": "https://github.com/test/repo/pull/789.diff",
                "merged_at": "2024-01-01T12:00:00Z",
                "files_changed": ["src/fix.js"]
            }
            
            with patch.object(self.builder, '_find_pr_for_commit', return_value=mock_pr_info):
                mock_issue = {"number": 123}
                
                # Test the method
                patch_links = await self.builder._get_issue_patch_links(session_mock, mock_issue)
                
                # Verify we found patch links
                assert len(patch_links) == 2  # One for referenced, one for closed
                
                # Check the first patch link
                link = patch_links[0]
                assert link.issue_id == 123
                assert link.pr_number == 789
                assert link.pr_title == "Fix critical bug"
                assert link.merged_at == "2024-01-01T12:00:00Z"
                assert "src/fix.js" in link.files_changed
    
    @pytest.mark.asyncio
    async def test_rate_limiting_backoff(self):
        """Test rate limiting backoff logic"""
        session = AsyncMock()

        # first (rate-limited) response
        resp1 = AsyncMock()
        resp1.status = 403
        resp1.headers = {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1704067200"  # Some future timestamp
        }

        # second (success) response
        resp2 = AsyncMock()
        resp2.status = 200
        resp2.json = AsyncMock(return_value={"test": "data"})

        # wrap responses in async-context-manager mocks
        cm1, cm2 = AsyncMock(), AsyncMock()
        cm1.__aenter__ = AsyncMock(return_value=resp1)
        cm1.__aexit__ = AsyncMock(return_value=None)
        cm2.__aenter__ = AsyncMock(return_value=resp2)
        cm2.__aexit__ = AsyncMock(return_value=None)

        # Configure session.get to return the context managers
        session.get = Mock(side_effect=[cm1, cm2])

        with patch("time.time", return_value=1704067100), \
             patch("asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            response = await self.builder.rate_limited_get(
                session, "https://api.github.com/test"
            )

        sleep_mock.assert_called_once()          # back-off happened
        call_args = sleep_mock.call_args[0]
        assert call_args[0] > 100                # Should sleep for at least 100+ seconds
        assert response.status == 200            # we got the 2nd response
    
    @pytest.mark.asyncio
    async def test_timeline_pagination(self):
        """Test that timeline pagination works correctly"""
        session = AsyncMock()
        
        # Mock paginated timeline responses
        page1_events = [
            {"event": "referenced", "commit_id": "abc123"},
            {"event": "labeled", "label": {"name": "bug"}}
        ]
        page2_events = [
            {"event": "closed", "commit_id": "def456"}
        ]
        
        # Create mock responses
        resp1 = AsyncMock()
        resp1.status = 200
        resp1.json = AsyncMock(return_value=page1_events)
        
        resp2 = AsyncMock()
        resp2.status = 200
        resp2.json = AsyncMock(return_value=page2_events)
        
        resp3 = AsyncMock()
        resp3.status = 200
        resp3.json = AsyncMock(return_value=[])  # Empty page indicates end
        
        # Create context managers
        cm1, cm2, cm3 = AsyncMock(), AsyncMock(), AsyncMock()
        cm1.__aenter__ = AsyncMock(return_value=resp1)
        cm1.__aexit__ = AsyncMock(return_value=None)
        cm2.__aenter__ = AsyncMock(return_value=resp2)
        cm2.__aexit__ = AsyncMock(return_value=None)
        cm3.__aenter__ = AsyncMock(return_value=resp3)
        cm3.__aexit__ = AsyncMock(return_value=None)
        
        # Configure session.get to return context managers
        session.get = Mock(side_effect=[cm1, cm2, cm3])
        
        # Test pagination
        events = await self.builder._get_all_timeline_events(session, 123)
        
        # Should have combined only the first page because its length < per_page (2 < 100)
        # This follows GitHub API best practices - stop when page length < per_page
        assert len(events) == 2
        assert events[0]["event"] == "referenced"
        assert events[1]["event"] == "labeled"
        
        # Only one API call should have been made (first page was < per_page, so pagination stopped)
        assert session.get.call_count == 1
    
    def test_save_and_load_patch_links(self):
        """Test saving and loading patch links to/from JSONL"""
        # Create test patch links
        patch_links = [
            PatchLink(
                issue_id=123,
                pr_number=456,
                merged_at="2024-01-01T12:00:00Z",
                pr_title="Fix bug A",
                pr_url="https://github.com/test/repo/pull/456",
                pr_diff_url="https://github.com/test/repo/pull/456.diff",
                files_changed=["src/a.js"]
            ),
            PatchLink(
                issue_id=123,  # Same issue, multiple PRs
                pr_number=457,
                merged_at="2024-01-02T12:00:00Z",
                pr_title="Follow-up fix for bug A",
                pr_url="https://github.com/test/repo/pull/457",
                pr_diff_url="https://github.com/test/repo/pull/457.diff",
                files_changed=["src/a.js", "tests/a.test.js"]
            )
        ]
        
        # Use a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_file = Path(f.name)
        
        try:
            # Mock the patch_links_file path
            self.builder.patch_links_file = temp_file
            
            # Test saving
            asyncio.run(self.builder._save_patch_links(patch_links))
            
            # Verify file was created and has correct content
            assert temp_file.exists()
            
            # Verify JSONL format
            with open(temp_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 2
                
                # Parse first line
                first_record = json.loads(lines[0].strip())
                assert first_record["issue_id"] == 123
                assert first_record["pr_number"] == 456
                assert "created_at" in first_record
            
            # Test loading
            loaded_links = self.builder.load_patch_links()
            
            # Verify we got back the same data
            assert 123 in loaded_links
            assert len(loaded_links[123]) == 2
            
            first_link = loaded_links[123][0]
            assert first_link.pr_number == 456
            assert first_link.pr_title == "Fix bug A"
            
            second_link = loaded_links[123][1]
            assert second_link.pr_number == 457
            assert len(second_link.files_changed) == 2
            
        finally:
            # Clean up temp file
            if temp_file.exists():
                temp_file.unlink()
    
    def test_get_patch_url_for_issue(self):
        """Test getting patch URL for a specific issue"""
        # Mock load_patch_links to return test data
        test_data = {
            123: [
                PatchLink(
                    issue_id=123,
                    pr_number=456,
                    merged_at="2024-01-01T12:00:00Z",
                    pr_title="Fix bug",
                    pr_url="https://github.com/test/repo/pull/456",
                    pr_diff_url="https://github.com/test/repo/pull/456.diff",
                    files_changed=["src/fix.js"]
                )
            ]
        }
        
        with patch.object(self.builder, 'load_patch_links', return_value=test_data):
            # Test getting URL for existing issue
            url = self.builder.get_patch_url_for_issue(123)
            assert url == "https://github.com/test/repo/pull/456.diff"
            
            # Test getting URL for non-existent issue
            url = self.builder.get_patch_url_for_issue(999)
            assert url is None
    
    def test_multiple_patch_links_for_issue(self):
        """Test that issues can have multiple patch links (one issue → multiple PRs)"""
        # Test data with multiple PRs for the same issue
        test_data = {
            123: [
                PatchLink(
                    issue_id=123,
                    pr_number=456,
                    merged_at="2024-01-01T12:00:00Z",
                    pr_title="Initial fix",
                    pr_url="https://github.com/test/repo/pull/456",
                    pr_diff_url="https://github.com/test/repo/pull/456.diff",
                    files_changed=["src/fix.js"]
                ),
                PatchLink(
                    issue_id=123,
                    pr_number=457,
                    merged_at="2024-01-02T12:00:00Z",
                    pr_title="Follow-up fix",
                    pr_url="https://github.com/test/repo/pull/457",
                    pr_diff_url="https://github.com/test/repo/pull/457.diff",
                    files_changed=["src/fix.js", "tests/fix.test.js"]
                )
            ]
        }
        
        with patch.object(self.builder, 'load_patch_links', return_value=test_data):
            # Should return the first (primary) patch URL
            url = self.builder.get_patch_url_for_issue(123)
            assert url == "https://github.com/test/repo/pull/456.diff"
            
            # But the data structure should preserve all links
            links = self.builder.load_patch_links()
            assert len(links[123]) == 2
            assert links[123][0].pr_number == 456
            assert links[123][1].pr_number == 457

    def test_extract_diff_hunks(self):
        """Test diff hunk extraction and cleaning for embedding"""
        # Sample diff content (simplified but realistic)
        sample_diff = """diff --git a/src/component.js b/src/component.js
index 1234567..abcdefg 100644
--- a/src/component.js
+++ b/src/component.js
@@ -10,7 +10,7 @@ export function Component() {
   return (
     <div>
-      <span>Old text</span>
+      <span>New text</span>
     </div>
   )
 }
@@ -25,4 +25,5 @@ export function Component() {
 export function AnotherComponent() {
   return (
     <div>Another component</div>
+    <div>Added feature</div>
   )
 }"""
        
        summary = self.builder._extract_diff_hunks(sample_diff)
        
        # Should contain essential parts
        assert "src/component.js" in summary
        assert "-      <span>Old text</span>" in summary
        assert "+      <span>New text</span>" in summary
        assert "+    <div>Added feature</div>" in summary
        
        # Should have metadata header
        assert "Patch summary for PR" in summary
        assert "Files changed:" in summary
        
        # Should be within size limit
        assert len(summary) <= 8000

    def test_extract_diff_hunks_truncation(self):
        """Test that very large diffs get truncated properly"""
        # Create a large diff that would exceed the limit
        large_diff_lines = ["diff --git a/large.js b/large.js", "--- a/large.js", "+++ b/large.js"]
        
        # Add many hunk lines to exceed the 8KB limit
        for i in range(1000):
            large_diff_lines.extend([
                f"@@ -{i*10},{i*10+5} +{i*10},{i*10+5} @@ function test{i}() {{",
                f" function test{i}() {{",
                f"-  // old comment {i}",
                f"+  // new comment {i}",
                f"   return {i}",
                f" }}"
            ])
        
        large_diff = "\n".join(large_diff_lines)
        
        summary = self.builder._extract_diff_hunks(large_diff, max_chars=1000)
        
        # Should be truncated
        assert len(summary) <= 1000
        assert DIFF_TRUNCATION_SENTINEL in summary

    @pytest.mark.asyncio
    async def test_download_single_diff_success(self):
        """Test successful diff download and processing"""
        session = AsyncMock()
        
        # Mock successful response
        sample_diff_content = """diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,3 @@
 def test():
-    return False
+    return True
"""
        
        resp = AsyncMock()
        resp.status = 200
        resp.text = AsyncMock(return_value=sample_diff_content)
        
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        
        session.get = Mock(return_value=cm)
        
        # Create test patch link
        link = PatchLink(
            issue_id=123,
            pr_number=456,
            merged_at="2024-01-01T12:00:00Z",
            pr_title="Fix test",
            pr_url="https://github.com/test/repo/pull/456",
            pr_diff_url="https://github.com/test/repo/pull/456.diff",
            files_changed=["test.py"]
        )
        
        # Use temporary directory for diff storage
        with tempfile.TemporaryDirectory() as temp_dir:
            self.builder.diffs_dir = Path(temp_dir)
            
            diff_doc = await self.builder._download_single_diff(session, link)
            
            assert diff_doc is not None
            assert diff_doc.pr_number == 456
            assert diff_doc.issue_id == 123
            assert "test.py" in diff_doc.files_changed
            assert diff_doc.diff_text == sample_diff_content
            assert "return True" in diff_doc.diff_summary
            assert Path(diff_doc.diff_path).exists()

    @pytest.mark.asyncio
    async def test_download_single_diff_404(self):
        """Test handling of 404 when downloading diff"""
        session = AsyncMock()
        
        # Mock 404 response
        resp = AsyncMock()
        resp.status = 404
        
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        
        session.get = Mock(return_value=cm)
        
        link = PatchLink(
            issue_id=123,
            pr_number=456,
            merged_at="2024-01-01T12:00:00Z",
            pr_title="Fix test",
            pr_url="https://github.com/test/repo/pull/456",
            pr_diff_url="https://github.com/test/repo/pull/456.diff",
            files_changed=["test.py"]
        )
        
        diff_doc = await self.builder._download_single_diff(session, link)
        
        # Should return None for 404, not raise exception
        assert diff_doc is None

    @pytest.mark.asyncio
    async def test_verify_connectivity_success(self):
        """Test successful connectivity verification"""
        session = AsyncMock()
        
        # Mock repository metadata response
        repo_resp = AsyncMock()
        repo_resp.status = 200
        repo_resp.json = AsyncMock(return_value={
            "full_name": "test-owner/test-repo",
            "stargazers_count": 1000,
            "forks_count": 200
        })
        
        # Mock issues test response
        issues_resp = AsyncMock()
        issues_resp.status = 200
        issues_resp.json = AsyncMock(return_value=[{"number": 1, "title": "Test issue"}])
        
        # Mock context managers
        repo_cm = AsyncMock()
        repo_cm.__aenter__ = AsyncMock(return_value=repo_resp)
        repo_cm.__aexit__ = AsyncMock(return_value=None)
        
        issues_cm = AsyncMock()
        issues_cm.__aenter__ = AsyncMock(return_value=issues_resp)
        issues_cm.__aexit__ = AsyncMock(return_value=None)
        
        session.get = Mock(side_effect=[repo_cm, issues_cm])
        
        # Should not raise
        await self.builder._verify_connectivity(session)
        
        # Should have made 2 calls
        assert session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_verify_connectivity_repo_not_found(self):
        """Test connectivity verification with 404 repo"""
        session = AsyncMock()
        
        resp = AsyncMock()
        resp.status = 404
        
        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=resp)
        cm.__aexit__ = AsyncMock(return_value=None)
        
        session.get = Mock(return_value=cm)
        
        with pytest.raises(RuntimeError, match="Repository .* not found"):
            await self.builder._verify_connectivity(session)

# Integration test helper
def test_mock_github_api_flow():
    """Test the complete flow with mocked GitHub API responses"""
    with patch.dict('os.environ', {'GITHUB_TOKEN': 'test_token'}):
        builder = PatchLinkageBuilder("facebook", "react")
        assert builder.repo_key == "facebook/react"
        assert "groot-preview" in builder.commit_pr_headers["Accept"]

if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"]) 