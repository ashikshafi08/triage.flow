"""
Triage Bot Service
Handles posting analysis results to GitHub issues as comments
"""
import logging
import re
import tempfile
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from .github_client import GitHubIssueClient
from .config import settings

logger = logging.getLogger(__name__)

class TriageBot:
    """
    GitHub bot for posting issue analysis results as comments
    """
    
    def __init__(self):
        self.github_client = GitHubIssueClient()
        self.bot_signature = self._get_bot_signature()
    
    def _get_bot_signature(self) -> str:
        """Generate bot signature for comments"""
        from .config import settings
        return f"\n\n---\n\n<div align=\"center\">\n\n🤖 **Automated Analysis by {settings.BOT_NAME}**\n\n*Intelligent issue triage and solution planning* • [View Source]({settings.BOT_REPO_URL}) • [Report Issues]({settings.BOT_REPO_URL}/issues)\n\n</div>"
    
    def _extract_code_diffs(self, plan_markdown: str) -> List[Tuple[str, str, str]]:
        """
        Extract code diffs from the solution plan
        
        Returns:
            List of tuples: (filename, language, diff_content)
        """
        diffs = []
        lines = plan_markdown.split('\n')
        in_code_block = False
        current_diff = []
        current_lang = ""
        current_filename = ""
        
        for i, line in enumerate(lines):
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block - check if it's a diff
                    diff_content = '\n'.join(current_diff)
                    if self._is_diff_content(diff_content, current_lang):
                        # Try to extract filename from the diff or context
                        base_filename = self._extract_filename_from_diff(diff_content, lines, i)
                        if base_filename:
                            # Add .diff extension to make it clear this is a diff file
                            filename = f"{base_filename}.diff"
                            diffs.append((filename, current_lang, diff_content))
                        else:
                            # Generate a generic filename
                            filename = f"changes_{len(diffs) + 1}.diff"
                            diffs.append((filename, current_lang, diff_content))
                    
                    current_diff = []
                    in_code_block = False
                    current_lang = ""
                else:
                    # Start of code block
                    current_lang = line.strip()[3:].strip()
                    in_code_block = True
            elif in_code_block:
                current_diff.append(line)
        
        return diffs
    
    def _is_diff_content(self, content: str, language: str) -> bool:
        """Check if content looks like a diff"""
        if not content.strip():
            return False
            
        return (
            # Standard diff patterns
            '---' in content and '+++' in content or
            'diff --git' in content or
            # Git diff patterns
            any(line.strip().startswith(('@@', 'index ', 'new file', 'deleted file')) for line in content.split('\n')) or
            # Line-based diff patterns
            any(line.startswith(('+', '-')) and not line.startswith(('+++', '---')) for line in content.split('\n')) or
            # Language hint
            language.lower() in ['diff', 'patch'] or
            # High ratio of + and - lines
            (len([l for l in content.split('\n') if l.startswith(('+', '-'))]) > len(content.split('\n')) * 0.2)
        )
    
    def _extract_filename_from_diff(self, diff_content: str, all_lines: List[str], current_index: int) -> Optional[str]:
        """Try to extract filename from diff content or surrounding context"""
        # Check for standard diff headers
        for line in diff_content.split('\n'):
            if line.startswith('--- a/') or line.startswith('+++ b/'):
                # Extract filename from git diff format
                filename = line.split('/')[-1] if '/' in line else line.split()[-1]
                return filename.strip()
            elif line.startswith('--- ') or line.startswith('+++ '):
                # Extract from unified diff format
                parts = line.split()
                if len(parts) > 1:
                    filename = parts[1]
                    if filename not in ['a/', 'b/', '/dev/null']:
                        return os.path.basename(filename)
        
        # Look for filename hints in surrounding text
        context_start = max(0, current_index - 10)
        context_end = min(len(all_lines), current_index + 5)
        
        for line in all_lines[context_start:context_end]:
            # Look for file mentions
            if any(ext in line.lower() for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']):
                # Extract potential filename
                words = line.split()
                for word in words:
                    if '.' in word and any(word.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']):
                        return os.path.basename(word.strip('`"\'()[]{}'))
        
        return None
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension based on language"""
        lang_map = {
            'python': '.py',
            'javascript': '.js',
            'typescript': '.ts',
            'java': '.java',
            'cpp': '.cpp',
            'c': '.c',
            'go': '.go',
            'rust': '.rs',
            'diff': '.diff',
            'patch': '.patch'
        }
        return lang_map.get(language.lower(), '.diff')
    
    async def _create_gist_for_diff(self, filename: str, content: str, description: str = "") -> Optional[str]:
        """Create a GitHub Gist for a diff file"""
        try:
            gist_data = {
                "description": description or f"Code changes for {filename}",
                "public": False,  # Private gist
                "files": {
                    filename: {
                        "content": content
                    }
                }
            }
            
            # Use the GitHub client to create gist
            gist_response = await self.github_client.create_gist(gist_data)
            if gist_response and 'html_url' in gist_response:
                return gist_response['html_url']
            
        except Exception as e:
            logger.warning(f"Failed to create gist for {filename}: {e}")
        
        return None
    
    def _format_classification(self, classification: Dict[str, Any]) -> str:
        """Format classification results for GitHub comment"""
        if not classification:
            return ""
        
        category = classification.get('category', 'unknown')
        confidence = classification.get('confidence', 0)
        reasoning = classification.get('reasoning', '')
        
        # Map categories to emojis
        category_emojis = {
            'bug-code': '🐛',
            'bug-test': '🧪',
            'documentation': '📚',
            'build/CI': '🔧',
            'dependency': '📦',
            'refactor': '♻️',
            'feature-request': '✨',
            'question': '❓',
            'enhancement': '🚀'
        }
        
        emoji = category_emojis.get(category, '🏷️')
        confidence_pct = int(confidence * 100)
        
        result = f"## {emoji} Issue Classification\n\n"
        result += f"**Category:** `{category}` ({confidence_pct}% confidence)\n\n"
        
        if reasoning:
            result += f"**Reasoning:** {reasoning}\n\n"
        
        return result
    
    def _format_pr_detection(self, pr_detection: Dict[str, Any]) -> str:
        """Format PR detection results for GitHub comment"""
        if not pr_detection:
            return ""
        
        result = "## 🔍 PR Detection Results\n\n"
        
        if not pr_detection.get('has_existing_prs', False):
            result += "✅ **No existing PRs found** - Safe to proceed with new implementation\n\n"
            return result
        
        result += "⚠️ **Existing work detected:**\n\n"
        result += f"{pr_detection.get('message', 'Related work found')}\n\n"
        
        # Direct PR link
        if pr_detection.get('pr_number'):
            pr_url = pr_detection.get('pr_url', '')
            pr_state = pr_detection.get('pr_state', 'unknown')
            result += f"**Direct PR:** [#{pr_detection['pr_number']}]({pr_url}) ({pr_state})\n\n"
        
        # Related merged PRs
        if pr_detection.get('related_merged_prs'):
            result += "**Related Merged PRs:**\n"
            for pr in pr_detection['related_merged_prs']:
                result += f"- [#{pr['pr_number']}]({pr['pr_url']}): {pr['pr_title']}\n"
            result += "\n"
        
        # Related open PRs
        if pr_detection.get('related_open_prs'):
            result += "**Related Open PRs:**\n"
            for pr in pr_detection['related_open_prs']:
                draft_badge = " (Draft)" if pr.get('draft') else ""
                result += f"- [#{pr['pr_number']}]({pr['url']}): {pr['title']} by @{pr['author']}{draft_badge}\n"
            result += "\n"
        
        result += "> 💡 Consider coordinating with existing work before proceeding\n\n"
        return result
    
    async def _format_solution_plan_with_gists(self, plan_markdown: str, issue_url: str) -> str:
        """Format solution plan for GitHub comment, creating gists for diffs"""
        if not plan_markdown:
            return ""
        
        # Extract diffs from the plan
        diffs = self._extract_code_diffs(plan_markdown)
        
        # Create gists for each diff
        gist_links = []
        for filename, language, diff_content in diffs:
            gist_url = await self._create_gist_for_diff(
                filename, 
                diff_content, 
                f"Code changes for {filename} - {issue_url}"
            )
            if gist_url:
                gist_links.append((filename, gist_url))
        
        # Format the plan without the diff code blocks
        result = "## 🎯 Solution Plan\n\n"
        
        # Clean up the plan by removing diff code blocks
        lines = plan_markdown.split('\n')
        cleaned_lines = []
        in_code_block = False
        current_diff = []
        current_lang = ""
        skip_block = False
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    if not skip_block:
                        # This is a regular code block, keep it
                        cleaned_lines.extend(current_diff)
                        cleaned_lines.append('```')
                    current_diff = []
                    in_code_block = False
                    skip_block = False
                    current_lang = ""
                else:
                    # Start of code block
                    current_lang = line.strip()[3:].strip()
                    current_diff = []
                    
                    # Check if we should skip this block (if it's a diff)
                    in_code_block = True
                    cleaned_lines.append(line)  # Add opening ```
                continue
            
            if in_code_block:
                current_diff.append(line)
                # Check if this looks like a diff block
                if not skip_block:
                    diff_content = '\n'.join(current_diff)
                    if self._is_diff_content(diff_content, current_lang):
                        # This is a diff, remove the opening ``` we just added
                        cleaned_lines.pop()
                        skip_block = True
                        current_diff = []  # Clear since we're skipping
                continue
            
            # Regular line processing
            if line.strip().startswith('# '):
                continue  # Skip top-level headers
            elif line.strip().startswith('## '):
                cleaned_lines.append(line.replace('## ', '### '))
            else:
                cleaned_lines.append(line)
        
        result += '\n'.join(cleaned_lines)
        
        # Add gist links section if we have any
        if gist_links:
            result += "\n\n### 📎 Code Changes\n\n"
            for filename, gist_url in gist_links:
                result += f"- **{filename}**: [View Changes]({gist_url})\n"
        
        result += "\n\n"
        return result
    
    def _format_solution_plan(self, plan_markdown: str) -> str:
        """Format solution plan for GitHub comment (fallback without gists)"""
        if not plan_markdown:
            return ""
        
        # Simple formatting without gists
        result = "## 🎯 Solution Plan\n\n"
        
        lines = plan_markdown.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip top-level headers
            if line.strip().startswith('# '):
                continue
            # Convert ## to ### to maintain hierarchy
            elif line.strip().startswith('## '):
                cleaned_lines.append(line.replace('## ', '### '))
            else:
                cleaned_lines.append(line)
        
        result += '\n'.join(cleaned_lines)
        result += "\n\n"
        
        return result
    
    def _format_analysis_summary(self, analysis_result: Dict[str, Any]) -> str:
        """Format a complete analysis summary for GitHub comment"""
        comment_parts = []
        
        # Header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        comment_parts.append(f"# 🔍 Triage Analysis Report\n\n*Generated on {timestamp}*\n")
        
        # Status check
        status = analysis_result.get('status', 'unknown')
        if status == 'skipped':
            reason = analysis_result.get('reason', 'Unknown reason')
            comment_parts.append(f"⚠️ **Analysis Skipped:** {reason}\n\n")
            return ''.join(comment_parts) + self.bot_signature
        elif status == 'error':
            error = analysis_result.get('error', 'Unknown error')
            comment_parts.append(f"❌ **Analysis Failed:** {error}\n\n")
            return ''.join(comment_parts) + self.bot_signature
        
        # Classification
        if analysis_result.get('final_result', {}).get('classification'):
            classification_text = self._format_classification(
                analysis_result['final_result']['classification']
            )
            if classification_text:
                comment_parts.append(classification_text)
        
        # PR Detection
        pr_detection_step = None
        for step in analysis_result.get('steps', []):
            if step.get('step') == 'PR Detection':
                pr_detection_step = step.get('result')
                break
        
        if pr_detection_step:
            pr_detection_text = self._format_pr_detection(pr_detection_step)
            if pr_detection_text:
                comment_parts.append(pr_detection_text)
        
        # Solution Plan (will be handled in the main method with gists)
        # This is a placeholder - the actual formatting happens in post_analysis_to_issue
        
        # Related Files (if available)
        related_files = analysis_result.get('final_result', {}).get('related_files')
        if related_files and isinstance(related_files, list) and len(related_files) > 0:
            comment_parts.append("## 📁 Key Files Identified\n\n")
            for file_path in related_files[:10]:  # Limit to top 10 files
                comment_parts.append(f"- `{file_path}`\n")
            comment_parts.append("\n")
        
        # Add bot signature
        comment_parts.append(self.bot_signature)
        
        return ''.join(comment_parts)
    
    async def post_analysis_to_issue(
        self, 
        issue_url: str, 
        analysis_result: Dict[str, Any],
        custom_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Post analysis results as a comment to the GitHub issue
        
        Args:
            issue_url: Full URL of the GitHub issue
            analysis_result: Analysis results from the issue analysis pipeline
            custom_message: Optional custom message to prepend
            
        Returns:
            Dictionary with success status and comment details
        """
        try:
            # Build the comment with gist-based solution plan
            comment_parts = []
            
            # Header
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
            comment_parts.append(f"# 🔍 Triage Analysis Report\n\n*Generated on {timestamp}*\n")
            
            # Status check
            status = analysis_result.get('status', 'unknown')
            if status == 'skipped':
                reason = analysis_result.get('reason', 'Unknown reason')
                comment_parts.append(f"⚠️ **Analysis Skipped:** {reason}\n\n")
                comment_body = ''.join(comment_parts) + self.bot_signature
            elif status == 'error':
                error = analysis_result.get('error', 'Unknown error')
                comment_parts.append(f"❌ **Analysis Failed:** {error}\n\n")
                comment_body = ''.join(comment_parts) + self.bot_signature
            else:
                # Classification
                if analysis_result.get('final_result', {}).get('classification'):
                    classification_text = self._format_classification(
                        analysis_result['final_result']['classification']
                    )
                    if classification_text:
                        comment_parts.append(classification_text)
                
                # PR Detection
                pr_detection_step = None
                for step in analysis_result.get('steps', []):
                    if step.get('step') == 'PR Detection':
                        pr_detection_step = step.get('result')
                        break
                
                if pr_detection_step:
                    pr_detection_text = self._format_pr_detection(pr_detection_step)
                    if pr_detection_text:
                        comment_parts.append(pr_detection_text)
                
                # Solution Plan with Gists
                if analysis_result.get('final_result', {}).get('remediation_plan'):
                    plan_text = await self._format_solution_plan_with_gists(
                        analysis_result['final_result']['remediation_plan'],
                        issue_url
                    )
                    if plan_text:
                        comment_parts.append(plan_text)
                
                # Related Files
                related_files = analysis_result.get('final_result', {}).get('related_files')
                if related_files and isinstance(related_files, list) and len(related_files) > 0:
                    comment_parts.append("## 📁 Key Files Identified\n\n")
                    for file_path in related_files[:10]:
                        comment_parts.append(f"- `{file_path}`\n")
                    comment_parts.append("\n")
                
                # Add bot signature
                comment_parts.append(self.bot_signature)
                comment_body = ''.join(comment_parts)
            
            # Add custom message if provided
            if custom_message:
                comment_body = f"{custom_message}\n\n{comment_body}"
            
            # Post the comment
            comment_data = await self.github_client.post_issue_comment(
                issue_url=issue_url,
                comment_body=comment_body
            )
            
            logger.info(f"Successfully posted analysis comment to {issue_url}")
            
            return {
                "success": True,
                "comment_id": comment_data["id"],
                "comment_url": comment_data["url"],
                "message": "Analysis posted successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to post analysis to {issue_url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to post analysis"
            }
    
    async def post_simple_comment(
        self, 
        issue_url: str, 
        message: str,
        add_signature: bool = True
    ) -> Dict[str, Any]:
        """
        Post a simple comment to a GitHub issue
        
        Args:
            issue_url: Full URL of the GitHub issue
            message: Comment message
            add_signature: Whether to add bot signature
            
        Returns:
            Dictionary with success status and comment details
        """
        try:
            comment_body = message
            if add_signature:
                comment_body += self.bot_signature
            
            comment_data = await self.github_client.post_issue_comment(
                issue_url=issue_url,
                comment_body=comment_body
            )
            
            logger.info(f"Successfully posted comment to {issue_url}")
            
            return {
                "success": True,
                "comment_id": comment_data["id"],
                "comment_url": comment_data["url"],
                "message": "Comment posted successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to post comment to {issue_url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to post comment"
            } 