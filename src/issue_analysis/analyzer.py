"""High-level orchestrator that turns a *single* GitHub issue into an actionable
analysis + plan.  It wires together:

* PRChecker – skip issues already being worked on;
* GitHubIssueClient – fetch issue details;
* IssueClassifier – label & confidence;
* AgenticRAGSystem – loads repo, produces rich context & exposes Explorer;
* PlanGenerator – final Markdown plan.

The function ``analyse_issue`` is async so it can be used inside background
workers.
"""
from __future__ import annotations

import re
import json
import logging
from typing import Dict, Any

from ..github_client import GitHubIssueClient
from ..agentic_rag import AgenticRAGSystem
from ..llm_client import format_rag_context_for_llm

from .pr_checker import PRChecker
from .classifier import IssueClassifier
from .plan_generator import PlanGenerator, PlanInput
from ..patch_linkage import PatchLinkageBuilder

logger = logging.getLogger(__name__)

_RE_ISSUE = re.compile(r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/issues/(?P<num>\d+)")


def _extract_repo(issue_url: str) -> str:
    m = _RE_ISSUE.search(issue_url)
    if not m:
        raise ValueError("Invalid GitHub issue URL")
    return f"https://github.com/{m.group('owner')}/{m.group('repo')}"


async def analyse_issue(issue_url: str) -> Dict[str, Any]:
    """Full agentic pipeline that leverages sophisticated agent tools for deep analysis."""

    # 1. Check for existing PRs using GitHub API
    pr_info = await PRChecker.from_issue_url(issue_url)
    
    # 1.5. Enhanced PR detection using patch linkage system
    enhanced_pr_info = None
    try:
        # Extract repo info from issue URL
        match = re.search(r"github\.com/([^/]+)/([^/]+)/issues/(\d+)", issue_url)
        if match:
            owner, repo, issue_number = match.groups()
            issue_number = int(issue_number)
            
            # Check for related PRs in our index
            patch_builder = PatchLinkageBuilder(owner, repo)
            
            # Load existing patch links
            patch_links = patch_builder.load_patch_links()
            related_prs = patch_links.get(issue_number, [])
            
            # Load open PRs and check for mentions of this issue
            open_prs = patch_builder.load_open_prs()
            related_open_prs = []
            
            for pr in open_prs:
                # Check if PR body or title mentions this issue
                pr_text = f"{pr.title} {pr.body}".lower()
                if f"#{issue_number}" in pr_text or f"issue {issue_number}" in pr_text:
                    related_open_prs.append(pr)
            
            if related_prs or related_open_prs:
                enhanced_pr_info = {
                    "has_related_work": True,
                    "related_merged_prs": related_prs,
                    "related_open_prs": related_open_prs,
                    "message": f"Found {len(related_prs)} related merged PR(s) and {len(related_open_prs)} related open PR(s)"
                }
                
                # If we found significant related work, consider skipping
                if len(related_open_prs) > 0:
                    logger.info("Issue %s has related open PRs: %s", issue_url, [pr.pr_number for pr in related_open_prs])
                    return {
                        "status": "skipped",
                        "reason": "related_open_prs",
                        "pr_info": pr_info.__dict__ if pr_info.state else None,
                        "enhanced_pr_info": enhanced_pr_info,
                    }
            else:
                enhanced_pr_info = {
                    "has_related_work": False,
                    "message": "No related PRs found in patch linkage index"
                }
                
    except Exception as e:
        logger.warning(f"Enhanced PR detection failed: {e}")
        enhanced_pr_info = {
            "has_related_work": False,
            "message": "Enhanced PR detection unavailable",
            "error": str(e)
        }
    
    # Original PR check - skip if direct PR exists
    if pr_info.state is not None:
        logger.info("Issue %s already has PR state=%s", issue_url, pr_info.state)
        return {
            "status": "skipped",
            "reason": "pr_exists",
            "pr_info": pr_info.__dict__,
            "enhanced_pr_info": enhanced_pr_info,
        }

    # 2. Fetch issue data
    github = GitHubIssueClient()
    issue_resp = await github.get_issue(issue_url)
    if issue_resp.status != "success" or not issue_resp.data:
        return {
            "status": "error",
            "error": issue_resp.error or "Could not fetch issue",
        }
    issue = issue_resp.data

    # 3. Quick classification for initial categorization
    classifier = IssueClassifier()
    classification = await classifier.classify(issue.title, issue.body)

    # 4. Create RAG + Agentic Explorer
    session_id = f"issue_{issue.number}"
    rag = AgenticRAGSystem(session_id=session_id)
    repo_url = _extract_repo(issue_url)
    await rag.initialize_core_systems(repo_url)

    # 5. **AGENTIC ANALYSIS** - This is where the magic happens!
    try:
        agentic_prompt = f"""
You are an expert codebase analysis agent with deep technical expertise. Your goal is to provide the most accurate and comprehensive analysis possible.

**ISSUE TO ANALYZE:**
- **Title:** {issue.title}
- **Number:** #{issue.number}
- **Body:** {issue.body or 'No description provided'}

**COMPREHENSIVE ANALYSIS APPROACH:**
Perform thorough, step-by-step investigation to provide the most accurate solution:

1. **Deep Issue Analysis**: 
   - Use `analyze_github_issue` with "#{issue.number}" for detailed classification
   - Understand the technical nature and underlying cause

2. **Comprehensive File Discovery**: 
   - Use `find_issue_related_files` with depth="deep" to identify all relevant files
   - Use `analyze_file_structure` on key files to understand their implementation details
   - Use `search_codebase` to find specific patterns, functions, or concepts mentioned

3. **Historical Intelligence**: 
   - Use `related_issues` to find similar past issues and learn from their solutions
   - Use `regression_detector` to determine if this is a regression
   - If specific functions/classes are mentioned, use `who_implemented_this` and `get_function_evolution`

4. **Technical Deep Dive**:
   - For code-related issues, examine the actual implementation details
   - Look for patterns like regex flags, configuration issues, API changes
   - Understand the root cause at a technical level

5. **Validation & Context**:
   - Use `check_issue_status_and_linked_pr` for current status
   - Cross-reference with similar resolved issues using `get_issue_resolution_summary`

**OUTPUT FORMAT:**
Provide a comprehensive JSON analysis with technical depth:
{{
    "issue_analysis": {{
        "classification": "Primary issue type with technical details",
        "confidence": 0.95,
        "complexity": "low|medium|high",
        "root_cause": "Detailed technical explanation of the underlying issue",
        "affected_components": ["component1", "component2"]
    }},
    "technical_investigation": {{
        "key_files_analyzed": ["file1.py", "file2.js"],
        "implementation_details": "What the code actually does and why it fails",
        "specific_patterns_found": ["regex patterns", "API calls", "configurations"],
        "technical_root_cause": "Precise technical explanation"
    }},
    "solution_strategy": {{
        "approach": "Detailed technical solution with specific implementation details",
        "entry_point": "file.py:line123",
        "specific_changes_needed": ["exact code changes", "configuration updates"],
        "effort": "low|medium|high",
        "testing_approach": "How to verify the fix works"
    }},
    "context": {{
        "similar_issues": ["#123", "#456"],
        "is_regression": false,
        "historical_solutions": "What worked for similar issues"
    }}
}}

**PRIORITY: Technical accuracy and comprehensive analysis over speed. Take the time needed to provide the most precise solution.**
"""

        logger.info("Starting comprehensive agentic issue analysis...")
        agentic_result = await rag.agentic_explorer.query(agentic_prompt)
        
        # Parse the agentic analysis result
        try:
            # The agent should return JSON, but let's be defensive
            if isinstance(agentic_result, dict):
                analysis_data = agentic_result.get("final_answer", agentic_result)
            else:
                # Try to extract JSON from the response
                import json
                json_match = re.search(r'\{.*\}', str(agentic_result), re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                else:
                    # Fallback - create structured data from text response
                    analysis_data = {
                        "issue_analysis": {"classification": classification.label, "confidence": classification.confidence},
                        "agentic_response": str(agentic_result)
                    }
        except Exception as e:
            logger.warning(f"Failed to parse agentic analysis JSON: {e}")
            analysis_data = {
                "issue_analysis": {"classification": classification.label, "confidence": classification.confidence},
                "agentic_response": str(agentic_result),
                "parse_error": str(e)
            }

    except Exception as e:
        logger.error(f"Agentic analysis failed: {e}")
        analysis_data = {
            "error": f"Agentic analysis failed: {str(e)}",
            "fallback_classification": classification.__dict__
        }

    # 6. Enhanced RAG context (still useful for plan generation)
    rag_context = await rag.get_enhanced_context(
        query=f"{issue.title}\n{issue.body or ''}",
        use_agentic_tools=True,
        include_issue_context=True,
    )
    rag_context_str = format_rag_context_for_llm(rag_context)

    # 7. Generate comprehensive plan using both agentic analysis and RAG context
    planner = PlanGenerator()
    plan_md = await planner.generate(
        PlanInput(
            issue_url=issue.url,
            title=issue.title,
            body=issue.body,
            classification=classification.label,
            explorer_json=analysis_data,  # Use agentic analysis instead of simple explorer_json
            rag_context=rag_context_str,
        )
    )

    await rag.cleanup()

    return {
        "status": "completed",
        "classification": classification.__dict__,
        "agentic_analysis": analysis_data,  # Rich agentic analysis results
        "plan_markdown": plan_md,
        "rag": rag_context,
        "enhanced_pr_info": enhanced_pr_info,
    }


async def analyse_issue_with_existing_rag(
    issue_url: str, 
    existing_rag: AgenticRAGSystem
) -> Dict[str, Any]:
    """Agentic analysis that reuses existing RAG system for better performance."""
    
    logger.info("Using existing RAG system - leveraging agentic tools with optimized performance")

    # 1. Check for existing PRs using GitHub API
    pr_info = await PRChecker.from_issue_url(issue_url)
    
    # 1.5. Enhanced PR detection using patch linkage system
    enhanced_pr_info = None
    try:
        import re
        
        # Extract repo info from issue URL
        match = re.search(r"github\.com/([^/]+)/([^/]+)/issues/(\d+)", issue_url)
        if match:
            owner, repo, issue_number = match.groups()
            issue_number = int(issue_number)
            
            # Check for related PRs in our index
            patch_builder = PatchLinkageBuilder(owner, repo)
            
            # Load existing patch links
            patch_links = patch_builder.load_patch_links()
            related_prs = patch_links.get(issue_number, [])
            
            # Load open PRs and check for mentions of this issue
            open_prs = patch_builder.load_open_prs()
            related_open_prs = []
            
            for pr in open_prs:
                # Check if PR body or title mentions this issue
                pr_text = f"{pr.title} {pr.body}".lower()
                if f"#{issue_number}" in pr_text or f"issue {issue_number}" in pr_text:
                    related_open_prs.append(pr)
            
            if related_prs or related_open_prs:
                enhanced_pr_info = {
                    "has_related_work": True,
                    "related_merged_prs": related_prs,
                    "related_open_prs": related_open_prs,
                    "message": f"Found {len(related_prs)} related merged PR(s) and {len(related_open_prs)} related open PR(s)"
                }
                
                # If we found significant related work, consider skipping
                if len(related_open_prs) > 0:
                    logger.info("Issue %s has related open PRs: %s", issue_url, [pr.pr_number for pr in related_open_prs])
                    return {
                        "status": "skipped",
                        "reason": "related_open_prs",
                        "pr_info": pr_info.__dict__ if pr_info.state else None,
                        "enhanced_pr_info": enhanced_pr_info,
                    }
            else:
                enhanced_pr_info = {
                    "has_related_work": False,
                    "message": "No related PRs found in patch linkage index"
                }
                
    except Exception as e:
        logger.warning(f"Enhanced PR detection failed: {e}")
        enhanced_pr_info = {
            "has_related_work": False,
            "message": "Enhanced PR detection unavailable",
            "error": str(e)
        }
    
    # Original PR check - skip if direct PR exists
    if pr_info.state is not None:
        logger.info("Issue %s already has PR state=%s", issue_url, pr_info.state)
        return {
            "status": "skipped",
            "reason": "pr_exists", 
            "pr_info": pr_info.__dict__,
            "enhanced_pr_info": enhanced_pr_info,
        }

    # 2. Fetch issue data
    github = GitHubIssueClient()
    issue_resp = await github.get_issue(issue_url)
    if issue_resp.status != "success" or not issue_resp.data:
        return {
            "status": "error",
            "error": issue_resp.error or "Could not fetch issue",
        }
    issue = issue_resp.data

    # 3. Quick classification
    classifier = IssueClassifier()
    classification = await classifier.classify(issue.title, issue.body)

    # 4. **AGENTIC ANALYSIS** using existing RAG system
    try:
        agentic_prompt = f"""
You are an expert codebase analysis agent with deep technical expertise. Your goal is to provide the most accurate and comprehensive analysis possible.

**ISSUE TO ANALYZE:**
- **Title:** {issue.title}
- **Number:** #{issue.number}
- **Body:** {issue.body or 'No description provided'}

**COMPREHENSIVE ANALYSIS APPROACH:**
Perform thorough, step-by-step investigation to provide the most accurate solution:

1. **Deep Issue Analysis**: 
   - Use `analyze_github_issue` with "#{issue.number}" for detailed classification
   - Understand the technical nature and underlying cause

2. **Comprehensive File Discovery**: 
   - Use `find_issue_related_files` with depth="deep" to identify all relevant files
   - Use `analyze_file_structure` on key files to understand their implementation details
   - Use `search_codebase` to find specific patterns, functions, or concepts mentioned

3. **Historical Intelligence**: 
   - Use `related_issues` to find similar past issues and learn from their solutions
   - Use `regression_detector` to determine if this is a regression
   - If specific functions/classes are mentioned, use `who_implemented_this` and `get_function_evolution`

4. **Technical Deep Dive**:
   - For code-related issues, examine the actual implementation details
   - Look for patterns like regex flags, configuration issues, API changes
   - Understand the root cause at a technical level

5. **Validation & Context**:
   - Use `check_issue_status_and_linked_pr` for current status
   - Cross-reference with similar resolved issues using `get_issue_resolution_summary`

**OUTPUT FORMAT:**
Provide a comprehensive JSON analysis with technical depth:
{{
    "issue_analysis": {{
        "classification": "Primary issue type with technical details",
        "confidence": 0.95,
        "complexity": "low|medium|high",
        "root_cause": "Detailed technical explanation of the underlying issue",
        "affected_components": ["component1", "component2"]
    }},
    "technical_investigation": {{
        "key_files_analyzed": ["file1.py", "file2.js"],
        "implementation_details": "What the code actually does and why it fails",
        "specific_patterns_found": ["regex patterns", "API calls", "configurations"],
        "technical_root_cause": "Precise technical explanation"
    }},
    "solution_strategy": {{
        "approach": "Detailed technical solution with specific implementation details",
        "entry_point": "file.py:line123",
        "specific_changes_needed": ["exact code changes", "configuration updates"],
        "effort": "low|medium|high",
        "testing_approach": "How to verify the fix works"
    }},
    "context": {{
        "similar_issues": ["#123", "#456"],
        "is_regression": false,
        "historical_solutions": "What worked for similar issues"
    }}
}}

**PRIORITY: Technical accuracy and comprehensive analysis over speed. Take the time needed to provide the most precise solution.**
"""

        logger.info("Starting comprehensive agentic analysis with existing RAG...")
        agentic_result = await existing_rag.agentic_explorer.query(agentic_prompt)
        
        # Parse the agentic analysis result
        try:
            if isinstance(agentic_result, dict):
                analysis_data = agentic_result.get("final_answer", agentic_result)
            else:
                # Try to extract JSON from response
                import json
                json_match = re.search(r'\{.*\}', str(agentic_result), re.DOTALL)
                if json_match:
                    analysis_data = json.loads(json_match.group())
                else:
                    # Fallback with agentic response
                    analysis_data = {
                        "issue_analysis": {
                            "classification": classification.label, 
                            "confidence": classification.confidence
                        },
                        "agentic_response": str(agentic_result)
                    }
        except Exception as e:
            logger.warning(f"Failed to parse agentic analysis JSON: {e}")
            analysis_data = {
                "issue_analysis": {
                    "classification": classification.label,
                    "confidence": classification.confidence
                },
                "agentic_response": str(agentic_result),
                "parse_error": str(e)
            }

    except Exception as e:
        logger.error(f"Agentic analysis with existing RAG failed: {e}")
        analysis_data = {
            "error": f"Agentic analysis failed: {str(e)}",
            "fallback_classification": classification.__dict__
        }

    # 5. Enhanced RAG context
    rag_context = await existing_rag.get_enhanced_context(
        query=f"{issue.title}\n{issue.body or ''}",
        use_agentic_tools=True,
        include_issue_context=True,
    )
    rag_context_str = format_rag_context_for_llm(rag_context)

    # 6. Generate comprehensive plan 
    planner = PlanGenerator()
    plan_md = await planner.generate(
        PlanInput(
            issue_url=issue.url,
            title=issue.title,
            body=issue.body,
            classification=classification.label,
            explorer_json=analysis_data,
            rag_context=rag_context_str,
        )
    )

    return {
        "status": "completed",
        "classification": classification.__dict__,
        "agentic_analysis": analysis_data,
        "plan_markdown": plan_md,
        "rag": rag_context,
        "enhanced_pr_info": enhanced_pr_info,
    }
