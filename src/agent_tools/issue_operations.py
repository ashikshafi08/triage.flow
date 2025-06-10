# src/agent_tools/issue_operations.py

import json
import logging
import asyncio
import re
from typing import List, Dict, Any, Optional, Annotated, TYPE_CHECKING
from pathlib import Path
import concurrent.futures

if TYPE_CHECKING:
    from ..issue_rag import IssueAwareRAG, IssueContext # Assuming IssueContext is defined there
    from ..git_tools import IssueClosingTools
    from .search_operations import SearchOperations # For find_issue_related_files
    from .pr_operations import PROperations # For check_issue_status_and_linked_pr
    from .utilities import get_repo_url_from_path_func # Renaming to avoid conflict

# For standalone utilities if needed
from .utilities import get_repo_url_from_path as get_repo_url_from_path_util

logger = logging.getLogger(__name__)

class IssueOperations:
    def __init__(self, 
                 repo_path: Path,
                 issue_rag_system: Optional['IssueAwareRAG'],
                 issue_closing_tools: 'IssueClosingTools',
                 search_ops: 'SearchOperations', # For find_issue_related_files
                 # pr_ops: 'PROperations', # For check_issue_status_and_linked_pr - will be added later
                 get_repo_url_from_path_func: callable
                ):
        self.repo_path = repo_path
        self.issue_rag_system = issue_rag_system
        self.issue_closing_tools = issue_closing_tools
        self.search_ops = search_ops
        # self.pr_ops = pr_ops 
        self._get_repo_url_from_path = get_repo_url_from_path_func

    def analyze_github_issue(
        self,
        issue_identifier: Annotated[str, "Issue number (#123) or full GitHub issue URL to analyze"]
    ) -> str:
        try:
            from ..github_client import GitHubIssueClient # Local import to avoid circularity if client uses these tools
            
            github_client = GitHubIssueClient() # Consider if this should be passed in
            
            if issue_identifier.startswith('#') or issue_identifier.isdigit():
                issue_number = issue_identifier.lstrip('#')
                repo_url = self._get_repo_url_from_path(self.repo_path)
                if not repo_url:
                    return json.dumps({"error": "Cannot determine repository URL", "suggestion": "Provide full issue URL."}, indent=2)
                issue_url = f"{repo_url}/issues/{issue_number}"
            else:
                issue_url = issue_identifier
            
            # This was asyncio.run, ensure it's handled correctly if called from async context
            # For now, assuming direct call or proper async handling in the agent
            loop = asyncio.get_event_loop()
            if loop.is_running():
                 # If in an event loop, use a thread pool to run the blocking asyncio.run
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, github_client.get_issue(issue_url))
                    issue_response = future.result(timeout=30)
            else:
                issue_response = asyncio.run(github_client.get_issue(issue_url))

            if issue_response.status != "success" or not issue_response.data:
                return json.dumps({"error": f"Failed to fetch issue: {issue_response.error}", "issue_identifier": issue_identifier}, indent=2)
            
            analysis = self._perform_issue_analysis(issue_response.data)
            return json.dumps(analysis, indent=2)
        except Exception as e:
            logger.error(f"Error analyzing GitHub issue {issue_identifier}: {e}")
            return json.dumps({"error": f"Failed to analyze issue: {str(e)}", "issue_identifier": issue_identifier}, indent=2)

    def _perform_issue_analysis(self, issue_data: Any) -> Dict[str, Any]: # issue_data is the raw data from GitHub client
        # Assuming issue_data has attributes like number, title, state, created_at, labels, assignees, comments, body
        analysis = {
            "issue_metadata": {
                "number": issue_data.number, "title": issue_data.title, "state": issue_data.state,
                "created_at": issue_data.created_at.isoformat() if hasattr(issue_data.created_at, 'isoformat') else str(issue_data.created_at),
                "labels": issue_data.labels, "assignees": issue_data.assignees,
                "comments_count": len(issue_data.comments or [])
            },
            "issue_classification": self._classify_issue_type(issue_data),
            "complexity_assessment": self._assess_issue_complexity(issue_data),
            "technical_requirements": self._extract_technical_requirements(issue_data),
            "suggested_approach": self._suggest_approach(issue_data),
            "estimated_effort": self._estimate_effort(issue_data),
            "related_keywords": self._extract_issue_keywords(issue_data.body or "")
        }
        if issue_data.comments:
            analysis["comments_analysis"] = self._analyze_comments(issue_data.comments)
        return analysis

    def _classify_issue_type(self, issue_data: Any) -> Dict[str, Any]:
        """Classify issue type based on title, labels, and content"""
        title = issue_data.title.lower()
        body = (issue_data.body or "").lower()
        labels = [label.lower() for label in (issue_data.labels or [])]
        
        # Define classification patterns
        bug_patterns = ["bug", "error", "fail", "crash", "broken", "issue", "problem", "not work"]
        feature_patterns = ["feature", "enhance", "improve", "add", "implement", "new"]
        docs_patterns = ["doc", "readme", "documentation", "guide", "tutorial"]
        performance_patterns = ["performance", "slow", "speed", "optimize", "memory"]
        security_patterns = ["security", "vulnerabil", "exploit", "auth", "permission"]
        
        scores = {
            "bug": 0, "feature": 0, "documentation": 0, 
            "performance": 0, "security": 0, "question": 0
        }
        
        # Check labels first (highest confidence)
        for label in labels:
            if any(pattern in label for pattern in bug_patterns):
                scores["bug"] += 0.8
            elif any(pattern in label for pattern in feature_patterns):
                scores["feature"] += 0.8
            elif any(pattern in label for pattern in docs_patterns):
                scores["documentation"] += 0.8
            elif any(pattern in label for pattern in performance_patterns):
                scores["performance"] += 0.8
            elif any(pattern in label for pattern in security_patterns):
                scores["security"] += 0.8
        
        # Check title and body
        text_content = f"{title} {body}"
        if any(pattern in text_content for pattern in bug_patterns):
            scores["bug"] += 0.4
        if any(pattern in text_content for pattern in feature_patterns):
            scores["feature"] += 0.4
        if any(pattern in text_content for pattern in docs_patterns):
            scores["documentation"] += 0.4
        if any(pattern in text_content for pattern in performance_patterns):
            scores["performance"] += 0.4
        if any(pattern in text_content for pattern in security_patterns):
            scores["security"] += 0.4
        if "?" in title or "how to" in text_content:
            scores["question"] += 0.3
        
        # Find primary type
        primary_type = max(scores.items(), key=lambda x: x[1])
        
        return {
            "primary_type": primary_type[0] if primary_type[1] > 0.1 else "general",
            "confidence_scores": scores
        }

    def _assess_issue_complexity(self, issue_data: Any) -> Dict[str, Any]:
        """Assess complexity of an issue"""
        title = issue_data.title.lower()
        body = (issue_data.body or "").lower()
        labels = issue_data.labels or []
        
        complexity_score = 0
        
        # Length and detail indicators
        if len(body) > 1000:
            complexity_score += 2
        elif len(body) > 500:
            complexity_score += 1
        
        # Technical complexity indicators
        technical_terms = ["api", "database", "architecture", "refactor", "migration", "algorithm", "protocol"]
        complexity_score += sum(1 for term in technical_terms if term in f"{title} {body}")
        
        # Multiple component indicators
        if "multiple" in f"{title} {body}" or "various" in f"{title} {body}":
            complexity_score += 2
        
        # Priority/severity labels
        high_priority_labels = ["critical", "urgent", "high priority", "blocker"]
        if any(label.lower() in high_priority_labels for label in labels):
            complexity_score += 1
        
        # Convert to level
        if complexity_score >= 5:
            level = "high"
        elif complexity_score >= 3:
            level = "medium"
        else:
            level = "low"
        
        return {
            "level": level,
            "score": complexity_score,
            "factors": {
                "content_length": len(body),
                "technical_terms_count": sum(1 for term in technical_terms if term in f"{title} {body}"),
                "has_priority_labels": any(label.lower() in high_priority_labels for label in labels)
            }
        }

    def _extract_technical_requirements(self, issue_data: Any) -> Dict[str, Any]:
        """Extract technical requirements from issue"""
        title = issue_data.title.lower()
        body = (issue_data.body or "").lower()
        text = f"{title} {body}"
        
        # Programming languages
        languages = ["python", "javascript", "typescript", "java", "go", "rust", "c++", "c#", "ruby", "php"]
        detected_languages = [lang for lang in languages if lang in text]
        
        # Frameworks and technologies
        frameworks = ["react", "vue", "angular", "django", "flask", "express", "spring", "rails"]
        detected_frameworks = [fw for fw in frameworks if fw in text]
        
        # Databases
        databases = ["postgresql", "mysql", "mongodb", "redis", "sqlite", "elasticsearch"]
        detected_databases = [db for db in databases if db in text]
        
        # Infrastructure
        infra_terms = ["docker", "kubernetes", "aws", "gcp", "azure", "nginx", "apache"]
        detected_infra = [term for term in infra_terms if term in text]
        
        return {
            "programming_languages": detected_languages,
            "frameworks": detected_frameworks,
            "databases": detected_databases,
            "infrastructure": detected_infra,
            "mentions_api": "api" in text,
            "mentions_database": any(db in text for db in ["database", "db", "sql"]),
            "mentions_ui": any(ui in text for ui in ["ui", "interface", "frontend", "css", "html"])
        }

    def _suggest_approach(self, issue_data: Any) -> List[str]:
        """Suggest approach for resolving the issue"""
        issue_type = self._classify_issue_type(issue_data)["primary_type"]
        complexity = self._assess_issue_complexity(issue_data)["level"]
        
        base_steps = []
        
        if issue_type == "bug":
            base_steps = [
                "1. Reproduce the issue using the provided steps",
                "2. Identify the root cause through debugging",
                "3. Implement a fix with proper error handling",
                "4. Add or update tests to prevent regression",
                "5. Test the fix thoroughly in different scenarios"
            ]
        elif issue_type == "feature":
            base_steps = [
                "1. Analyze the feature requirements and scope",
                "2. Design the implementation approach",
                "3. Break down into smaller, manageable tasks",
                "4. Implement core functionality with tests",
                "5. Add documentation and usage examples"
            ]
        elif issue_type == "documentation":
            base_steps = [
                "1. Review existing documentation for gaps",
                "2. Gather requirements for the new documentation",
                "3. Write clear, comprehensive documentation",
                "4. Include code examples where applicable",
                "5. Review and test documentation accuracy"
            ]
        else:
            base_steps = [
                "1. Analyze the issue requirements and context",
                "2. Research potential solutions and approaches",
                "3. Plan implementation with consideration for edge cases",
                "4. Implement solution with proper testing",
                "5. Document changes and update relevant files"
            ]
        
        # Add complexity-specific suggestions
        if complexity == "high":
            base_steps.insert(1, "1.5. Consider breaking into smaller sub-issues")
            base_steps.append("6. Coordinate with team for review of complex changes")
        
        return base_steps

    def _estimate_effort(self, issue_data: Any) -> Dict[str, Any]:
        """Estimate effort required for the issue"""
        complexity = self._assess_issue_complexity(issue_data)
        issue_type = self._classify_issue_type(issue_data)["primary_type"]
        
        # Base estimates in hours
        base_estimates = {
            "bug": {"low": 2, "medium": 8, "high": 24},
            "feature": {"low": 8, "medium": 24, "high": 80},
            "documentation": {"low": 1, "medium": 4, "high": 16},
            "performance": {"low": 4, "medium": 16, "high": 40},
            "security": {"low": 8, "medium": 24, "high": 48},
            "question": {"low": 0.5, "medium": 2, "high": 4}
        }
        
        complexity_level = complexity["level"]
        estimated_hours = base_estimates.get(issue_type, base_estimates["bug"])[complexity_level]
        
        # Adjust based on technical requirements
        tech_reqs = self._extract_technical_requirements(issue_data)
        if len(tech_reqs["programming_languages"]) > 1:
            estimated_hours *= 1.3
        if tech_reqs["mentions_database"] and tech_reqs["mentions_api"]:
            estimated_hours *= 1.2
        
        return {
            "estimated_hours": round(estimated_hours, 1),
            "confidence": "low" if complexity_level == "high" else "medium",
            "factors": {
                "issue_type": issue_type,
                "complexity": complexity_level,
                "multi_language": len(tech_reqs["programming_languages"]) > 1,
                "cross_component": tech_reqs["mentions_database"] and tech_reqs["mentions_api"]
            }
        }

    def _analyze_comments(self, comments_data: List[Any]) -> Dict[str, Any]:
        """Analyze comments for additional context"""
        if not comments_data:
            return {"total_comments": 0, "key_points": []}
        
        # Extract key information from comments
        key_points = []
        solution_attempts = 0
        question_count = 0
        
        for comment in comments_data:
            content = comment.body.lower() if hasattr(comment, 'body') else str(comment).lower()
            
            if any(word in content for word in ["tried", "attempt", "fix", "solution"]):
                solution_attempts += 1
            
            if "?" in content:
                question_count += 1
            
            # Extract potential key points (first sentence of longer comments)
            if len(content) > 100:
                first_sentence = content.split('.')[0][:100]
                if len(first_sentence) > 20:
                    key_points.append(first_sentence.strip())
        
        return {
            "total_comments": len(comments_data),
            "solution_attempts": solution_attempts,
            "questions_asked": question_count,
            "key_points": key_points[:5],  # Limit to top 5
            "high_engagement": len(comments_data) > 10
        }
        
    def _extract_issue_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract keywords from issue text"""
        if not text:
            return {"primary": [], "contextual": []}
        
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        stop_words = {
            "the", "is", "at", "which", "on", "and", "a", "to", "are", "as", "an", "will", "be", "or", 
            "of", "with", "for", "this", "that", "from", "they", "we", "been", "have", "has", "had",
            "but", "not", "would", "there", "their", "what", "all", "were", "when", "where", "how",
            "if", "more", "may", "these", "some", "could", "other", "after", "first", "well", "many"
        }
        
        # Get meaningful words
        meaningful_words = [word for word in words if len(word) > 2 and word not in stop_words]
        
        # Count frequency
        word_freq = {}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Categorize
        primary = [word for word, freq in sorted_words[:5] if freq > 1]
        contextual = [word for word, freq in sorted_words[5:15] if freq >= 1]
        
        return {"primary": primary, "contextual": contextual}

    def find_issue_related_files(
        self,
        issue_description: Annotated[str, "Description of the issue or feature to find related files for"],
        search_depth: Annotated[str, "Search depth: 'surface' for obvious matches, 'deep' for comprehensive analysis"] = "surface"
    ) -> str:
        try:
            search_terms = self._extract_issue_keywords(issue_description)
            relevant_files = []
            
            for term in search_terms.get('primary', []):
                # Use self.search_ops for codebase search
                search_results_str = self.search_ops.search_codebase(term, ['.py', '.js', '.ts', '.java', '.go', '.rs', '.rb'])
                search_results = json.loads(search_results_str)
                for res in search_results.get('results', []):
                    relevant_files.append({"file": res['file'], "relevance_score": len(res['matches']) * 2, "match_reason": f"Contains '{term}'"})

            if search_depth == "deep":
                for term in search_terms.get('contextual', []):
                    semantic_results_str = self.search_ops.semantic_content_search(term)
                    semantic_results = json.loads(semantic_results_str)
                    for res in semantic_results.get('results', []):
                         if not any(rf['file'] == res['file'] for rf in relevant_files): # Avoid duplicates
                            relevant_files.append({"file": res['file'], "relevance_score": res['score'], "match_reason": f"Semantic match for '{term}'"})
            
            # Add configuration and test files
            config_files = self._find_configuration_files(search_terms)
            relevant_files.extend(config_files)
            
            if relevant_files:
                # Find related test files
                source_files = [rf["file"] for rf in relevant_files[:10]]  # Top 10 source files
                test_files = self._find_related_test_files(source_files)
                relevant_files.extend(test_files)
            
            # Remove duplicates and sort
            seen_files = set()
            unique_files = []
            for file_info in relevant_files:
                if file_info["file"] not in seen_files:
                    seen_files.add(file_info["file"])
                    unique_files.append(file_info)
            
            sorted_files = sorted(unique_files, key=lambda x: x['relevance_score'], reverse=True)
            
            # Categorize and generate recommendations
            categorized = self._categorize_files(sorted_files)
            recommendations = self._generate_file_recommendations(sorted_files, search_terms)
            
            return json.dumps({
                "issue_description": issue_description,
                "top_relevant_files": sorted_files[:15],
                "categorized_files": categorized,
                "recommendations": recommendations,
                "search_depth": search_depth
            }, indent=2)
        except Exception as e:
            logger.error(f"Error finding issue-related files: {e}")
            return json.dumps({"error": str(e)}, indent=2)

    def _find_configuration_files(self, search_terms: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Find configuration files that might be relevant"""
        config_patterns = [
            "*.json", "*.yaml", "*.yml", "*.toml", "*.ini", "*.cfg", "*.conf",
            "Dockerfile", "docker-compose*", "requirements.txt", "package.json",
            "Makefile", "*.mk", ".env*", "config/*"
        ]
        
        config_files = []
        for pattern in config_patterns:
            try:
                search_results_str = self.search_ops.search_codebase("", None)  # Get all files
                search_results = json.loads(search_results_str)
                
                for result in search_results.get('results', []):
                    file_path = result['file']
                    if any(term in file_path.lower() for term in search_terms.get('primary', [])):
                        config_files.append({
                            "file": file_path,
                            "relevance_score": 1,
                            "match_reason": "Configuration file with keyword match"
                        })
            except:
                continue
                
        return config_files[:5]  # Limit to 5 config files

    def _find_related_test_files(self, source_files: List[str]) -> List[Dict[str, Any]]:
        """Find test files related to source files"""
        test_files = []
        
        for source_file in source_files:
            # Generate potential test file names
            base_name = source_file.replace('.py', '').replace('.js', '').replace('.ts', '')
            test_patterns = [
                f"test_{base_name}.*",
                f"{base_name}_test.*",
                f"tests/*{base_name}*",
                f"**/test*{base_name}*",
                f"**/*{base_name}*test*"
            ]
            
            for pattern in test_patterns:
                try:
                    # Use semantic search to find test files
                    search_results_str = self.search_ops.semantic_content_search(f"test {base_name}")
                    search_results = json.loads(search_results_str)
                    
                    for result in search_results.get('results', []):
                        if 'test' in result['file'].lower():
                            test_files.append({
                                "file": result['file'],
                                "relevance_score": result.get('score', 1),
                                "match_reason": f"Test file for {source_file}"
                            })
                except:
                    continue
                    
        return test_files[:10]  # Limit to 10 test files

    def _categorize_files(self, files: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize files by type"""
        categories = {
            "source_code": [],
            "tests": [],
            "configuration": [],
            "documentation": [],
            "build_scripts": []
        }
        
        for file_info in files:
            file_path = file_info["file"].lower()
            
            if "test" in file_path or file_path.endswith(('.test.py', '.test.js', '_test.py', '_test.js')):
                categories["tests"].append(file_info["file"])
            elif file_path.endswith(('.md', '.rst', '.txt', '.doc')):
                categories["documentation"].append(file_info["file"])
            elif file_path.endswith(('.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.env')):
                categories["configuration"].append(file_info["file"])
            elif file_path.endswith(('makefile', 'dockerfile', '.mk', '.sh', '.bat')):
                categories["build_scripts"].append(file_info["file"])
            else:
                categories["source_code"].append(file_info["file"])
                
        return categories

    def _generate_file_recommendations(self, files: List[Dict[str, Any]], search_terms: Dict[str, List[str]]) -> List[str]:
        """Generate recommendations for files to examine"""
        if not files:
            return ["No relevant files found. Consider expanding search terms."]
        
        recommendations = []
        
        # Primary files to examine
        top_files = files[:3]
        if top_files:
            recommendations.append(f"Start by examining these high-relevance files: {', '.join([f['file'] for f in top_files])}")
        
        # Test coverage
        test_files = [f for f in files if 'test' in f['file'].lower()]
        if test_files:
            recommendations.append(f"Review existing tests: {', '.join([f['file'] for f in test_files[:2]])}")
        else:
            recommendations.append("Consider adding tests for any new functionality")
        
        # Configuration files
        config_files = [f for f in files if any(ext in f['file'].lower() for ext in ['.json', '.yaml', '.yml', '.cfg'])]
        if config_files:
            recommendations.append(f"Check configuration files: {', '.join([f['file'] for f in config_files[:2]])}")
        
        # Documentation
        doc_files = [f for f in files if f['file'].lower().endswith(('.md', '.rst', '.txt'))]
        if doc_files:
            recommendations.append(f"Update documentation: {', '.join([f['file'] for f in doc_files[:2]])}")
        
        return recommendations

    def related_issues(
        self,
        query: Annotated[str, "Issue title, bug description, or error message to find similar past issues for"],
        k: Annotated[int, "Number of similar issues to return (default: 5)"] = 5,
        state: Annotated[str, "Issue state filter: 'open', 'closed', or 'all' (default: 'all')"] = "all",
        similarity: Annotated[float, "Minimum similarity threshold from 0.0 to 1.0 (default: 0.75)"] = 0.75
    ) -> str:
        if not self.issue_rag_system or not self.issue_rag_system.is_initialized():
            return json.dumps({"error": "Issue RAG system not available/initialized."}, indent=2)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.issue_rag_system.get_issue_context(query, max_issues=k))
                    issue_context_result: 'IssueContext' = future.result(timeout=30)
            else:
                issue_context_result: 'IssueContext' = asyncio.run(self.issue_rag_system.get_issue_context(query, max_issues=k))

            owner, repo = self.issue_rag_system.indexer.repo_owner, self.issue_rag_system.indexer.repo_name
            formatted_issues = []
            for res in issue_context_result.related_issues:
                issue = res.issue
                
                # Apply state filter
                if state != "all" and issue.state != state:
                    continue
                    
                # Apply similarity filter
                if res.similarity < similarity:
                    continue
                
                formatted_issues.append({
                    "number": issue.id, "title": issue.title, "state": issue.state,
                    "url": f"https://github.com/{owner}/{repo}/issues/{issue.id}",
                    "similarity": round(res.similarity, 3), "labels": issue.labels,
                    "created_at": str(issue.created_at), "body_preview": (issue.body or "")[:200] + "..."
                })
            
            return json.dumps({
                "query": query, 
                "related_issues": formatted_issues,
                "filters": {"state": state, "min_similarity": similarity},
                "total_found": len(formatted_issues)
            }, indent=2)
        except Exception as e:
            logger.error(f"Error in related_issues: {e}")
            return json.dumps({"error": str(e)}, indent=2)

    def get_issue_closing_info(self, issue_number: Annotated[int, "Issue number"]) -> str:
        try:
            result = self.issue_closing_tools.get_issue_closing_info(issue_number)
            return json.dumps(result, indent=2) 
        except Exception as e: 
            return json.dumps({"error": str(e)})

    def get_open_issues_related_to_commit(self, commit_sha: Annotated[str, "Commit SHA"]) -> str:
        try:
            result = self.issue_closing_tools.get_open_issues_related_to_commit(commit_sha)
            return json.dumps(result, indent=2)
        except Exception as e: 
            return json.dumps({"error": str(e)})

    def find_issues_related_to_file(self, file_path: Annotated[str, "File path"]) -> str:
        try:
            # Use issue RAG system if available
            if self.issue_rag_system and self.issue_rag_system.is_initialized():
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self.issue_rag_system.get_issue_context(f"file {file_path}", max_issues=10)
                        )
                        issue_context_result = future.result(timeout=30)
                else:
                    issue_context_result = asyncio.run(
                        self.issue_rag_system.get_issue_context(f"file {file_path}", max_issues=10)
                    )
                
                owner, repo = self.issue_rag_system.indexer.repo_owner, self.issue_rag_system.indexer.repo_name
                related_issues = []
                
                for res in issue_context_result.related_issues:
                    issue = res.issue
                    if file_path in (issue.body or ""):
                        related_issues.append({
                            "number": issue.id,
                            "title": issue.title,
                            "state": issue.state,
                            "url": f"https://github.com/{owner}/{repo}/issues/{issue.id}",
                            "similarity": round(res.similarity, 3)
                        })
                
                return json.dumps({"file_path": file_path, "related_issues": related_issues}, indent=2)
            
            return json.dumps({"error": "Issue RAG system not available"})
        except Exception as e: 
            return json.dumps({"error": str(e)})

    def get_issue_resolution_summary(self, issue_number: Annotated[int, "Issue number"]) -> str:
        try:
            if self.issue_rag_system and self.issue_rag_system.is_initialized():
                # Get issue details and related PRs
                issue_docs = self.issue_rag_system.indexer.issue_docs
                if issue_number in issue_docs:
                    issue = issue_docs[issue_number]
                    
                    # Find related PRs
                    related_prs = []
                    if hasattr(self.issue_rag_system.indexer, 'diff_docs'):
                        for pr_num, pr_doc in self.issue_rag_system.indexer.diff_docs.items():
                            if hasattr(pr_doc, 'issue_id') and pr_doc.issue_id == issue_number:
                                related_prs.append({
                                    "pr_number": pr_num,
                                    "files_changed": pr_doc.files_changed,
                                    "diff_summary": pr_doc.diff_summary[:200] + "..." if len(pr_doc.diff_summary) > 200 else pr_doc.diff_summary
                                })
                    
                    return json.dumps({
                        "issue_number": issue_number,
                        "title": issue.title,
                        "state": issue.state,
                        "resolution_prs": related_prs,
                        "closed_at": str(issue.closed_at) if hasattr(issue, 'closed_at') else None
                    }, indent=2)
            
            return json.dumps({"error": "Issue not found or RAG system not available"})
        except Exception as e: 
            return json.dumps({"error": str(e)})

    async def check_issue_status_and_linked_pr(self, issue_identifier: Annotated[str, "Issue identifier"]) -> str:
        try:
            from ..github_client import GitHubIssueClient
            
            github_client = GitHubIssueClient()
            
            # Handle different issue identifier formats
            if issue_identifier.startswith('#') or issue_identifier.isdigit():
                issue_number = issue_identifier.lstrip('#')
                repo_url = self._get_repo_url_from_path(self.repo_path)
                if not repo_url:
                    return json.dumps({"error": "Cannot determine repository URL"}, indent=2)
                issue_url = f"{repo_url}/issues/{issue_number}"
            else:
                issue_url = issue_identifier
            
            # Get issue details
            issue_response = await github_client.get_issue(issue_url)
            if issue_response.status != "success":
                return json.dumps({"error": f"Failed to fetch issue: {issue_response.error}"}, indent=2)
            
            issue_data = issue_response.data
            
            # Find linked PRs
            linked_prs = []
            if self.issue_rag_system and self.issue_rag_system.is_initialized():
                issue_num = int(issue_number) if issue_identifier.startswith('#') or issue_identifier.isdigit() else None
                if issue_num and hasattr(self.issue_rag_system.indexer, 'diff_docs'):
                    for pr_num, pr_doc in self.issue_rag_system.indexer.diff_docs.items():
                        if hasattr(pr_doc, 'issue_id') and pr_doc.issue_id == issue_num:
                            linked_prs.append({
                                "pr_number": pr_num,
                                "status": getattr(pr_doc, 'status', 'unknown'),
                                "files_changed": pr_doc.files_changed
                            })
            
            return json.dumps({
                "issue_number": issue_data.number,
                "title": issue_data.title,
                "state": issue_data.state,
                "linked_prs": linked_prs,
                "url": issue_url
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error checking issue status: {e}")
            return json.dumps({"error": str(e)})

    async def regression_detector(self, issue_query: Annotated[str, "Issue query"]) -> str:
        """Detect if a new issue is a regression by analyzing similar closed issues"""
        try:
            if not self.issue_rag_system or not self.issue_rag_system.is_initialized():
                return json.dumps({"error": "Issue RAG system not available for regression detection"}, indent=2)
            
            # Search for similar closed issues
            issue_context = await self.issue_rag_system.get_issue_context(issue_query, max_issues=10)
            
            closed_similar_issues = []
            potential_regressions = []
            
            for res in issue_context.related_issues:
                issue = res.issue
                if issue.state == "closed" and res.similarity > 0.7:
                    closed_similar_issues.append({
                        "issue_id": issue.id,
                        "title": issue.title,
                        "similarity": res.similarity,
                        "closed_at": str(issue.closed_at) if hasattr(issue, 'closed_at') else None,
                        "labels": issue.labels
                    })
                    
                    # Check if this could be a regression
                    if res.similarity > 0.85:
                        potential_regressions.append({
                            "issue_id": issue.id,
                            "title": issue.title,
                            "similarity": res.similarity,
                            "regression_likelihood": "high" if res.similarity > 0.9 else "medium"
                        })
            
            # Analyze patterns
            regression_indicators = []
            if potential_regressions:
                regression_indicators.append("High similarity to previously closed issues detected")
            
            # Check for common regression patterns in the query
            regression_keywords = ["again", "reoccur", "still", "broken again", "came back", "regression"]
            if any(keyword in issue_query.lower() for keyword in regression_keywords):
                regression_indicators.append("Query contains regression-indicating keywords")
            
            return json.dumps({
                "query": issue_query,
                "is_potential_regression": len(potential_regressions) > 0,
                "regression_likelihood": "high" if len(potential_regressions) > 0 and any(r["regression_likelihood"] == "high" for r in potential_regressions) else "low",
                "similar_closed_issues": closed_similar_issues[:5],
                "potential_regressions": potential_regressions,
                "regression_indicators": regression_indicators
            }, indent=2)
            
        except Exception as e:
            logger.error(f"Error in regression detector: {e}")
            return json.dumps({"error": str(e)})
