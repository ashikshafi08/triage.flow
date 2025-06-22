"""
Professional Response Formatter

Transforms raw agent output into structured, professional responses similar to 
Cursor IDE, Copilot, and other modern AI coding assistants.

This module provides structured response formatting with:
- Executive summaries
- Categorized results with visual indicators
- Collapsible detail sections
- Source links and timestamps
- Next action suggestions
- Professional metadata
"""

import json
import re
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ResponseType(Enum):
    """Types of responses the formatter can generate"""
    ANALYSIS = "analysis"
    SEARCH_RESULTS = "search_results"
    CODE_EXPLANATION = "code_explanation"
    ISSUE_INVESTIGATION = "issue_investigation"
    FILE_EXPLORATION = "file_exploration"
    GIT_HISTORY = "git_history"
    ERROR_DIAGNOSIS = "error_diagnosis"
    FEATURE_SUMMARY = "feature_summary"

class Priority(Enum):
    """Priority levels for response items"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class SourceReference:
    """Reference to source material"""
    type: str  # "file", "commit", "pr", "issue", "line"
    path: Optional[str] = None
    url: Optional[str] = None
    line_number: Optional[int] = None
    line_range: Optional[Tuple[int, int]] = None
    commit_sha: Optional[str] = None
    pr_number: Optional[int] = None
    issue_number: Optional[int] = None
    title: Optional[str] = None
    timestamp: Optional[str] = None

@dataclass
class ActionItem:
    """Suggested next action"""
    type: str  # "investigate", "fix", "optimize", "implement", "review"
    title: str
    description: str
    priority: Priority
    sources: List[SourceReference]
    estimated_effort: Optional[str] = None  # "5 minutes", "1 hour", "2 days"

@dataclass
class DetailSection:
    """Collapsible detail section"""
    title: str
    content: str
    collapsible: bool = True
    default_collapsed: bool = True
    type: str = "details"  # "code", "logs", "analysis", "details"
    sources: List[SourceReference] = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []

@dataclass
class ResponseCategory:
    """Categorized response section"""
    title: str
    icon: str  # Emoji or icon identifier
    count: int
    priority: Priority
    items: List[Dict[str, Any]]
    summary: str
    details: List[DetailSection]
    sources: List[SourceReference]
    
    def __post_init__(self):
        if not self.details:
            self.details = []
        if not self.sources:
            self.sources = []

@dataclass
class StructuredResponse:
    """Complete structured response"""
    response_type: ResponseType
    title: str
    executive_summary: str
    key_findings: List[str]
    categories: List[ResponseCategory]
    next_actions: List[ActionItem]
    metadata: Dict[str, Any]
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "response_type": self.response_type.value,
            "title": self.title,
            "executive_summary": self.executive_summary,
            "key_findings": self.key_findings,
            "categories": [
                {
                    "title": cat.title,
                    "icon": cat.icon,
                    "count": cat.count,
                    "priority": cat.priority.value,
                    "items": cat.items,
                    "summary": cat.summary,
                    "details": [asdict(detail) for detail in cat.details],
                    "sources": [asdict(source) for source in cat.sources]
                }
                for cat in self.categories
            ],
            "next_actions": [
                {
                    "type": action.type,
                    "title": action.title,
                    "description": action.description,
                    "priority": action.priority.value,
                    "sources": [asdict(source) for source in action.sources],
                    "estimated_effort": action.estimated_effort
                }
                for action in self.next_actions
            ],
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "confidence_score": self.confidence_score,
            "timestamp": datetime.now().isoformat()
        }

class ResponseFormatter:
    """Professional response formatter"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.response_patterns = self._load_response_patterns()
        
    def _load_response_patterns(self) -> Dict[str, Dict]:
        """Load response patterns for different types of queries"""
        return {
            "file_search": {
                "icons": {"files": "📁", "matches": "🔍", "code": "💻"},
                "categories": ["Files Found", "Code Matches", "Import Relationships"]
            },
            "issue_analysis": {
                "icons": {"bugs": "🐛", "features": "🚀", "enhancements": "⚡"},
                "categories": ["Issues", "Pull Requests", "Related Code"]
            },
            "git_history": {
                "icons": {"commits": "📝", "authors": "👨‍💻", "files": "📄"},
                "categories": ["Recent Commits", "Contributors", "File Changes"]
            },
            "code_explanation": {
                "icons": {"functions": "⚙️", "classes": "🏗️", "imports": "🔗"},
                "categories": ["Core Functions", "Class Structure", "Dependencies"]
            }
        }
    
    def format_search_results(self, raw_results: str, query: str) -> StructuredResponse:
        """Format search results professionally"""
        try:
            # Parse raw JSON results
            if isinstance(raw_results, str):
                try:
                    results_data = json.loads(raw_results)
                except json.JSONDecodeError:
                    # Handle non-JSON results
                    results_data = {"raw_content": raw_results}
            else:
                results_data = raw_results
            
            # Extract key metrics
            files_found = results_data.get("files_with_matches", 0)
            total_processed = results_data.get("total_files_processed", 0)
            results = results_data.get("results", [])
            
            # Create executive summary
            summary = f"Found {files_found} files with matches for '{query}' across {total_processed} files searched."
            
            # Key findings
            key_findings = []
            if files_found > 0:
                key_findings.append(f"{files_found} files contain the search term")
                if files_found > 10:
                    key_findings.append("High number of matches suggests widespread usage")
                
                # Analyze file types
                file_types = {}
                for result in results:
                    file_path = result.get("file", "")
                    ext = Path(file_path).suffix
                    file_types[ext] = file_types.get(ext, 0) + 1
                
                if file_types:
                    dominant_type = max(file_types.items(), key=lambda x: x[1])
                    key_findings.append(f"Most matches in {dominant_type[0]} files ({dominant_type[1]} files)")
            
            # Create categories
            categories = []
            
            # Files category
            if results:
                file_items = []
                sources = []
                
                for result in results[:10]:  # Limit to top 10
                    file_path = result.get("file", "")
                    matches = result.get("matches", [])
                    
                    # Create source reference
                    source = SourceReference(
                        type="file",
                        path=file_path,
                        title=f"{len(matches)} matches in {file_path}"
                    )
                    sources.append(source)
                    
                    # Create file item
                    file_items.append({
                        "path": file_path,
                        "matches": len(matches),
                        "preview": matches[0].get("line", "")[:100] if matches else "",
                        "line_numbers": [m.get("line_number") for m in matches[:3]]
                    })
                
                # Create detail sections
                details = []
                for result in results:
                    file_path = result.get("file", "")
                    matches = result.get("matches", [])
                    
                    if matches:
                        match_content = "\n".join([
                            f"Line {m.get('line_number', 'N/A')}: {m.get('line', '')}"
                            for m in matches[:5]
                        ])
                        
                        details.append(DetailSection(
                            title=f"{file_path} ({len(matches)} matches)",
                            content=match_content,
                            type="code",
                            sources=[SourceReference(type="file", path=file_path)]
                        ))
                
                categories.append(ResponseCategory(
                    title="Files Found",
                    icon="📁",
                    count=len(results),
                    priority=Priority.HIGH,
                    items=file_items,
                    summary=f"Found matches in {len(results)} files",
                    details=details,
                    sources=sources
                ))
            
            # Next actions
            next_actions = []
            if files_found > 0:
                next_actions.append(ActionItem(
                    type="investigate",
                    title="Examine top matches",
                    description=f"Review the {min(3, files_found)} files with the most matches",
                    priority=Priority.HIGH,
                    sources=[SourceReference(type="file", path=r.get("file")) for r in results[:3]],
                    estimated_effort="10 minutes"
                ))
                
                if files_found > 10:
                    next_actions.append(ActionItem(
                        type="optimize",
                        title="Refine search scope",
                        description="Consider narrowing search to specific directories or file types",
                        priority=Priority.MEDIUM,
                        sources=[],
                        estimated_effort="2 minutes"
                    ))
            
            return StructuredResponse(
                response_type=ResponseType.SEARCH_RESULTS,
                title=f"Search Results: '{query}'",
                executive_summary=summary,
                key_findings=key_findings,
                categories=categories,
                next_actions=next_actions,
                metadata={
                    "query": query,
                    "total_files_processed": total_processed,
                    "files_with_matches": files_found,
                    "search_directory": results_data.get("search_directory"),
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error formatting search results: {e}")
            # Return basic structured response on error
            return StructuredResponse(
                response_type=ResponseType.SEARCH_RESULTS,
                title=f"Search Results: '{query}'",
                executive_summary="Search completed but formatting failed",
                key_findings=["Raw results available in details"],
                categories=[],
                next_actions=[],
                metadata={"error": str(e), "raw_results": raw_results}
            )
    
    def format_file_analysis(self, file_content: str, file_path: str, analysis_type: str = "general") -> StructuredResponse:
        """Format file analysis results"""
        try:
            # Basic analysis
            lines = file_content.split('\n')
            line_count = len(lines)
            
            # Detect file type and relevant patterns
            file_ext = Path(file_path).suffix
            
            # Key findings based on content
            key_findings = [f"File contains {line_count} lines"]
            
            # Language-specific analysis
            if file_ext in ['.py', '.js', '.ts', '.jsx', '.tsx']:
                # Count functions/methods
                function_pattern = r'def\s+\w+|function\s+\w+|const\s+\w+\s*=.*=>|\w+\s*:\s*\([^)]*\)\s*=>'
                functions = re.findall(function_pattern, file_content, re.MULTILINE)
                if functions:
                    key_findings.append(f"Contains {len(functions)} functions/methods")
                
                # Count classes
                class_pattern = r'class\s+\w+'
                classes = re.findall(class_pattern, file_content, re.MULTILINE)
                if classes:
                    key_findings.append(f"Defines {len(classes)} classes")
                
                # Count imports
                import_pattern = r'import\s+|from\s+.*\s+import'
                imports = re.findall(import_pattern, file_content, re.MULTILINE)
                if imports:
                    key_findings.append(f"Has {len(imports)} import statements")
            
            # Create categories
            categories = []
            
            # Structure category
            structure_items = []
            if file_ext in ['.py', '.js', '.ts']:
                # Extract main components
                for i, line in enumerate(lines[:50], 1):  # First 50 lines
                    line_stripped = line.strip()
                    if line_stripped.startswith(('class ', 'def ', 'function ', 'const ', 'let ', 'var ')):
                        structure_items.append({
                            "line": i,
                            "type": "definition",
                            "content": line_stripped[:80],
                            "importance": "high" if any(kw in line_stripped for kw in ['class', 'def', 'function']) else "medium"
                        })
            
            if structure_items:
                categories.append(ResponseCategory(
                    title="Code Structure",
                    icon="🏗️",
                    count=len(structure_items),
                    priority=Priority.HIGH,
                    items=structure_items,
                    summary=f"File defines {len(structure_items)} main components",
                    details=[DetailSection(
                        title="Full Structure",
                        content="\n".join([f"Line {item['line']}: {item['content']}" for item in structure_items]),
                        type="code"
                    )],
                    sources=[SourceReference(type="file", path=file_path)]
                ))
            
            # Next actions
            next_actions = []
            if line_count > 500:
                next_actions.append(ActionItem(
                    type="review",
                    title="Review file complexity",
                    description="Large file may benefit from refactoring",
                    priority=Priority.MEDIUM,
                    sources=[SourceReference(type="file", path=file_path)],
                    estimated_effort="15 minutes"
                ))
            
            if len(structure_items) > 10:
                next_actions.append(ActionItem(
                    type="optimize",
                    title="Consider splitting file",
                    description="File has many definitions and could be split into smaller modules",
                    priority=Priority.LOW,
                    sources=[SourceReference(type="file", path=file_path)],
                    estimated_effort="30 minutes"
                ))
            
            return StructuredResponse(
                response_type=ResponseType.FILE_EXPLORATION,
                title=f"File Analysis: {Path(file_path).name}",
                executive_summary=f"Analyzed {Path(file_path).name} ({line_count} lines) and identified key structural components",
                key_findings=key_findings,
                categories=categories,
                next_actions=next_actions,
                metadata={
                    "file_path": file_path,
                    "file_size": len(file_content),
                    "line_count": line_count,
                    "file_type": file_ext,
                    "analysis_type": analysis_type,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error formatting file analysis: {e}")
            return StructuredResponse(
                response_type=ResponseType.FILE_EXPLORATION,
                title=f"File Analysis: {Path(file_path).name}",
                executive_summary="File analysis completed with errors",
                key_findings=["Analysis partially failed"],
                categories=[],
                next_actions=[],
                metadata={"error": str(e)}
            )
    
    def format_generic_response(self, content: str, response_type: ResponseType = ResponseType.ANALYSIS) -> StructuredResponse:
        """Format generic response content"""
        try:
            # Try to extract structure from content
            lines = content.split('\n')
            
            # Look for list items or structured content
            key_findings = []
            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith(('-', '*', '•')) or re.match(r'^\d+\.', line_stripped):
                    key_findings.append(line_stripped.lstrip('-*•').strip())
            
            # If no structured content found, create generic findings
            if not key_findings:
                key_findings = ["Response generated successfully"]
                if len(content) > 500:
                    key_findings.append("Detailed analysis provided")
            
            # Create a single category for the content
            categories = [ResponseCategory(
                title="Analysis Results",
                icon="📊",
                count=1,
                priority=Priority.HIGH,
                items=[{
                    "type": "analysis",
                    "content_preview": content[:200] + "..." if len(content) > 200 else content,
                    "full_length": len(content)
                }],
                summary="Comprehensive analysis completed",
                details=[DetailSection(
                    title="Full Analysis",
                    content=content,
                    type="analysis",
                    default_collapsed=len(content) > 1000
                )],
                sources=[]
            )]
            
            return StructuredResponse(
                response_type=response_type,
                title="Analysis Complete",
                executive_summary=f"Analysis completed with {len(key_findings)} key findings",
                key_findings=key_findings[:5],  # Limit to 5 key findings
                categories=categories,
                next_actions=[],
                metadata={
                    "content_length": len(content),
                    "response_type": response_type.value,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error formatting generic response: {e}")
            return StructuredResponse(
                response_type=response_type,
                title="Response Generated",
                executive_summary="Content processed with formatting errors",
                key_findings=["Raw content available"],
                categories=[],
                next_actions=[],
                metadata={"error": str(e), "raw_content": content}
            )
    
    def detect_response_type(self, content: str, tool_name: str = None) -> ResponseType:
        """Detect the most appropriate response type based on content and context"""
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content)
        content_lower = content.lower()
        
        # Check tool name first
        if tool_name:
            if "search" in tool_name:
                return ResponseType.SEARCH_RESULTS
            elif "git" in tool_name or "commit" in tool_name:
                return ResponseType.GIT_HISTORY
            elif "file" in tool_name or "read" in tool_name:
                return ResponseType.FILE_EXPLORATION
            elif "issue" in tool_name:
                return ResponseType.ISSUE_INVESTIGATION
        
        # Check content patterns
        if any(keyword in content_lower for keyword in ["search", "found", "matches", "files"]):
            return ResponseType.SEARCH_RESULTS
        elif any(keyword in content_lower for keyword in ["commit", "git", "author", "branch"]):
            return ResponseType.GIT_HISTORY
        elif any(keyword in content_lower for keyword in ["issue", "bug", "problem", "error"]):
            return ResponseType.ISSUE_INVESTIGATION
        elif any(keyword in content_lower for keyword in ["function", "class", "method", "code"]):
            return ResponseType.CODE_EXPLANATION
        elif any(keyword in content_lower for keyword in ["file", "directory", "path"]):
            return ResponseType.FILE_EXPLORATION
        
        return ResponseType.ANALYSIS
    
    def format_response(self, raw_content: str, tool_name: str = None, query: str = None) -> StructuredResponse:
        """Main method to format any response content"""
        try:
            # Ensure raw_content is a string
            if not isinstance(raw_content, str):
                if isinstance(raw_content, dict):
                    raw_content = json.dumps(raw_content, indent=2)
                else:
                    raw_content = str(raw_content)
            
            # Detect response type
            response_type = self.detect_response_type(raw_content, tool_name)
            
            # Handle search results specifically
            if response_type == ResponseType.SEARCH_RESULTS and query:
                return self.format_search_results(raw_content, query)
            
            # Handle JSON content
            if raw_content.strip().startswith('{') and raw_content.strip().endswith('}'):
                try:
                    json_data = json.loads(raw_content)
                    if "results" in json_data and "query" in json_data:
                        return self.format_search_results(raw_content, json_data.get("query", ""))
                except json.JSONDecodeError:
                    pass
            
            # Default to generic formatting
            return self.format_generic_response(raw_content, response_type)
            
        except Exception as e:
            logger.error(f"Error formatting response: {e}")
            # Fallback to basic response
            return StructuredResponse(
                response_type=ResponseType.ANALYSIS,
                title="Response Generated",
                executive_summary="Response completed with formatting issues",
                key_findings=["Content available in raw format"],
                categories=[],
                next_actions=[],
                metadata={"error": str(e), "raw_content": raw_content}
            )