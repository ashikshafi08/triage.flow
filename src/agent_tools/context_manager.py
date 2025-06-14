"""
Enhanced Context Manager for Tool Execution

This module implements context sharing between tools following Cognition AI's principles:
1. Share context - Full execution traces, not just individual messages
2. Actions carry implicit decisions - Track and coordinate decisions across tools

Key Features:
- Execution context tracking
- Decision registry and conflict detection
- Tool coordination and result sharing
- Performance optimization through context reuse
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ToolExecution:
    """Record of a single tool execution"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    execution_time: float
    timestamp: datetime
    decisions_made: Dict[str, Any] = field(default_factory=dict)
    context_used: Dict[str, Any] = field(default_factory=dict)
    files_accessed: Set[str] = field(default_factory=set)
    related_executions: List[str] = field(default_factory=list)

@dataclass
class ExecutionContext:
    """Shared context across tool executions"""
    session_id: str
    query: str
    discovered_files: Dict[str, Any] = field(default_factory=dict)
    analyzed_components: Dict[str, Any] = field(default_factory=dict)
    decisions_made: Dict[str, Any] = field(default_factory=dict)
    execution_trace: List[ToolExecution] = field(default_factory=list)
    performance_cache: Dict[str, Any] = field(default_factory=dict)
    conflict_resolutions: List[Dict[str, Any]] = field(default_factory=list)

class ContextManager:
    """
    Manages execution context and enables enhanced tool coordination
    """
    
    def __init__(self, session_id: str, repo_path: Path):
        self.session_id = session_id
        self.repo_path = repo_path
        self.current_context: Optional[ExecutionContext] = None
        self.tool_relationships: Dict[str, List[str]] = {}
        self.decision_handlers: Dict[str, Callable] = {}
        
        # Initialize tool relationship mapping
        self._initialize_tool_relationships()
    
    def start_execution(self, query: str) -> ExecutionContext:
        """Start a new execution context for a query"""
        self.current_context = ExecutionContext(
            session_id=self.session_id,
            query=query
        )
        logger.info(f"Started execution context for query: {query[:100]}...")
        return self.current_context
    
    def get_context_for_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant context for a specific tool execution"""
        if not self.current_context:
            return {}
        
        context = {
            "session_id": self.session_id,
            "current_query": self.current_context.query,
            "execution_count": len(self.current_context.execution_trace),
        }
        
        # Add relevant previous executions
        related_executions = self._get_related_executions(tool_name, parameters)
        if related_executions:
            context["related_executions"] = related_executions
        
        # Add discovered files relevant to this tool
        relevant_files = self._get_relevant_files(tool_name, parameters)
        if relevant_files:
            context["relevant_files"] = relevant_files
        
        # Add decisions that might affect this tool
        relevant_decisions = self._get_relevant_decisions(tool_name, parameters)
        if relevant_decisions:
            context["previous_decisions"] = relevant_decisions
        
        # Add performance optimizations
        cache_key = self._generate_cache_key(tool_name, parameters)
        if cache_key in self.current_context.performance_cache:
            context["cached_result"] = self.current_context.performance_cache[cache_key]
        
        return context
    
    def record_execution(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any], 
        result: Any, 
        execution_time: float,
        context_used: Dict[str, Any] = None
    ) -> ToolExecution:
        """Record a tool execution with its context and decisions"""
        if not self.current_context:
            raise ValueError("No active execution context")
        
        # Extract decisions and files from result
        decisions_made = self._extract_decisions(tool_name, result)
        files_accessed = self._extract_files_accessed(tool_name, parameters, result)
        
        execution = ToolExecution(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            execution_time=execution_time,
            timestamp=datetime.now(),
            decisions_made=decisions_made,
            context_used=context_used or {},
            files_accessed=files_accessed,
            related_executions=self._find_related_executions(tool_name, parameters)
        )
        
        # Check for conflicts before recording
        conflicts = self._detect_conflicts(execution)
        if conflicts:
            resolution = self._resolve_conflicts(execution, conflicts)
            self.current_context.conflict_resolutions.append(resolution)
        
        # Update context with new information
        self._update_context_from_execution(execution)
        
        # Cache result for performance
        cache_key = self._generate_cache_key(tool_name, parameters)
        self.current_context.performance_cache[cache_key] = {
            "result": result,
            "timestamp": execution.timestamp,
            "decisions": decisions_made
        }
        
        self.current_context.execution_trace.append(execution)
        
        logger.debug(f"Recorded execution: {tool_name} -> {len(str(result))} chars result")
        return execution
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the current execution context"""
        if not self.current_context:
            return {"status": "no_active_context"}
        
        return {
            "session_id": self.current_context.session_id,
            "query": self.current_context.query,
            "total_executions": len(self.current_context.execution_trace),
            "tools_used": list(set(ex.tool_name for ex in self.current_context.execution_trace)),
            "files_discovered": len(self.current_context.discovered_files),
            "decisions_made": len(self.current_context.decisions_made),
            "conflicts_resolved": len(self.current_context.conflict_resolutions),
            "cache_hits": len(self.current_context.performance_cache)
        }
    
    def _initialize_tool_relationships(self):
        """Initialize relationships between tools for better coordination"""
        self.tool_relationships = {
            # File operations often lead to analysis
            "read_file": ["analyze_file_structure", "find_related_files", "git_blame_function"],
            "explore_directory": ["read_file", "find_related_files", "analyze_file_structure"],
            
            # Search operations complement each other
            "search_codebase": ["find_related_files", "semantic_content_search"],
            "find_related_files": ["read_file", "analyze_file_structure", "git_blame_function"],
            
            # Issue analysis leads to file discovery
            "analyze_github_issue": ["find_issue_related_files", "related_issues", "search_codebase"],
            "find_issue_related_files": ["read_file", "git_blame_function", "get_file_history"],
            
            # Git operations build on each other
            "git_blame_function": ["find_commits_touching_function", "get_function_evolution"],
            "get_file_history": ["get_commit_details", "analyze_commit_patterns"],
            
            # Code generation uses analysis results
            "generate_code_example": ["read_file", "analyze_file_structure", "find_related_files"],
            "write_complete_code": ["generate_code_example", "read_file", "analyze_file_structure"]
        }
    
    def _get_related_executions(self, tool_name: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get executions that are related to the current tool call"""
        if not self.current_context:
            return []
        
        related = []
        for execution in self.current_context.execution_trace:
            # Check if tools are related
            if execution.tool_name in self.tool_relationships.get(tool_name, []):
                related.append({
                    "tool_name": execution.tool_name,
                    "parameters": execution.parameters,
                    "decisions_made": execution.decisions_made,
                    "files_accessed": list(execution.files_accessed),
                    "timestamp": execution.timestamp.isoformat()
                })
            
            # Check if they share file parameters
            elif self._share_file_parameters(parameters, execution.parameters):
                related.append({
                    "tool_name": execution.tool_name,
                    "shared_files": self._get_shared_files(parameters, execution.parameters),
                    "decisions_made": execution.decisions_made,
                    "timestamp": execution.timestamp.isoformat()
                })
        
        return related[-5:]  # Return last 5 related executions
    
    def _get_relevant_files(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get files that are relevant to the current tool execution"""
        if not self.current_context:
            return {}
        
        relevant = {}
        
        # Files mentioned in parameters
        file_params = self._extract_file_parameters(parameters)
        for file_path in file_params:
            if file_path in self.current_context.discovered_files:
                relevant[file_path] = self.current_context.discovered_files[file_path]
        
        # Files discovered by related tools
        for execution in self.current_context.execution_trace:
            if execution.tool_name in self.tool_relationships.get(tool_name, []):
                for file_path in execution.files_accessed:
                    if file_path in self.current_context.discovered_files:
                        relevant[file_path] = self.current_context.discovered_files[file_path]
        
        return relevant
    
    def _get_relevant_decisions(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get decisions that might affect the current tool execution"""
        if not self.current_context:
            return {}
        
        relevant_decisions = {}
        
        # Get decisions from related tools
        for execution in self.current_context.execution_trace:
            if execution.tool_name in self.tool_relationships.get(tool_name, []):
                for decision_type, decision_value in execution.decisions_made.items():
                    relevant_decisions[f"{execution.tool_name}_{decision_type}"] = decision_value
        
        return relevant_decisions
    
    def _extract_decisions(self, tool_name: str, result: Any) -> Dict[str, Any]:
        """Extract implicit decisions from tool results"""
        decisions = {}
        
        try:
            # Try to parse JSON result
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict):
                        result = parsed_result
                except json.JSONDecodeError:
                    pass
            
            if isinstance(result, dict):
                # Extract file-related decisions
                if "file" in result or "files" in result:
                    decisions["file_analysis_approach"] = "detailed" if len(str(result)) > 1000 else "summary"
                
                # Extract search-related decisions
                if "search" in tool_name.lower() or "find" in tool_name.lower():
                    if isinstance(result, dict) and "items" in result:
                        decisions["search_scope"] = "comprehensive" if len(result.get("items", [])) > 10 else "focused"
                
                # Extract git-related decisions
                if "git" in tool_name.lower() or "commit" in tool_name.lower():
                    decisions["git_analysis_depth"] = "deep" if "history" in str(result).lower() else "surface"
                
                # Extract issue-related decisions
                if "issue" in tool_name.lower():
                    decisions["issue_analysis_type"] = "comprehensive" if len(str(result)) > 2000 else "basic"
        
        except Exception as e:
            logger.warning(f"Error extracting decisions from {tool_name}: {e}")
        
        return decisions
    
    def _extract_files_accessed(self, tool_name: str, parameters: Dict[str, Any], result: Any) -> Set[str]:
        """Extract files that were accessed during tool execution"""
        files = set()
        
        # Files from parameters
        files.update(self._extract_file_parameters(parameters))
        
        # Files from result
        try:
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                    if isinstance(parsed_result, dict):
                        result = parsed_result
                except json.JSONDecodeError:
                    pass
            
            if isinstance(result, dict):
                # Extract file paths from various result formats
                if "file" in result:
                    files.add(result["file"])
                if "files" in result and isinstance(result["files"], list):
                    files.update(result["files"])
                if "path" in result:
                    files.add(result["path"])
                if "items" in result and isinstance(result["items"], list):
                    for item in result["items"]:
                        if isinstance(item, dict) and "path" in item:
                            files.add(item["path"])
        
        except Exception as e:
            logger.warning(f"Error extracting files from {tool_name}: {e}")
        
        return files
    
    def _extract_file_parameters(self, parameters: Dict[str, Any]) -> Set[str]:
        """Extract file paths from tool parameters"""
        files = set()
        
        for key, value in parameters.items():
            if isinstance(value, str):
                # Common parameter names that contain file paths
                if any(keyword in key.lower() for keyword in ["file", "path", "directory"]):
                    files.add(value)
                # Check if value looks like a file path
                elif "/" in value or "." in value:
                    files.add(value)
        
        return files
    
    def _detect_conflicts(self, execution: ToolExecution) -> List[Dict[str, Any]]:
        """Detect conflicts with previous executions"""
        conflicts = []
        
        if not self.current_context:
            return conflicts
        
        for prev_execution in self.current_context.execution_trace:
            # Check for conflicting decisions on the same files
            shared_files = execution.files_accessed.intersection(prev_execution.files_accessed)
            if shared_files:
                for decision_type in execution.decisions_made:
                    if decision_type in prev_execution.decisions_made:
                        if execution.decisions_made[decision_type] != prev_execution.decisions_made[decision_type]:
                            conflicts.append({
                                "type": "decision_conflict",
                                "decision_type": decision_type,
                                "current_tool": execution.tool_name,
                                "previous_tool": prev_execution.tool_name,
                                "current_decision": execution.decisions_made[decision_type],
                                "previous_decision": prev_execution.decisions_made[decision_type],
                                "shared_files": list(shared_files)
                            })
        
        return conflicts
    
    def _resolve_conflicts(self, execution: ToolExecution, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts between tool executions"""
        resolution = {
            "timestamp": datetime.now().isoformat(),
            "execution": execution.tool_name,
            "conflicts": conflicts,
            "resolution_strategy": "latest_wins",  # Default strategy
            "resolved_decisions": {}
        }
        
        # Simple resolution: latest execution wins
        for conflict in conflicts:
            decision_type = conflict["decision_type"]
            resolution["resolved_decisions"][decision_type] = conflict["current_decision"]
            
            # Update context with resolved decision
            self.current_context.decisions_made[decision_type] = conflict["current_decision"]
        
        logger.info(f"Resolved {len(conflicts)} conflicts for {execution.tool_name}")
        return resolution
    
    def _update_context_from_execution(self, execution: ToolExecution):
        """Update the execution context with information from the execution"""
        if not self.current_context:
            return
        
        # Update discovered files
        for file_path in execution.files_accessed:
            if file_path not in self.current_context.discovered_files:
                self.current_context.discovered_files[file_path] = {
                    "discovered_by": execution.tool_name,
                    "timestamp": execution.timestamp.isoformat(),
                    "access_count": 1
                }
            else:
                self.current_context.discovered_files[file_path]["access_count"] += 1
        
        # Update global decisions
        for decision_type, decision_value in execution.decisions_made.items():
            self.current_context.decisions_made[decision_type] = decision_value
        
        # Update analyzed components
        if execution.tool_name.startswith("analyze_"):
            component_key = f"{execution.tool_name}_{hash(str(execution.parameters))}"
            self.current_context.analyzed_components[component_key] = {
                "tool": execution.tool_name,
                "parameters": execution.parameters,
                "result_summary": str(execution.result)[:200],
                "timestamp": execution.timestamp.isoformat()
            }
    
    def _generate_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate a cache key for tool execution"""
        # Create a deterministic key from tool name and parameters
        param_str = json.dumps(parameters, sort_keys=True)
        return f"{tool_name}_{hash(param_str)}"
    
    def _share_file_parameters(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> bool:
        """Check if two parameter sets share file references"""
        files1 = self._extract_file_parameters(params1)
        files2 = self._extract_file_parameters(params2)
        return bool(files1.intersection(files2))
    
    def _get_shared_files(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> List[str]:
        """Get files shared between two parameter sets"""
        files1 = self._extract_file_parameters(params1)
        files2 = self._extract_file_parameters(params2)
        return list(files1.intersection(files2))
    
    def _find_related_executions(self, tool_name: str, parameters: Dict[str, Any]) -> List[str]:
        """Find IDs of related executions"""
        if not self.current_context:
            return []
        
        related_ids = []
        for i, execution in enumerate(self.current_context.execution_trace):
            if (execution.tool_name in self.tool_relationships.get(tool_name, []) or
                self._share_file_parameters(parameters, execution.parameters)):
                related_ids.append(f"exec_{i}")
        
        return related_ids
    
    def cleanup(self):
        """Clean up the context manager"""
        if self.current_context:
            logger.info(f"Cleaning up context for session {self.session_id}")
            summary = self.get_execution_summary()
            logger.info(f"Execution summary: {summary}")
        
        self.current_context = None 