"""
Core Agents for Multi-Agent Codebase Intelligence

These agents leverage the existing AgenticCodebaseExplorer and add specialized
capabilities for research planning, code analysis, and implementation.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.llms.openrouter import OpenRouter

from ..agentic_tools import AgenticCodebaseExplorer
from ..llm_client import LLMClient
from ..config import settings
from .events import ResearchTask, CodeComponent, ImplementationStep
from .structured_outputs import QueryAnalysis, ImplementationStrategy

logger = logging.getLogger(__name__)


class BaseAgent:
    """Base class for all agents with common functionality"""
    
    def __init__(self, name: str, llm_client: Optional[LLMClient] = None):
        self.name = name
        self.llm_client = llm_client or LLMClient()
        self.start_time = None
        
    def _start_timing(self):
        self.start_time = time.time()
        
    def _get_elapsed_time(self) -> float:
        if self.start_time:
            return time.time() - self.start_time
        return 0.0
        
    def _log_performance(self, operation: str):
        elapsed = self._get_elapsed_time()
        logger.info(f"[{self.name}] {operation} completed in {elapsed:.2f}s")


class ResearchPlannerAgent(BaseAgent):
    """
    Agent responsible for breaking down complex queries into research tasks
    and determining the optimal investigation strategy.
    """
    
    def __init__(self, codebase_explorer: AgenticCodebaseExplorer, llm_client: Optional[LLMClient] = None):
        super().__init__("ResearchPlanner", llm_client)
        self.explorer = codebase_explorer
        
    async def create_research_plan(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a comprehensive research plan for the given query"""
        self._start_timing()
        
        try:
            # Analyze the query to understand what type of research is needed
            query_analysis = await self._analyze_query_intent(query, context)
            
            # Generate research tasks based on the analysis
            research_tasks = await self._generate_research_tasks(query, query_analysis)
            
            # Estimate complexity and set priority
            complexity = self._estimate_complexity(query, research_tasks)
            priority = self._determine_priority(query, query_analysis)
            
            plan = {
                "query": query,
                "query_analysis": query_analysis,
                "research_tasks": research_tasks,
                "estimated_complexity": complexity,
                "priority": priority,
                "created_at": datetime.now().isoformat(),
                "agent": self.name
            }
            
            self._log_performance("Research plan creation")
            return plan
            
        except Exception as e:
            logger.error(f"[{self.name}] Error creating research plan: {e}")
            return self._create_fallback_plan(query)
    
    async def _analyze_query_intent(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze the query to understand what type of research is needed"""
        
        try:
            # Create LLM for structured output
            llm = OpenRouter(
                api_key=settings.openrouter_api_key,
                model=settings.default_model
            )
            
            # Create structured output program
            program = LLMTextCompletionProgram.from_defaults(
                output_parser=PydanticOutputParser(QueryAnalysis),
                prompt_template_str="""
                Analyze this software development query:
                
                Query: {query}
                Context: {context}
                
                Determine the query type, scope, required knowledge, technical domains, urgency level, and complexity.
                """,
                llm=llm,
                verbose=False
            )
            
            # Get structured output
            analysis = await program.acall(
                query=query,
                context=str(context or {})
            )
            
            return analysis.dict()
            
        except Exception as e:
            logger.warning(f"Failed to analyze query with structured output: {e}")
            return self._default_query_analysis(query)
    
    async def _generate_research_tasks(self, query: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific research tasks based on query analysis"""
        
        tasks = []
        
        # Always start with basic code analysis
        tasks.append({
            "task_id": "code_analysis_1", 
            "task_type": "code_analysis",
            "description": f"Analyze codebase for patterns related to: {query}",
            "agent_type": "CodeAnalysisAgent",
            "parameters": {"query": query, "search_depth": "surface"},
            "dependencies": [],
            "estimated_duration": 45
        })
        
        # Add issue analysis if relevant
        if analysis.get("query_type") in ["bug_investigation", "feature_development"]:
            tasks.append({
                "task_id": "issue_analysis_1",
                "task_type": "issue_analysis", 
                "description": f"Find related issues and their solutions for: {query}",
                "agent_type": "CodeAnalysisAgent",
                "parameters": {"query": query, "include_historical": True},
                "dependencies": [],
                "estimated_duration": 30
            })
        
        # Add pattern detection for complex queries
        if analysis.get("estimated_complexity", 5) > 6:
            tasks.append({
                "task_id": "pattern_analysis_1",
                "task_type": "pattern_detection",
                "description": f"Detect implementation patterns for: {query}",
                "agent_type": "CodeAnalysisAgent", 
                "parameters": {"query": query, "pattern_scope": "implementation"},
                "dependencies": ["code_analysis_1"],
                "estimated_duration": 60
            })
        
        return tasks
    
    def _estimate_complexity(self, query: str, tasks: List[Dict[str, Any]]) -> int:
        """Estimate query complexity on a scale of 1-10"""
        
        base_complexity = 3
        
        # Increase based on number of tasks
        complexity = base_complexity + min(len(tasks), 4)
        
        # Increase based on keywords
        complex_keywords = ["architecture", "refactor", "migration", "performance", "security"]
        if any(keyword in query.lower() for keyword in complex_keywords):
            complexity += 2
            
        # Increase based on scope indicators
        scope_keywords = ["entire", "whole", "all", "complete", "comprehensive"]
        if any(keyword in query.lower() for keyword in scope_keywords):
            complexity += 1
            
        return min(complexity, 10)
    
    def _determine_priority(self, query: str, analysis: Dict[str, Any]) -> str:
        """Determine query priority"""
        
        urgency = analysis.get("urgency_level", "normal")
        if urgency in ["critical", "high"]:
            return urgency
            
        # Check for urgent keywords
        urgent_keywords = ["bug", "error", "broken", "critical", "urgent", "fix"]
        if any(keyword in query.lower() for keyword in urgent_keywords):
            return "high"
            
        return "normal"
    
    def _default_query_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis when LLM parsing fails"""
        return {
            "query_type": "general_inquiry",
            "scope": "multiple_files", 
            "required_knowledge": ["code_patterns"],
            "technical_domains": ["general"],
            "urgency_level": "normal",
            "estimated_complexity": 5
        }
    
    def _create_fallback_plan(self, query: str) -> Dict[str, Any]:
        """Create a basic fallback plan when planning fails"""
        return {
            "query": query,
            "query_analysis": self._default_query_analysis(query),
            "research_tasks": [{
                "task_id": "basic_analysis",
                "task_type": "code_analysis", 
                "description": f"Basic analysis for: {query}",
                "agent_type": "CodeAnalysisAgent",
                "parameters": {"query": query},
                "dependencies": [],
                "estimated_duration": 60
            }],
            "estimated_complexity": 5,
            "priority": "normal",
            "created_at": datetime.now().isoformat(),
            "agent": self.name,
            "fallback": True
        }


class CodeAnalysisAgent(BaseAgent):
    """
    Agent responsible for deep code analysis, pattern detection,
    and leveraging the existing codebase exploration tools.
    """
    
    def __init__(self, codebase_explorer: AgenticCodebaseExplorer, llm_client: Optional[LLMClient] = None):
        super().__init__("CodeAnalysis", llm_client)
        self.explorer = codebase_explorer
        
    async def execute_research_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific research task"""
        self._start_timing()
        
        task_type = task.get("task_type")
        parameters = task.get("parameters", {})
        
        try:
            if task_type == "code_analysis":
                result = await self._perform_code_analysis(parameters)
            elif task_type == "issue_analysis":
                result = await self._perform_issue_analysis(parameters)
            elif task_type == "pattern_detection":
                result = await self._perform_pattern_detection(parameters)
            elif task_type == "historical_analysis":
                result = await self._perform_historical_analysis(parameters)
            else:
                result = await self._perform_general_analysis(parameters)
                
            result.update({
                "task_id": task.get("task_id"),
                "completed_at": datetime.now().isoformat(),
                "agent": self.name,
                "execution_time": self._get_elapsed_time()
            })
            
            self._log_performance(f"Task {task_type}")
            return result
            
        except Exception as e:
            logger.error(f"[{self.name}] Error executing task {task.get('task_id')}: {e}")
            return self._create_error_result(task, str(e))
    
    async def _perform_code_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive code analysis"""
        
        query = parameters.get("query", "")
        search_depth = parameters.get("search_depth", "surface")
        
        results = {}
        
        # Search for relevant code patterns
        try:
            search_results = self.explorer.search_codebase(query)
            results["code_search"] = search_results
        except Exception as e:
            logger.warning(f"Code search failed: {e}")
            results["code_search"] = "Search failed"
        
        # Analyze file structure if deep search requested
        if search_depth == "deep":
            try:
                structure_analysis = self.explorer.analyze_file_structure()
                results["structure_analysis"] = structure_analysis
            except Exception as e:
                logger.warning(f"Structure analysis failed: {e}")
        
        # Find related files
        try:
            semantic_results = self.explorer.semantic_content_search(query)
            results["semantic_search"] = semantic_results
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
        
        return {
            "analysis_type": "code_analysis",
            "query": query,
            "results": results,
            "confidence": self._calculate_confidence(results)
        }
    
    async def _perform_issue_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze related issues and their solutions"""
        
        query = parameters.get("query", "")
        include_historical = parameters.get("include_historical", False)
        
        results = {}
        
        # Find related issues
        try:
            related_issues = self.explorer.related_issues(query)
            results["related_issues"] = related_issues
        except Exception as e:
            logger.warning(f"Issue search failed: {e}")
            results["related_issues"] = "Issue search failed"
        
        # Get issue-related files if available
        try:
            issue_files = self.explorer.find_issue_related_files(query)
            results["issue_files"] = issue_files
        except Exception as e:
            logger.warning(f"Issue file search failed: {e}")
        
        return {
            "analysis_type": "issue_analysis", 
            "query": query,
            "results": results,
            "confidence": self._calculate_confidence(results)
        }
    
    async def _perform_pattern_detection(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Detect implementation patterns"""
        
        query = parameters.get("query", "")
        pattern_scope = parameters.get("pattern_scope", "implementation")
        
        results = {}
        
        # Use semantic search to find patterns
        try:
            pattern_results = self.explorer.semantic_content_search(f"implementation patterns {query}")
            results["pattern_search"] = pattern_results
        except Exception as e:
            logger.warning(f"Pattern search failed: {e}")
        
        # Generate code examples based on patterns
        try:
            code_example = self.explorer.generate_code_example(f"Example implementation for {query}")
            results["code_example"] = code_example
        except Exception as e:
            logger.warning(f"Code example generation failed: {e}")
        
        return {
            "analysis_type": "pattern_detection",
            "query": query,
            "results": results,
            "confidence": self._calculate_confidence(results)
        }
    
    async def _perform_historical_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze historical evolution and decisions"""
        
        query = parameters.get("query", "")
        include_git_history = parameters.get("include_git_history", False)
        
        results = {}
        
        # Analyze feature evolution
        try:
            evolution = self.explorer.summarize_feature_evolution(query)
            results["feature_evolution"] = evolution
        except Exception as e:
            logger.warning(f"Feature evolution analysis failed: {e}")
        
        # Find who implemented features
        try:
            implementer_info = self.explorer.who_implemented_this(query)
            results["implementer_info"] = implementer_info
        except Exception as e:
            logger.warning(f"Implementer analysis failed: {e}")
        
        return {
            "analysis_type": "historical_analysis",
            "query": query, 
            "results": results,
            "confidence": self._calculate_confidence(results)
        }
    
    async def _perform_general_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Perform general analysis when task type is unknown"""
        
        query = parameters.get("query", "")
        
        # Use the explorer's general query method
        try:
            general_result = await self.explorer.query(query)
            return {
                "analysis_type": "general_analysis",
                "query": query,
                "results": {"general_response": general_result},
                "confidence": 0.7  # Medium confidence for general queries
            }
        except Exception as e:
            logger.error(f"General analysis failed: {e}")
            return self._create_error_result({"task_id": "general"}, str(e))
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence score based on result quality"""
        
        if not results:
            return 0.0
        
        # Count successful operations
        successful_ops = sum(1 for v in results.values() if v and "failed" not in str(v).lower())
        total_ops = len(results)
        
        if total_ops == 0:
            return 0.0
            
        base_confidence = successful_ops / total_ops
        
        # Boost confidence if we have multiple types of successful results
        if successful_ops > 1:
            base_confidence = min(base_confidence + 0.1, 1.0)
            
        return base_confidence
    
    def _create_error_result(self, task: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create an error result for failed tasks"""
        return {
            "task_id": task.get("task_id"),
            "analysis_type": "error",
            "error": error_message,
            "completed_at": datetime.now().isoformat(),
            "agent": self.name,
            "execution_time": self._get_elapsed_time(),
            "confidence": 0.0
        }


class ImplementationAgent(BaseAgent):
    """
    Agent responsible for generating implementation plans and code
    based on research results.
    """
    
    def __init__(self, codebase_explorer: AgenticCodebaseExplorer, llm_client: Optional[LLMClient] = None):
        super().__init__("Implementation", llm_client)
        self.explorer = codebase_explorer
        
    async def create_implementation_plan(self, query: str, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive implementation plan"""
        self._start_timing()
        
        try:
            # Analyze research results to understand what needs to be implemented
            implementation_strategy = await self._analyze_implementation_strategy(query, research_results)
            
            # Generate code components
            code_components = await self._generate_code_components(query, research_results, implementation_strategy)
            
            # Create implementation steps
            implementation_steps = await self._create_implementation_steps(implementation_strategy, code_components)
            
            # Assess risks and create validation plan
            risk_assessment = await self._assess_implementation_risks(query, implementation_strategy)
            validation_steps = await self._create_validation_plan(implementation_strategy, code_components)
            
            plan = {
                "original_query": query,
                "research_summary": self._summarize_research(research_results),
                "implementation_strategy": implementation_strategy,
                "code_components": code_components,
                "implementation_steps": implementation_steps,
                "validation_steps": validation_steps,
                "risk_assessment": risk_assessment,
                "created_at": datetime.now().isoformat(),
                "agent": self.name,
                "estimated_effort": self._estimate_implementation_effort(implementation_steps)
            }
            
            self._log_performance("Implementation plan creation")
            return plan
            
        except Exception as e:
            logger.error(f"[{self.name}] Error creating implementation plan: {e}")
            return self._create_fallback_implementation_plan(query, research_results)
    
    async def _analyze_implementation_strategy(self, query: str, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze research results to determine implementation strategy"""
        
        # Extract key insights from research
        code_patterns = []
        existing_solutions = []
        
        for result in research_results.get("research_results", []):
            if result.get("analysis_type") == "code_analysis":
                code_patterns.extend(self._extract_patterns_from_result(result))
            elif result.get("analysis_type") == "issue_analysis":
                existing_solutions.extend(self._extract_solutions_from_result(result))
        
        try:
            # Create LLM for structured output
            llm = OpenRouter(
                api_key=settings.openrouter_api_key,
                model=settings.default_model
            )
            
            # Create structured output program
            program = LLMTextCompletionProgram.from_defaults(
                output_parser=PydanticOutputParser(ImplementationStrategy),
                prompt_template_str="""
                Based on the research findings, determine the best implementation strategy:
                
                Query: {query}
                Code Patterns: {code_patterns}
                Existing Solutions: {existing_solutions}
                
                Provide a high-level approach, technology choices, file organization, integration points, and testing strategy.
                """,
                llm=llm,
                verbose=False
            )
            
            # Get structured output
            strategy = await program.acall(
                query=query,
                code_patterns=str(code_patterns[:3]),
                existing_solutions=str(existing_solutions[:3])
            )
            
            return strategy.dict()
            
        except Exception as e:
            logger.warning(f"Failed to generate implementation strategy with structured output: {e}")
            return self._default_implementation_strategy(query)
    
    async def _generate_code_components(self, query: str, research_results: Dict[str, Any], strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate code components based on the implementation strategy"""
        
        components = []
        
        # Generate main implementation using the existing code generation tool
        try:
            main_code = self.explorer.write_complete_code(
                description=f"Implementation for: {query}",
                language="python",  # Default to Python, could be made configurable
                output_format="raw"
            )
            
            components.append({
                "component_type": "main_implementation",
                "name": self._generate_component_name(query),
                "description": f"Main implementation for {query}",
                "code": main_code,
                "language": "python",
                "dependencies": [],
                "tests": None,
                "documentation": None
            })
        except Exception as e:
            logger.warning(f"Main code generation failed: {e}")
        
        # Generate additional components based on strategy
        additional_components = strategy.get("additional_components", [])
        for comp_spec in additional_components[:3]:  # Limit to 3 additional components
            try:
                comp_code = self.explorer.generate_code_example(
                    description=comp_spec.get("description", ""),
                    context_files=[]
                )
                
                components.append({
                    "component_type": comp_spec.get("type", "utility"),
                    "name": comp_spec.get("name", "generated_component"),
                    "description": comp_spec.get("description", ""),
                    "code": comp_code,
                    "language": comp_spec.get("language", "python"),
                    "dependencies": comp_spec.get("dependencies", []),
                    "tests": None,
                    "documentation": None
                })
            except Exception as e:
                logger.warning(f"Additional component generation failed: {e}")
        
        return components
    
    async def _create_implementation_steps(self, strategy: Dict[str, Any], components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create step-by-step implementation plan"""
        
        steps = []
        
        # Step 1: Create main files
        for i, component in enumerate(components):
            steps.append({
                "step_id": f"create_component_{i}",
                "description": f"Create {component['name']} component",
                "action_type": "create_file",
                "parameters": {
                    "file_path": f"{component['name']}.{self._get_file_extension(component['language'])}",
                    "content": component["code"]
                },
                "dependencies": [],
                "validation_criteria": ["Syntax validation", "Import validation"]
            })
        
        # Step 2: Integration steps
        if len(components) > 1:
            steps.append({
                "step_id": "integrate_components",
                "description": "Integrate all components",
                "action_type": "modify_file",
                "parameters": {
                    "integration_type": "import_and_wire"
                },
                "dependencies": [f"create_component_{i}" for i in range(len(components))],
                "validation_criteria": ["Integration test"]
            })
        
        # Step 3: Testing
        steps.append({
            "step_id": "create_tests",
            "description": "Create comprehensive tests",
            "action_type": "create_file",
            "parameters": {
                "test_type": "unit_and_integration"
            },
            "dependencies": ["create_component_0"],
            "validation_criteria": ["Test coverage > 80%"]
        })
        
        return steps
    
    async def _create_validation_plan(self, strategy: Dict[str, Any], components: List[Dict[str, Any]]) -> List[str]:
        """Create validation steps for the implementation"""
        
        validation_steps = [
            "Syntax validation for all generated code",
            "Import and dependency validation",
            "Unit test execution",
            "Integration test execution",
            "Code style and formatting check",
            "Security vulnerability scan",
            "Performance impact assessment"
        ]
        
        # Add language-specific validations
        languages = set(comp.get("language", "python") for comp in components)
        for lang in languages:
            if lang == "python":
                validation_steps.extend([
                    "Python linting with flake8/pylint",
                    "Type checking with mypy"
                ])
            elif lang == "javascript":
                validation_steps.extend([
                    "ESLint validation",
                    "Jest test execution"
                ])
        
        return validation_steps
    
    async def _assess_implementation_risks(self, query: str, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Assess potential risks in the implementation"""
        
        risks = {
            "complexity_risk": self._assess_complexity_risk(strategy),
            "integration_risk": self._assess_integration_risk(strategy),
            "performance_risk": self._assess_performance_risk(query, strategy),
            "security_risk": self._assess_security_risk(query, strategy),
            "maintenance_risk": self._assess_maintenance_risk(strategy)
        }
        
        # Add mitigation strategies
        mitigations = {
            "complexity_risk": ["Break down into smaller components", "Add comprehensive documentation"],
            "integration_risk": ["Create integration tests", "Use dependency injection"],
            "performance_risk": ["Add performance monitoring", "Implement caching where appropriate"],
            "security_risk": ["Security review", "Input validation", "Access control checks"],
            "maintenance_risk": ["Code documentation", "Clear naming conventions", "Automated tests"]
        }
        
        return {
            "risk_assessment": risks,
            "mitigation_strategies": mitigations,
            "overall_risk_level": self._calculate_overall_risk(risks)
        }
    
    def _extract_patterns_from_result(self, result: Dict[str, Any]) -> List[str]:
        """Extract useful patterns from code analysis results"""
        patterns = []
        results = result.get("results", {})
        
        # Extract from different result types
        for key, value in results.items():
            if isinstance(value, str) and len(value) > 50:
                # Extract key insights from text results
                lines = value.split('\n')[:5]  # First 5 lines
                patterns.extend([line.strip() for line in lines if line.strip()])
        
        return patterns[:10]  # Limit to 10 patterns
    
    def _extract_solutions_from_result(self, result: Dict[str, Any]) -> List[str]:
        """Extract existing solutions from issue analysis results"""
        solutions = []
        results = result.get("results", {})
        
        # Look for solution-related content
        for key, value in results.items():
            if isinstance(value, str) and any(word in key.lower() for word in ["solution", "fix", "resolution"]):
                solutions.append(value[:200])  # First 200 chars
        
        return solutions[:5]  # Limit to 5 solutions
    
    def _generate_component_name(self, query: str) -> str:
        """Generate an appropriate component name from the query"""
        # Simple name generation - could be improved with NLP
        words = query.lower().split()
        meaningful_words = [w for w in words if len(w) > 3 and w not in ['the', 'and', 'for', 'with']]
        return '_'.join(meaningful_words[:3]) if meaningful_words else 'implementation'
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for a programming language"""
        extensions = {
            "python": "py",
            "javascript": "js", 
            "typescript": "ts",
            "java": "java",
            "go": "go",
            "rust": "rs",
            "csharp": "cs"
        }
        return extensions.get(language.lower(), "txt")
    
    def _assess_complexity_risk(self, strategy: Dict[str, Any]) -> str:
        """Assess complexity risk of the implementation"""
        # Simple heuristic based on strategy complexity
        components = strategy.get("additional_components", [])
        if len(components) > 5:
            return "high"
        elif len(components) > 2:
            return "medium"
        else:
            return "low"
    
    def _assess_integration_risk(self, strategy: Dict[str, Any]) -> str:
        """Assess integration risk"""
        integration_points = strategy.get("integration_points", [])
        return "high" if len(integration_points) > 3 else "medium" if len(integration_points) > 1 else "low"
    
    def _assess_performance_risk(self, query: str, strategy: Dict[str, Any]) -> str:
        """Assess performance risk"""
        perf_keywords = ["performance", "speed", "optimization", "large scale", "high volume"]
        return "high" if any(keyword in query.lower() for keyword in perf_keywords) else "low"
    
    def _assess_security_risk(self, query: str, strategy: Dict[str, Any]) -> str:
        """Assess security risk"""
        security_keywords = ["authentication", "authorization", "security", "password", "token", "api"]
        return "high" if any(keyword in query.lower() for keyword in security_keywords) else "medium"
    
    def _assess_maintenance_risk(self, strategy: Dict[str, Any]) -> str:
        """Assess maintenance risk"""
        # Simple heuristic - more components = higher maintenance risk
        components = strategy.get("additional_components", [])
        return "high" if len(components) > 4 else "medium" if len(components) > 1 else "low"
    
    def _calculate_overall_risk(self, risks: Dict[str, str]) -> str:
        """Calculate overall risk level"""
        risk_scores = {"low": 1, "medium": 2, "high": 3}
        total_score = sum(risk_scores.get(risk, 1) for risk in risks.values())
        avg_score = total_score / len(risks)
        
        if avg_score >= 2.5:
            return "high"
        elif avg_score >= 1.5:
            return "medium"
        else:
            return "low"
    
    def _estimate_implementation_effort(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate implementation effort"""
        
        total_steps = len(steps)
        estimated_hours = total_steps * 2  # 2 hours per step baseline
        
        # Adjust based on step complexity
        for step in steps:
            if step.get("action_type") == "create_file":
                estimated_hours += 1
            elif step.get("action_type") == "integrate_components":
                estimated_hours += 3
            elif "test" in step.get("description", "").lower():
                estimated_hours += 2
        
        return {
            "total_steps": total_steps,
            "estimated_hours": estimated_hours,
            "estimated_days": max(1, estimated_hours // 8),
            "complexity_level": "high" if estimated_hours > 16 else "medium" if estimated_hours > 8 else "low"
        }
    
    def _summarize_research(self, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize research results for inclusion in implementation plan"""
        
        summary = {
            "total_analyses": len(research_results.get("research_results", [])),
            "analysis_types": [],
            "key_findings": [],
            "confidence_scores": []
        }
        
        for result in research_results.get("research_results", []):
            analysis_type = result.get("analysis_type", "unknown")
            summary["analysis_types"].append(analysis_type)
            
            confidence = result.get("confidence", 0.0)
            summary["confidence_scores"].append(confidence)
            
            # Extract key findings
            if result.get("results"):
                finding = f"{analysis_type}: {len(result['results'])} results found"
                summary["key_findings"].append(finding)
        
        # Calculate average confidence
        if summary["confidence_scores"]:
            summary["average_confidence"] = sum(summary["confidence_scores"]) / len(summary["confidence_scores"])
        else:
            summary["average_confidence"] = 0.0
        
        return summary
    
    def _default_implementation_strategy(self, query: str) -> Dict[str, Any]:
        """Default implementation strategy when analysis fails"""
        return {
            "approach": "incremental_development",
            "technology_choices": ["python"],
            "file_organization": "modular",
            "integration_points": [],
            "testing_strategy": "unit_and_integration",
            "additional_components": []
        }
    
    def _create_fallback_implementation_plan(self, query: str, research_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic fallback implementation plan"""
        return {
            "original_query": query,
            "research_summary": {"error": "Research summary failed"},
            "implementation_strategy": self._default_implementation_strategy(query),
            "code_components": [],
            "implementation_steps": [{
                "step_id": "manual_implementation",
                "description": "Manual implementation required",
                "action_type": "manual",
                "parameters": {"query": query},
                "dependencies": [],
                "validation_criteria": ["Manual review"]
            }],
            "validation_steps": ["Manual validation required"],
            "risk_assessment": {
                "overall_risk_level": "high",
                "reason": "Automated planning failed"
            },
            "created_at": datetime.now().isoformat(),
            "agent": self.name,
            "fallback": True,
            "estimated_effort": {
                "estimated_hours": 8,
                "complexity_level": "medium"
            }
        } 