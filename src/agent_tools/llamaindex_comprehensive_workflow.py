"""
LlamaIndex AgentWorkflow for Comprehensive Repository Analysis

This implements the proper LlamaIndex AgentWorkflow pattern from:
- https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/
- https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/
- https://www.llamaindex.ai/blog/introducing-agentworkflow-a-powerful-system-for-building-ai-agent-systems
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from llama_index.core.workflow import (
    Context, Event, StartEvent, StopEvent, Workflow, step
)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import BaseTool
from llama_index.core.llms import LLM

from .file_operations import FileOperations
from .search_operations import SearchOperations
from .context_manager import ContextManager
from .llm_config import get_llm_instance

logger = logging.getLogger(__name__)

# Workflow Events
class StructureAnalysisEvent(Event):
    """Event triggered after structure analysis"""
    structure_data: Dict[str, Any]

class DependencyAnalysisEvent(Event):
    """Event triggered after dependency analysis"""
    dependency_data: Dict[str, Any]

class SecurityAnalysisEvent(Event):
    """Event triggered after security analysis"""
    security_findings: List[Dict[str, Any]]

class PerformanceAnalysisEvent(Event):
    """Event triggered after performance analysis"""
    performance_findings: List[Dict[str, Any]]

class QualityAnalysisEvent(Event):
    """Event triggered after quality analysis"""
    quality_findings: List[Dict[str, Any]]

class ReportGenerationEvent(Event):
    """Event triggered to generate final report"""
    all_findings: Dict[str, Any]

@dataclass
class AnalysisRequest:
    """Initial analysis request"""
    query: str
    focus_areas: List[str]
    session_id: str
    repo_path: str

class ComprehensiveAnalysisWorkflow(Workflow):
    """
    LlamaIndex AgentWorkflow for comprehensive repository analysis.
    
    This uses proper LlamaIndex workflow patterns with:
    - Context for shared state
    - Events for step communication
    - FunctionCallingAgentWorker for tool execution
    - Proper step decorators
    """
    
    def __init__(
        self,
        session_id: str,
        repo_path: str,
        context_manager: ContextManager,
        llm: Optional[LLM] = None,
        tools: Optional[List[BaseTool]] = None
    ):
        super().__init__(timeout=600.0, verbose=True)
        
        self.session_id = session_id
        self.repo_path = repo_path
        self.context_manager = context_manager
        
        # Initialize LLM - All OpenRouter models support function calling
        if llm:
            self.llm = llm
        else:
            # Use default model from config
            self.llm = get_llm_instance()
        
        # Initialize tools
        self.file_ops = FileOperations(repo_path, chunk_store_instance=None)
        self.search_ops = SearchOperations(repo_path)
        
        # Create LlamaIndex tools from the operations
        from llama_index.core.tools import FunctionTool
        
        # Create tools for the agent
        analysis_tools = [
            FunctionTool.from_defaults(
                fn=self.file_ops.explore_directory,
                name="explore_directory",
                description="Explore directory contents with metadata"
            ),
            FunctionTool.from_defaults(
                fn=self.file_ops.read_file,
                name="read_file", 
                description="Read complete file contents"
            ),
            FunctionTool.from_defaults(
                fn=self.file_ops.analyze_file_structure,
                name="analyze_file_structure",
                description="Analyze file structure and relationships"
            ),
            FunctionTool.from_defaults(
                fn=self.search_ops.search_codebase,
                name="search_codebase",
                description="Search through codebase files for patterns"
            ),
            FunctionTool.from_defaults(
                fn=self.search_ops.find_related_files,
                name="find_related_files",
                description="Find files related to a given file"
            ),
            FunctionTool.from_defaults(
                fn=self.search_ops.semantic_content_search,
                name="semantic_content_search",
                description="Search for content semantically across files"
            )
        ]
        
        # Add any additional tools passed in
        all_tools = analysis_tools + (tools or [])
        
        # Create function calling agent worker
        self.agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=all_tools,
            llm=self.llm,
            verbose=True
        )
        
        # Create agent from worker
        from llama_index.core.agent import AgentRunner
        self.agent = AgentRunner(self.agent_worker)
        
        logger.info(f"Initialized ComprehensiveAnalysisWorkflow for {repo_path}")
    
    @step
    async def start_analysis(
        self, ctx: Context, ev: StartEvent
    ) -> StructureAnalysisEvent:
        """Entry point: Start the comprehensive analysis"""
        logger.info("Starting comprehensive repository analysis")
        
        # Extract request from StartEvent
        request = ev.get("request")
        if not request:
            raise ValueError("AnalysisRequest required in StartEvent")
        
        # Store request in context
        ctx.data["request"] = request
        ctx.data["start_time"] = datetime.now()
        ctx.data["stages_completed"] = []
        
        # Always start with structure analysis
        logger.info("Stage 1: Beginning structure analysis")
        structure_data = await self._analyze_structure(ctx)
        
        ctx.data["stages_completed"].append("structure")
        
        return StructureAnalysisEvent(structure_data=structure_data)
    
    @step
    async def handle_structure_analysis(
        self, ctx: Context, ev: StructureAnalysisEvent
    ) -> DependencyAnalysisEvent:
        """Handle structure analysis results and move to dependencies"""
        logger.info("Structure analysis completed, starting dependency analysis")
        
        # Store structure data in context
        ctx.data["structure_data"] = ev.structure_data
        
        request = ctx.data["request"]
        if "dependencies" in request.focus_areas:
            logger.info("Stage 2: Beginning dependency analysis")
            dependency_data = await self._analyze_dependencies(ctx)
            ctx.data["stages_completed"].append("dependencies")
        else:
            logger.info("Skipping dependency analysis (not in focus areas)")
            dependency_data = {}
        
        return DependencyAnalysisEvent(dependency_data=dependency_data)
    
    @step
    async def handle_dependency_analysis(
        self, ctx: Context, ev: DependencyAnalysisEvent
    ) -> SecurityAnalysisEvent:
        """Handle dependency analysis results and move to security"""
        logger.info("Dependency analysis completed, starting security analysis")
        
        # Store dependency data in context
        ctx.data["dependency_data"] = ev.dependency_data
        
        request = ctx.data["request"]
        if "security" in request.focus_areas:
            logger.info("Stage 3: Beginning security analysis")
            security_findings = await self._analyze_security(ctx)
            ctx.data["stages_completed"].append("security")
        else:
            logger.info("Skipping security analysis (not in focus areas)")
            security_findings = []
        
        return SecurityAnalysisEvent(security_findings=security_findings)
    
    @step
    async def handle_security_analysis(
        self, ctx: Context, ev: SecurityAnalysisEvent
    ) -> PerformanceAnalysisEvent:
        """Handle security analysis results and move to performance"""
        logger.info("Security analysis completed, starting performance analysis")
        
        # Store security data in context
        ctx.data["security_findings"] = ev.security_findings
        
        request = ctx.data["request"]
        if "performance" in request.focus_areas:
            logger.info("Stage 4: Beginning performance analysis")
            performance_findings = await self._analyze_performance(ctx)
            ctx.data["stages_completed"].append("performance")
        else:
            logger.info("Skipping performance analysis (not in focus areas)")
            performance_findings = []
        
        return PerformanceAnalysisEvent(performance_findings=performance_findings)
    
    @step
    async def handle_performance_analysis(
        self, ctx: Context, ev: PerformanceAnalysisEvent
    ) -> QualityAnalysisEvent:
        """Handle performance analysis results and move to quality"""
        logger.info("Performance analysis completed, starting quality analysis")
        
        # Store performance data in context
        ctx.data["performance_findings"] = ev.performance_findings
        
        request = ctx.data["request"]
        if "quality" in request.focus_areas:
            logger.info("Stage 5: Beginning quality analysis")
            quality_findings = await self._analyze_quality(ctx)
            ctx.data["stages_completed"].append("quality")
        else:
            logger.info("Skipping quality analysis (not in focus areas)")
            quality_findings = []
        
        return QualityAnalysisEvent(quality_findings=quality_findings)
    
    @step
    async def handle_quality_analysis(
        self, ctx: Context, ev: QualityAnalysisEvent
    ) -> ReportGenerationEvent:
        """Handle quality analysis results and prepare for report generation"""
        logger.info("Quality analysis completed, preparing report generation")
        
        # Store quality data in context
        ctx.data["quality_findings"] = ev.quality_findings
        
        # Compile all findings
        all_findings = {
            "structure_data": ctx.data.get("structure_data", {}),
            "dependency_data": ctx.data.get("dependency_data", {}),
            "security_findings": ctx.data.get("security_findings", []),
            "performance_findings": ctx.data.get("performance_findings", []),
            "quality_findings": ctx.data.get("quality_findings", [])
        }
        
        return ReportGenerationEvent(all_findings=all_findings)
    
    @step
    async def handle_report_generation(
        self, ctx: Context, ev: ReportGenerationEvent
    ) -> StopEvent:
        """Generate final comprehensive report"""
        logger.info("Stage 6: Generating comprehensive report")
        
        request = ctx.data["request"]
        start_time = ctx.data["start_time"]
        stages_completed = ctx.data["stages_completed"]
        
        # Generate comprehensive report
        report = {
            "analysis_metadata": {
                "session_id": request.session_id,
                "repo_path": request.repo_path,
                "query": request.query,
                "focus_areas": request.focus_areas,
                "stages_completed": stages_completed,
                "total_duration_seconds": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            },
            "repository_overview": self._generate_overview(ev.all_findings["structure_data"]),
            "dependencies_analysis": ev.all_findings["dependency_data"],
            "security_analysis": {
                "findings_count": len(ev.all_findings["security_findings"]),
                "findings": ev.all_findings["security_findings"],
                "risk_level": self._calculate_risk_level(ev.all_findings["security_findings"])
            },
            "performance_analysis": {
                "findings_count": len(ev.all_findings["performance_findings"]),
                "findings": ev.all_findings["performance_findings"],
                "performance_score": self._calculate_performance_score(ev.all_findings["performance_findings"])
            },
            "quality_analysis": {
                "findings": ev.all_findings["quality_findings"],
                "quality_metrics": self._calculate_quality_metrics(ev.all_findings["quality_findings"])
            },
            "recommendations": self._generate_recommendations(ev.all_findings),
            "summary": self._generate_summary(ev.all_findings, stages_completed)
        }
        
        # Store final report in context
        ctx.data["final_report"] = report
        ctx.data["stages_completed"].append("report_generation")
        
        logger.info(f"Comprehensive analysis completed in {report['analysis_metadata']['total_duration_seconds']:.2f}s")
        
        return StopEvent(result=report)
    
    # Analysis implementation methods (simplified for brevity)
    async def _analyze_structure(self, ctx: Context) -> Dict[str, Any]:
        """Analyze repository structure"""
        import json
        
        try:
            # Get root directory structure
            root_structure = json.loads(self.file_ops.explore_directory(''))
            
            # Get overall file structure analysis
            structure_analysis = json.loads(self.file_ops.analyze_file_structure(''))
            
            return {
                "root": root_structure,
                "analysis": structure_analysis,
                "summary": {
                    "total_files": len(root_structure.get('items', [])),
                    "directories": [item for item in root_structure.get('items', []) if item['type'] == 'directory']
                }
            }
        except Exception as e:
            logger.error(f"Structure analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_dependencies(self, ctx: Context) -> Dict[str, Any]:
        """Analyze dependencies"""
        import json
        from pathlib import Path
        
        try:
            dependencies = {}
            req_files = ['requirements.txt', 'requirements.sglang.txt', 'package.json']
            
            for req_file in req_files:
                if (Path(self.repo_path) / req_file).exists():
                    try:
                        content = json.loads(self.file_ops.read_file(req_file))
                        dependencies[req_file] = {
                            'content': content.get('content', ''),
                            'size': content.get('size', 0),
                            'lines': content.get('lines', 0)
                        }
                        
                        # Parse Python requirements
                        if req_file.endswith('.txt'):
                            deps = self._parse_python_requirements(content.get('content', ''))
                            dependencies[req_file]['parsed_dependencies'] = deps
                            
                    except Exception as e:
                        logger.warning(f"Could not read {req_file}: {e}")
            
            return dependencies
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_security(self, ctx: Context) -> List[Dict[str, Any]]:
        """Analyze security using LLM agent"""
        import json
        from pathlib import Path
        
        try:
            security_findings = []
            
            # First do basic pattern matching for quick wins
            security_patterns = [
                ("hardcoded_secrets", "password|secret|key|token"),
                ("sql_injection", "execute|query.*%|format.*sql"),
                ("unsafe_imports", "pickle|yaml.load|eval|exec")
            ]
            
            pattern_results = []
            for pattern_name, pattern in security_patterns:
                try:
                    search_result = json.loads(self.search_ops.search_codebase(
                        query=pattern,
                        file_types=['.py', '.js', '.ts'],
                        directory_path=None
                    ))
                    
                    if search_result.get('files_with_matches', 0) > 0:
                        pattern_results.append({
                            'pattern': pattern_name,
                            'files_affected': search_result.get('files_with_matches', 0),
                            'matches': search_result.get('results', [])[:3]  # First 3 matches
                        })
                except Exception as e:
                    logger.warning(f"Security pattern search failed for {pattern_name}: {e}")
            
            # Check for sensitive files
            sensitive_files = ['.env', '.env.local', 'config.ini']
            found_sensitive = []
            for filename in sensitive_files:
                if (Path(self.repo_path) / filename).exists():
                    found_sensitive.append(filename)
            
            # Use LLM agent to analyze security comprehensively
            logger.info("Using LLM agent for comprehensive security analysis...")
            
            # Create security analysis prompt
            security_prompt = f"""
You are a security analyst. Use the available tools to perform a comprehensive security analysis of this repository.

Initial pattern matching found:
{json.dumps(pattern_results, indent=2) if pattern_results else "No concerning patterns found"}

Sensitive files found: {found_sensitive if found_sensitive else "None"}

TASK: You MUST perform an EXHAUSTIVE security analysis. Follow these steps IN ORDER:

**STEP 1: Repository Exploration**
1. explore_directory("") - Check root structure and identify key directories
2. explore_directory("src") - Explore main source code
3. explore_directory("api") if it exists - Check API endpoints
4. explore_directory("config") or similar if found

**STEP 2: Configuration & Secrets Analysis**
5. search_codebase with query="password" to find hardcoded passwords
6. search_codebase with query="secret" to find API secrets  
7. search_codebase with query="token" to find auth tokens
8. search_codebase with query="key" to find API keys
9. read_file on ANY config files you find (.env, config.py, settings.py, *.yml, *.json)

**STEP 3: Code Security Analysis** 
10. search_codebase with query="eval(" to find code injection risks
11. search_codebase with query="exec(" to find code execution risks
12. search_codebase with query="pickle" to find unsafe serialization
13. search_codebase with query="subprocess" to find command injection risks
14. search_codebase with query="os.system" to find shell injection risks

**STEP 4: Authentication & Authorization**
15. search_codebase with query="login" to find auth implementations
16. search_codebase with query="authenticate" to find auth logic
17. search_codebase with query="authorize" to find authorization logic
18. read_file on auth-related Python files you discover

**STEP 5: Dependency Security**
19. search_codebase with query="import" to find all imports
20. read_file on requirements.txt, package.json if they exist
21. Look for known vulnerable packages

**REQUIREMENTS:**
- You MUST examine AT LEAST 10 different files
- You MUST perform ALL 21 steps above
- You MUST provide detailed findings for EACH security issue found
- If you find fewer than 3 security issues, you haven't looked hard enough - keep searching

After completing ALL steps, provide findings in this EXACT JSON format:
[
  {{
    "category": "hardcoded_secrets|code_injection|unsafe_deserialization|auth_bypass|weak_crypto|info_disclosure|command_injection|dependency_vulnerability|configuration_error|access_control",
    "severity": "high|medium|low", 
    "description": "detailed description of the security issue",
    "file_path": "exact/path/to/file",
    "line_numbers": [1, 2, 3],
    "risk_level": "High|Medium|Low",
    "remediation": "specific actionable steps to fix this issue",
    "evidence": "code snippet or evidence of the issue"
  }}
]

START NOW with step 1 - explore_directory("")"""

            try:
                # Use the agent to analyze
                response = await self.agent.achat(security_prompt)
                
                # Try to parse JSON response
                response_text = str(response)
                if response_text:
                    # Look for JSON in the response
                    import re
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        security_findings = json.loads(json_match.group())
                    else:
                        # Fallback: create findings from pattern results
                        security_findings = []
                        for result in pattern_results:
                            security_findings.append({
                                'category': result['pattern'],
                                'severity': 'medium',
                                'description': f"Found {result['files_affected']} files with potential {result['pattern']} issues",
                                'files_affected': result['files_affected'],
                                'risk_level': 'Medium',
                                'remediation': f"Review and remediate {result['pattern']} vulnerabilities"
                            })
                        
                        # Add sensitive file findings
                        for filename in found_sensitive:
                            security_findings.append({
                                'category': 'sensitive_files',
                                'severity': 'high',
                                'description': f'Sensitive configuration file exposed: {filename}',
                                'file_path': filename,
                                'risk_level': 'High',
                                'remediation': f'Move {filename} to secure location and add to .gitignore'
                            })
                else:
                    security_findings = []
                    
            except Exception as e:
                logger.error(f"LLM security analysis failed: {e}")
                # Fallback to pattern-based findings
                security_findings = []
                for result in pattern_results:
                    security_findings.append({
                        'category': result['pattern'],
                        'severity': 'medium',
                        'description': f"Pattern analysis found {result['files_affected']} files with potential {result['pattern']} issues",
                        'files_affected': result['files_affected']
                    })
                
                for filename in found_sensitive:
                    security_findings.append({
                        'category': 'sensitive_files',
                        'severity': 'high',
                        'description': f'Sensitive file found: {filename}',
                        'file_path': filename
                    })
            
            logger.info(f"Security analysis completed with {len(security_findings)} findings")
            return security_findings
            
        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            return [{"error": str(e)}]
    
    async def _analyze_performance(self, ctx: Context) -> List[Dict[str, Any]]:
        """Analyze performance using LLM agent"""
        import json
        
        try:
            # First do basic pattern matching
            performance_patterns = [
                ("inefficient_loops", "for.*for.*for"),
                ("blocking_calls", "sleep|time.sleep"),
                ("large_file_operations", "read_csv|open.*rb")
            ]
            
            pattern_results = []
            for pattern_name, pattern in performance_patterns:
                try:
                    search_result = json.loads(self.search_ops.search_codebase(
                        query=pattern,
                        file_types=['.py', '.js', '.ts'],
                        directory_path=None
                    ))
                    
                    if search_result.get('files_with_matches', 0) > 0:
                        pattern_results.append({
                            'pattern': pattern_name,
                            'files_affected': search_result.get('files_with_matches', 0),
                            'matches': search_result.get('results', [])[:2]
                        })
                except Exception as e:
                    logger.warning(f"Performance pattern search failed for {pattern_name}: {e}")
            
            # Use LLM agent for comprehensive performance analysis
            logger.info("Using LLM agent for comprehensive performance analysis...")
            
            performance_prompt = f"""
You are a performance analyst. Use the available tools to analyze this repository for performance bottlenecks.

Initial pattern matching found:
{json.dumps(pattern_results, indent=2) if pattern_results else "No performance anti-patterns detected"}

TASK: Perform a COMPREHENSIVE performance analysis. Execute these steps IN ORDER:

**STEP 1: Codebase Structure Analysis**
1. explore_directory("") - Map the repository structure
2. explore_directory("src") - Examine main source code
3. explore_directory("api") if exists - Check API performance

**STEP 2: Blocking Operations Analysis**
4. search_codebase with query="time.sleep" - Find blocking sleep calls
5. search_codebase with query="requests.get" - Find synchronous HTTP calls
6. search_codebase with query="requests.post" - Find more sync HTTP operations
7. search_codebase with query="sync" - Find synchronous operations
8. search_codebase with query="blocking" - Find blocking code patterns

**STEP 3: Database & I/O Performance**
9. search_codebase with query="query" - Find database queries
10. search_codebase with query="execute" - Find SQL execution
11. search_codebase with query="fetchall" - Find inefficient data fetching
12. search_codebase with query="open(" - Find file I/O operations
13. search_codebase with query="read()" - Find file reading patterns

**STEP 4: Loop & Algorithm Analysis**
14. search_codebase with query="for.*for" - Find nested loops
15. search_codebase with query="while" - Find while loops
16. search_codebase with query="recursion" - Find recursive patterns
17. read_file on any files with complex loops or algorithms

**STEP 5: Memory & Resource Analysis**
18. search_codebase with query="pandas" - Check for large data operations
19. search_codebase with query="numpy" - Check for array operations
20. search_codebase with query="json.load" - Find JSON parsing
21. search_codebase with query="cache" - Check caching strategies

**STEP 6: Async/Await Analysis**
22. search_codebase with query="async def" - Find async functions
23. search_codebase with query="await" - Check await usage
24. search_codebase with query="asyncio" - Check async patterns
25. read_file on key async files to analyze patterns

**REQUIREMENTS:**
- You MUST examine AT LEAST 8 different files
- You MUST perform ALL 25 steps above
- You MUST find at least 2 performance issues (if you don't, look harder)
- Focus on bottlenecks, inefficient algorithms, blocking I/O, memory usage

After completing ALL steps, provide findings in this EXACT JSON format:
[
  {{
    "category": "blocking_io|inefficient_loops|memory_leak|sync_operations|database_bottleneck|large_data_processing|inefficient_algorithms|missing_caching|resource_exhaustion",
    "impact": "high|medium|low",
    "description": "detailed description of the performance issue", 
    "file_path": "exact/path/to/file",
    "line_numbers": [1, 2, 3],
    "optimization_opportunity": "specific optimization recommendation",
    "estimated_improvement": "expected performance gain (e.g., '50% faster', '30% less memory')",
    "code_snippet": "relevant code showing the issue"
  }}
]

START NOW with step 1 - explore_directory("")"""

            try:
                response = await self.agent.achat(performance_prompt)
                
                performance_findings = []
                response_text = str(response)
                if response_text:
                    import re
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        performance_findings = json.loads(json_match.group())
                    else:
                        # Fallback to pattern results
                        for result in pattern_results:
                            performance_findings.append({
                                'category': result['pattern'],
                                'impact': 'medium',
                                'description': f"Found {result['files_affected']} files with potential {result['pattern']} issues",
                                'files_affected': result['files_affected'],
                                'optimization_opportunity': f"Review and optimize {result['pattern']} code patterns"
                            })
                            
            except Exception as e:
                logger.error(f"LLM performance analysis failed: {e}")
                # Fallback to pattern-based findings
                performance_findings = []
                for result in pattern_results:
                    performance_findings.append({
                        'category': result['pattern'],
                        'impact': 'medium',
                        'description': f"Pattern analysis found {result['files_affected']} files with potential {result['pattern']} issues",
                        'files_affected': result['files_affected']
                    })
            
            logger.info(f"Performance analysis completed with {len(performance_findings)} findings")
            return performance_findings
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return [{"error": str(e)}]
    
    async def _analyze_quality(self, ctx: Context) -> List[Dict[str, Any]]:
        """Analyze code quality using LLM agent"""
        import json
        
        try:
            # First do basic pattern matching
            quality_patterns = [
                ("todo_fixme", "TODO|FIXME|HACK"),
                ("documentation", "docstring|@param|@return"),
                ("testing", "test_|spec_|describe"),
                ("error_handling", "try:|except:|catch")
            ]
            
            pattern_results = []
            for pattern_name, pattern in quality_patterns:
                try:
                    search_result = json.loads(self.search_ops.search_codebase(
                        query=pattern,
                        file_types=['.py', '.js', '.ts'],
                        directory_path=None
                    ))
                    
                    files_with_pattern = search_result.get('files_with_matches', 0)
                    total_files = search_result.get('total_files_processed', 1)
                    coverage = files_with_pattern / total_files if total_files > 0 else 0
                    
                    pattern_results.append({
                        'pattern': pattern_name,
                        'files_with_pattern': files_with_pattern,
                        'total_files': total_files,
                        'coverage': coverage
                    })
                except Exception as e:
                    logger.warning(f"Quality pattern search failed for {pattern_name}: {e}")
            
            # Use LLM agent for comprehensive quality analysis
            logger.info("Using LLM agent for comprehensive quality analysis...")
            
            quality_prompt = f"""
You are a code quality analyst. Use the available tools to analyze this repository for code quality issues.

Initial pattern analysis found:
{json.dumps(pattern_results, indent=2) if pattern_results else "No initial patterns detected"}

TASK: Perform a THOROUGH code quality analysis. Execute these steps IN ORDER:

**STEP 1: Repository Structure Assessment**
1. explore_directory("") - Understand project structure
2. explore_directory("src") - Examine main source code
3. explore_directory("tests") if exists - Check test organization

**STEP 2: Code Organization Analysis**
4. search_codebase with query="class" - Find class definitions
5. search_codebase with query="def" - Find function definitions
6. search_codebase with query="import" - Analyze import patterns
7. read_file on main Python files to assess code organization

**STEP 3: Documentation Quality**
8. search_codebase with query="\"\"\"" - Find docstrings
9. search_codebase with query="#" - Find comments
10. search_codebase with query="TODO" - Find TODO items
11. search_codebase with query="FIXME" - Find FIXME items

**STEP 4: Error Handling & Robustness**
12. search_codebase with query="try:" - Find exception handling
13. search_codebase with query="except" - Check exception patterns
14. search_codebase with query="raise" - Find error raising
15. search_codebase with query="assert" - Find assertions

**STEP 5: Testing Quality**
16. search_codebase with query="test_" - Find test functions
17. search_codebase with query="pytest" - Check testing framework
18. search_codebase with query="mock" - Find mocking usage
19. read_file on test files to assess test quality

**STEP 6: Code Complexity & Maintainability**
20. search_codebase with query="if.*if.*if" - Find complex conditionals
21. search_codebase with query="lambda" - Find lambda usage
22. search_codebase with query="global" - Find global variables
23. read_file on complex files to assess maintainability

**REQUIREMENTS:**
- You MUST examine AT LEAST 6 different files
- You MUST perform ALL 23 steps above
- You MUST identify specific quality issues with actionable improvements
- Look for: poor naming, lack of documentation, complex functions, missing tests, poor error handling

After completing ALL steps, provide findings in this EXACT JSON format:
[
  {{
    "category": "documentation|testing|error_handling|code_complexity|naming_conventions|code_organization|maintainability|technical_debt",
    "severity": "high|medium|low",
    "description": "detailed description of the quality issue",
    "file_path": "exact/path/to/file",
    "line_numbers": [1, 2, 3],
    "quality_impact": "how this affects code maintainability",
    "improvement_suggestion": "specific steps to improve code quality",
    "code_example": "example of the issue or suggested improvement"
  }}
]

START NOW with step 1 - explore_directory("")"""            
            
            try:
                # Use the agent to analyze
                response = await self.agent.achat(quality_prompt)
                
                quality_findings = []
                response_text = str(response)
                if response_text:
                    import re
                    json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
                    if json_match:
                        quality_findings = json.loads(json_match.group())
                    else:
                        # Fallback to pattern results
                        for result in pattern_results:
                            quality_findings.append({
                                'category': result['pattern'],
                                'severity': 'medium',
                                'description': f"Found {result['files_with_pattern']} files with {result['pattern']} patterns",
                                'files_with_pattern': result['files_with_pattern'],
                                'coverage': result['coverage'],
                                'quality_impact': f"Code quality indicator for {result['pattern']}"
                            })
                            
            except Exception as e:
                logger.error(f"LLM quality analysis failed: {e}")
                # Fallback to pattern-based findings  
                quality_findings = []
                for result in pattern_results:
                    quality_findings.append({
                        'category': result['pattern'],
                        'severity': 'medium',
                        'description': f"Pattern analysis found {result['files_with_pattern']} files with {result['pattern']} indicators",
                        'files_with_pattern': result['files_with_pattern'],
                        'coverage': result['coverage']
                    })
            
            logger.info(f"Quality analysis completed with {len(quality_findings)} findings")
            return quality_findings
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return [{"error": str(e)}]
    
    # Helper methods
    def _parse_python_requirements(self, content: str) -> List[Dict[str, str]]:
        """Parse Python requirements.txt content"""
        dependencies = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                if '==' in line:
                    name, version = line.split('==', 1)
                    dependencies.append({'name': name.strip(), 'version': version.strip()})
                else:
                    dependencies.append({'name': line, 'version': 'latest'})
        return dependencies
    
    def _calculate_risk_level(self, findings: List[Dict[str, Any]]) -> str:
        """Calculate security risk level"""
        if not findings:
            return "low"
        
        high_risk = sum(1 for f in findings if f.get('severity') == 'high')
        if high_risk > 0:
            return "high"
        elif len(findings) > 2:
            return "medium"
        else:
            return "low"
    
    def _calculate_performance_score(self, findings: List[Dict[str, Any]]) -> int:
        """Calculate performance score"""
        return max(10, 100 - (len(findings) * 10))
    
    def _calculate_quality_metrics(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics"""
        metrics = {}
        for finding in findings:
            if 'coverage' in finding:
                metrics[f"{finding['category']}_coverage"] = finding['coverage']
        return metrics
    
    def _generate_overview(self, structure_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate repository overview"""
        analysis = structure_data.get('analysis', {})
        return {
            "file_type_distribution": analysis.get('files_by_type', {}),
            "total_size_bytes": analysis.get('total_size', 0),
            "structure_summary": analysis.get('structure_summary', '')
        }
    
    def _generate_recommendations(self, all_findings: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations"""
        recommendations = []
        
        security_findings = all_findings.get('security_findings', [])
        performance_findings = all_findings.get('performance_findings', [])
        
        # Security recommendations
        high_security = [f for f in security_findings if f.get('severity') == 'high']
        if high_security:
            recommendations.append({
                "category": "security",
                "priority": "high",
                "title": "Address High-Risk Security Issues",
                "description": f"Found {len(high_security)} high-risk security issues."
            })
        
        # Performance recommendations
        if len(performance_findings) > 3:
            recommendations.append({
                "category": "performance",
                "priority": "medium",
                "title": "Review Performance Issues",
                "description": f"Found {len(performance_findings)} potential performance bottlenecks."
            })
        
        return recommendations
    
    def _generate_summary(self, all_findings: Dict[str, Any], stages_completed: List[str]) -> str:
        """Generate executive summary"""
        total_issues = (len(all_findings.get('security_findings', [])) + 
                       len(all_findings.get('performance_findings', [])))
        
        return f"""
Comprehensive repository analysis completed.

Key Findings:
- Total Issues Found: {total_issues}
- Analysis Stages Completed: {len(stages_completed)}
- Security Findings: {len(all_findings.get('security_findings', []))}
- Performance Findings: {len(all_findings.get('performance_findings', []))}

The repository shows {'good' if total_issues < 5 else 'moderate' if total_issues < 15 else 'significant'} areas for improvement.
        """.strip()

# Factory function for easy integration
async def run_comprehensive_analysis(
    session_id: str,
    repo_path: str,
    query: str,
    focus_areas: List[str],
    context_manager: ContextManager,
    llm: Optional[LLM] = None,
    tools: Optional[List[BaseTool]] = None
) -> Dict[str, Any]:
    """
    Factory function to run comprehensive analysis using LlamaIndex AgentWorkflow
    """
    workflow = ComprehensiveAnalysisWorkflow(
        session_id=session_id,
        repo_path=repo_path,
        context_manager=context_manager,
        llm=llm,
        tools=tools
    )
    
    request = AnalysisRequest(
        query=query,
        focus_areas=focus_areas,
        session_id=session_id,
        repo_path=repo_path
    )
    
    # Run the workflow
    result = await workflow.run(request=request)
    return result