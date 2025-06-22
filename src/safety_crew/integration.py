"""
Integration module for Safety Crew with existing triage.flow infrastructure
"""

from typing import Dict, Any, List, Optional
import asyncio
import json
from dotenv import load_dotenv

# Ensure .env is loaded before importing anything else
load_dotenv()

# CrewAI and LlamaIndex imports
from crewai import Agent, Task, Crew, Process
from crewai_tools import LlamaIndexTool
from llama_index.core.tools import FunctionTool

# Import existing infrastructure
from src.llm_client import LLMClient
from src.agentic_rag import CompositeAgenticRetriever
from src.agent_tools.core import AgenticCodebaseExplorer
from src.agent_tools.context_manager import ContextManager
from src.agent_tools.llm_config import get_llm_instance
from src.session_manager import SessionManager
from src.cache.redis_cache_manager import EnhancedCacheManager
from src.config import settings

# Import safety tools - use standalone versions for now as they're more reliable
# TODO: Fix CrewAI tool integration later
from .tools.standalone_tools import (
    StandaloneSemgrepScanner as SemgrepScanner,
    StandaloneHallucinationDetector as HallucinationDetector,
    StandaloneSecurityPatternAnalyzer as SecurityPatternAnalyzer,
    StandaloneCodeQualityAnalyzer as CodeQualityAnalyzer
)
CREWAI_AVAILABLE = False  # Force standalone for now


class SafetyCrewIntegration:
    """Integrates Safety Crew with existing triage.flow infrastructure"""
    
    def __init__(
        self,
        session_manager: SessionManager,
        cache_manager: Optional[EnhancedCacheManager] = None
    ):
        self.session_manager = session_manager
        self.cache_manager = cache_manager
        self.llm_client = LLMClient()
        
    async def create_integrated_tools(
        self,
        session_id: str,
        repository_path: str
    ) -> Dict[str, List[LlamaIndexTool]]:
        """Create tools integrated with existing infrastructure"""
        
        # Get session and RAG system
        session = await self.session_manager.get_session(session_id)
        rag_system = session.rag_system if hasattr(session, 'rag_system') else None
        
        # Initialize context manager
        context_manager = ContextManager(repository_path=repository_path)
        
        # Create explorer for existing tools
        explorer = AgenticCodebaseExplorer(
            repository_path=repository_path,
            rag_retriever=rag_system,
            context_manager=context_manager
        )
        
        # Create safety-specific tools
        safety_tools = self._create_safety_tools(rag_system, context_manager)
        
        # Get existing tools from explorer
        existing_tools = self._wrap_explorer_tools(explorer)
        
        return {
            "security": safety_tools["security"] + existing_tools.get("search", []),
            "grounding": safety_tools["grounding"] + existing_tools.get("file", []),
            "quality": safety_tools["quality"] + existing_tools.get("code", []),
            "orchestrator": existing_tools.get("git", [])
        }
    
    def _create_safety_tools(
        self,
        rag_system: Optional[CompositeAgenticRetriever],
        context_manager: Optional[ContextManager]
    ) -> Dict[str, List[LlamaIndexTool]]:
        """Create safety-specific tools"""
        
        # Initialize tool instances
        semgrep = SemgrepScanner()
        hallucination = HallucinationDetector(
            rag_system=rag_system,
            context_manager=context_manager
        )
        security_analyzer = SecurityPatternAnalyzer()
        quality_analyzer = CodeQualityAnalyzer()
        
        # Create LlamaIndex tools
        tools = {
            "security": [
                self._create_crewai_tool(
                    semgrep._run,
                    "semgrep_scanner",
                    "Scan code for security vulnerabilities using Semgrep"
                ),
                self._create_crewai_tool(
                    security_analyzer._run,
                    "security_pattern_analyzer",
                    "Analyze code for security anti-patterns"
                )
            ],
            "grounding": [
                self._create_crewai_tool(
                    hallucination._run,
                    "hallucination_detector",
                    "Detect AI hallucinations by verifying against codebase"
                )
            ],
            "quality": [
                self._create_crewai_tool(
                    quality_analyzer._run,
                    "code_quality_analyzer",
                    "Analyze code quality metrics and best practices"
                )
            ]
        }
        
        return tools
    
    def _create_crewai_tool(self, fn, name: str, description: str) -> LlamaIndexTool:
        """Helper to create CrewAI tool from function"""
        llama_tool = FunctionTool.from_defaults(
            fn=fn,
            name=name,
            description=description
        )
        return LlamaIndexTool.from_tool(llama_tool)
    
    def _wrap_explorer_tools(self, explorer: AgenticCodebaseExplorer) -> Dict[str, List[LlamaIndexTool]]:
        """Wrap existing explorer tools for CrewAI"""
        
        tools = {
            "file": [],
            "search": [],
            "code": [],
            "git": []
        }
        
        # File operations
        if hasattr(explorer, 'file_ops'):
            tools["file"].extend([
                self._create_crewai_tool(
                    explorer.file_ops.read_file,
                    "read_file",
                    "Read file contents from the repository"
                ),
                self._create_crewai_tool(
                    explorer.file_ops.explore_directory,
                    "explore_directory",
                    "Explore directory structure"
                )
            ])
        
        # Search operations
        if hasattr(explorer, 'search_ops'):
            tools["search"].extend([
                self._create_crewai_tool(
                    explorer.search_ops.search_codebase,
                    "search_codebase",
                    "Search codebase for patterns"
                ),
                self._create_crewai_tool(
                    explorer.search_ops.semantic_content_search,
                    "semantic_search",
                    "Semantic search using RAG"
                )
            ])
        
        # Code generation operations
        if hasattr(explorer, 'code_gen_ops'):
            tools["code"].extend([
                self._create_crewai_tool(
                    explorer.code_gen_ops.analyze_code_structure,
                    "analyze_structure",
                    "Analyze code structure and patterns"
                )
            ])
        
        # Git operations
        if hasattr(explorer, 'git_ops'):
            tools["git"].extend([
                self._create_crewai_tool(
                    explorer.git_ops.git_blame_function,
                    "git_blame",
                    "Get git blame for functions"
                )
            ])
        
        return tools
    
    async def create_safety_crew(
        self,
        session_id: str,
        repository_path: str,
        analysis_depth: str = "standard"
    ) -> Crew:
        """Create a safety analysis crew with integrated tools"""
        
        # Get LLM based on analysis depth
        llm = self._get_llm_for_depth(analysis_depth)
        
        # Create integrated tools
        tools = await self.create_integrated_tools(session_id, repository_path)
        
        # Create agents
        agents = self._create_agents(llm, tools)
        
        # Create tasks (would be created based on specific analysis needs)
        tasks = []
        
        # Create crew
        return Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
            memory=True
        )
    
    def _get_llm_for_depth(self, depth: str):
        """Get appropriate LLM based on analysis depth"""
        if depth == "quick":
            # Use cheap model for quick analysis
            return get_llm_instance(default_model=settings.cheap_model)
        else:
            # Use default model for standard/deep analysis
            return get_llm_instance(default_model=settings.default_model)
    
    def _create_agents(self, llm, tools: Dict[str, List]) -> Dict[str, Agent]:
        """Create safety analysis agents"""
        
        return {
            "security": Agent(
                role="Security Specialist",
                goal="Identify all security vulnerabilities comprehensively",
                backstory="Expert security engineer with deep knowledge of vulnerabilities",
                tools=tools.get("security", []),
                llm=llm,
                verbose=True
            ),
            "grounding": Agent(
                role="Grounding Specialist",
                goal="Ensure all code is grounded in reality, no hallucinations",
                backstory="AI safety expert preventing hallucinations in generated code",
                tools=tools.get("grounding", []),
                llm=llm,
                verbose=True
            ),
            "quality": Agent(
                role="Quality Architect",
                goal="Ensure code quality and architectural excellence",
                backstory="Principal architect focused on maintainable, quality code",
                tools=tools.get("quality", []),
                llm=llm,
                verbose=True
            ),
            "orchestrator": Agent(
                role="Safety Orchestrator",
                goal="Synthesize findings into actionable insights",
                backstory="Chief Safety Officer coordinating comprehensive analysis",
                tools=tools.get("orchestrator", []),
                llm=llm,
                verbose=True,
                allow_delegation=True
            )
        }
    
    async def analyze_code(
        self,
        code: str,
        session_id: str,
        repository_path: str,
        file_path: Optional[str] = None,
        analysis_depth: str = "standard"
    ) -> Dict[str, Any]:
        """Run safety analysis on code"""
        
        # Check cache
        cache_key = None
        if self.cache_manager:
            cache_key = f"safety:{session_id}:{hash(code)}"
            cached = await self.cache_manager.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Create crew
        crew = await self.create_safety_crew(session_id, repository_path, analysis_depth)
        
        # Get agents list properly
        agents_list = list(crew.agents) if hasattr(crew, 'agents') else crew._agents
        
        # Create specific tasks for this code
        tasks = self._create_analysis_tasks(agents_list, code, file_path)
        crew.tasks = tasks
        
        # Run analysis
        try:
            results = crew.kickoff()
            
            # Structure results
            structured_results = {
                "status": "success",
                "code": code,
                "file_path": file_path,
                "analysis_depth": analysis_depth,
                "findings": self._parse_crew_results(results),
                "session_id": session_id
            }
            
            # Cache results
            if self.cache_manager and cache_key:
                await self.cache_manager.set(
                    cache_key,
                    json.dumps(structured_results),
                    ttl=300  # 5 minutes
                )
            
            return structured_results
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "code": code,
                "session_id": session_id
            }
    
    def _create_analysis_tasks(
        self,
        agents: List[Agent],
        code: str,
        file_path: Optional[str]
    ) -> List[Task]:
        """Create analysis tasks for the given code"""
        
        # Find agents by role
        security_agent = next(a for a in agents if "Security" in a.role)
        grounding_agent = next(a for a in agents if "Grounding" in a.role)
        quality_agent = next(a for a in agents if "Quality" in a.role)
        orchestrator = next(a for a in agents if "Orchestrator" in a.role)
        
        tasks = [
            Task(
                description=f"""
                Analyze this code for security vulnerabilities:
                
                File: {file_path or "inline code"}
                ```
                {code}
                ```
                
                Use all available security tools to find vulnerabilities.
                """,
                expected_output="List of security vulnerabilities with severity and remediation",
                agent=security_agent
            ),
            Task(
                description=f"""
                Check for AI hallucinations in this code:
                
                ```
                {code}
                ```
                
                Verify all APIs and imports exist in the actual codebase.
                """,
                expected_output="List of hallucinations with evidence",
                agent=grounding_agent
            ),
            Task(
                description=f"""
                Review code quality:
                
                ```
                {code}
                ```
                
                Analyze complexity, maintainability, and best practices.
                """,
                expected_output="Quality assessment with improvement suggestions",
                agent=quality_agent
            ),
            Task(
                description="Synthesize all findings into a prioritized action plan",
                expected_output="Executive summary with actionable recommendations",
                agent=orchestrator
            )
        ]
        
        # Set task dependencies properly
        if len(tasks) >= 4:
            # Synthesis task depends on the other three
            tasks[3].context = [tasks[0], tasks[1], tasks[2]]
        
        return tasks
    
    def _parse_crew_results(self, results: Any) -> Dict[str, Any]:
        """Parse crew results into structured format"""
        
        import json
        
        # Initialize structured result
        parsed = {
            "security_findings": [],
            "hallucinations": [],
            "quality_issues": [],
            "synthesis": "",
            "recommendations": []
        }
        
        # Handle different result formats
        if hasattr(results, 'tasks_output'):
            # CrewAI format with task outputs
            for task_output in results.tasks_output:
                if hasattr(task_output, 'raw_output'):
                    try:
                        # Try to parse JSON output
                        if task_output.raw_output.strip().startswith('{'):
                            data = json.loads(task_output.raw_output)
                            
                            # Extract findings based on content
                            if 'findings' in data:
                                for finding in data['findings']:
                                    if 'severity' in finding:
                                        parsed['security_findings'].append(finding)
                            
                            if 'hallucinations' in data:
                                parsed['hallucinations'].extend(data['hallucinations'])
                            
                            if 'issues' in data:
                                parsed['quality_issues'].extend(data['issues'])
                        else:
                            # Plain text output - add to synthesis
                            if task_output.raw_output:
                                parsed['synthesis'] += f"\n{task_output.raw_output}"
                    except:
                        # If parsing fails, treat as text
                        parsed['synthesis'] += f"\n{task_output.raw_output}"
        
        elif isinstance(results, str):
            # Simple string result
            parsed['synthesis'] = results
        
        elif isinstance(results, dict):
            # Already structured - merge
            for key in ['security_findings', 'hallucinations', 'quality_issues']:
                if key in results:
                    parsed[key].extend(results[key])
            if 'synthesis' in results:
                parsed['synthesis'] = results['synthesis']
        
        # Extract recommendations from synthesis
        if parsed['synthesis']:
            lines = parsed['synthesis'].strip().split('\n')
            for line in lines:
                if line.strip() and any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'fix']):
                    parsed['recommendations'].append({
                        "text": line.strip(),
                        "priority": "medium"
                    })
        
        return parsed
    
    async def analyze_code_with_crew(
        self,
        request,  # SafetyAnalysisRequest
        repository_path: str
    ):
        """Analyze code using the actual Safety Crew with proper request/response models"""
        
        # Import the crews and models
        from .crews.safety_crew import SafetyCrew
        from .models import SafetyAnalysisRequest, SafetyAnalysisResponse
        
        # Initialize the SafetyCrew with existing infrastructure
        session = await self.session_manager.get_session(request.session_id)
        
        # Get RAG system and context manager from session
        rag_system = None
        context_manager = None
        existing_tools = []
        
        if session:
            # Get RAG system from session if available
            if hasattr(session, 'rag_system'):
                rag_system = session.rag_system
            
            # Get context manager
            from pathlib import Path
            context_manager = ContextManager(request.session_id, Path(repository_path) if repository_path else Path("."))
            
            # Get existing tools from session if available
            if hasattr(session, 'agent_tools'):
                existing_tools = session.agent_tools
        
        # Create SafetyCrew instance
        safety_crew = SafetyCrew(
            existing_rag=rag_system,
            existing_tools=existing_tools,
            context_manager=context_manager,
            cache_manager=self.cache_manager,
            verbose=True,
            use_cheap_model_for_analysis=True
        )
        
        # Run the analysis
        response = await safety_crew.analyze(request)
        
        return response
    
    async def analyze_code_standalone(
        self,
        request,  # SafetyAnalysisRequest
        repository_path: str
    ):
        """Analyze code using standalone tools (fallback when CrewAI is not available)"""
        
        import time
        from datetime import datetime
        from .models import SafetyAnalysisResponse, SafetyMetrics
        
        start_time = time.time()
        
        # Check cache
        cache_key = None
        if self.cache_manager:
            cache_key = f"safety_standalone:{request.session_id}:{hash(request.code)}"
            try:
                cached = await self.cache_manager.get(cache_key)
                if cached:
                    return SafetyAnalysisResponse(**json.loads(cached))
            except Exception:
                pass
        
        # Get session for RAG system
        session = await self.session_manager.get_session(request.session_id)
        rag_system = None
        if session and hasattr(session, 'rag_system'):
            rag_system = session.rag_system
        
        # Initialize tools
        semgrep = SemgrepScanner()
        hallucination_detector = HallucinationDetector(
            rag_system=rag_system,
            context_manager=None
        )
        security_analyzer = SecurityPatternAnalyzer()
        quality_analyzer = CodeQualityAnalyzer()
        
        # Run analyses
        try:
            # Security analysis
            if hasattr(semgrep, 'scan'):
                security_results = semgrep.scan(request.code, request.language)
            else:
                security_results = json.loads(semgrep._run(request.code, request.language))
            
            # Hallucination detection
            if hasattr(hallucination_detector, 'detect'):
                hallucination_results = hallucination_detector.detect(request.code, request.session_id)
            else:
                hallucination_results = json.loads(hallucination_detector._run(request.code, request.session_id))
            
            # Security pattern analysis
            if hasattr(security_analyzer, 'analyze'):
                pattern_results = security_analyzer.analyze(request.code)
            else:
                pattern_results = json.loads(security_analyzer._run(request.code))
            
            # Quality analysis
            if hasattr(quality_analyzer, 'analyze'):
                quality_results = quality_analyzer.analyze(request.code)
            else:
                quality_results = json.loads(quality_analyzer._run(request.code))
            
            # Convert results to safety models format
            security_findings = []
            hallucination_flags = []
            quality_issues = []
            recommendations = []
            
            # Process security findings
            if security_results.get("status") == "success":
                for finding in security_results.get("findings", []):
                    if "error" not in finding:
                        # Map Semgrep severity to our enum values
                        severity_map = {
                            "ERROR": "high",
                            "WARNING": "medium", 
                            "INFO": "info",
                            "HIGH": "high",
                            "MEDIUM": "medium",
                            "LOW": "low"
                        }
                        raw_severity = finding.get("severity", "medium")
                        mapped_severity = severity_map.get(raw_severity.upper(), "medium")
                        
                        security_findings.append({
                            "id": f"sec-{len(security_findings)}",
                            "type": "other",
                            "severity": mapped_severity,
                            "title": finding.get("rule_id", "Security Issue"),
                            "description": finding.get("message", "Security vulnerability detected"),
                            "line_number": finding.get("line", 0),
                            "remediation": finding.get("fix", "Review and fix the security issue"),
                            "confidence": 0.8
                        })
            
            # Process pattern findings
            if pattern_results.get("status") == "success":
                for finding in pattern_results.get("findings", []):
                    # Map severity for pattern findings too
                    raw_severity = finding.get("severity", "medium")
                    severity_map = {"ERROR": "high", "WARNING": "medium", "INFO": "info", "HIGH": "high", "MEDIUM": "medium", "LOW": "low"}
                    mapped_severity = severity_map.get(raw_severity.upper(), "medium")
                    
                    security_findings.append({
                        "id": f"pat-{len(security_findings)}",
                        "type": finding.get("type", "other"),
                        "severity": mapped_severity,
                        "title": f"Security Pattern: {finding.get('type', 'Unknown')}",
                        "description": f"Found {finding.get('type')} pattern at line {finding.get('line')}",
                        "line_number": finding.get("line", 0),
                        "remediation": f"Review and fix the {finding.get('type')} vulnerability",
                        "confidence": 0.9
                    })
            
            # Process hallucination findings
            if hallucination_results.get("status") == "success":
                for hallucination in hallucination_results.get("hallucinations", []):
                    hallucination_flags.append({
                        "id": f"hal-{len(hallucination_flags)}",
                        "type": hallucination.get("type", "other"),
                        "severity": hallucination.get("severity", "high"),
                        "description": hallucination.get("reason", "Potential hallucination detected"),
                        "hallucinated_code": hallucination.get("code", ""),
                        "confidence": 0.85
                    })
            
            # Process quality findings
            if quality_results.get("status") == "success":
                for issue in quality_results.get("issues", []):
                    quality_issues.append({
                        "id": f"qual-{len(quality_issues)}",
                        "type": issue.get("type", "other"),
                        "severity": issue.get("severity", "low"),
                        "title": issue.get("type", "Quality Issue"),
                        "description": issue.get("message", "Code quality issue detected"),
                        "improvement_suggestion": "Review and improve the code quality",
                        "impact": "May affect maintainability"
                    })
            
            # Generate recommendations
            if security_findings:
                recommendations.append({
                    "agent_name": "SecuritySpecialist",
                    "recommendation": f"Found {len(security_findings)} security issues that need attention",
                    "priority": "high",
                    "rationale": "Security vulnerabilities pose risks to the application",
                    "action_items": ["Review security findings", "Apply recommended fixes"]
                })
            
            if hallucination_flags:
                recommendations.append({
                    "agent_name": "GroundingSpecialist", 
                    "recommendation": f"Detected {len(hallucination_flags)} potential AI hallucinations",
                    "priority": "high",
                    "rationale": "Hallucinated code may not work as expected",
                    "action_items": ["Verify all APIs and imports", "Test the code thoroughly"]
                })
            
            if quality_issues:
                recommendations.append({
                    "agent_name": "QualityArchitect",
                    "recommendation": f"Found {len(quality_issues)} code quality issues",
                    "priority": "medium",
                    "rationale": "Quality issues affect maintainability",
                    "action_items": ["Improve code structure", "Add documentation"]
                })
            
            # Calculate metrics
            total_findings = len(security_findings) + len(hallucination_flags) + len(quality_issues)
            critical_count = sum(1 for f in security_findings if f.get("severity") == "critical")
            high_count = sum(1 for f in security_findings + hallucination_flags if f.get("severity") == "high")
            medium_count = sum(1 for f in security_findings + quality_issues if f.get("severity") == "medium")
            low_count = sum(1 for f in quality_issues if f.get("severity") == "low")
            
            # Calculate scores (10 = perfect, 0 = worst)
            security_score = max(0, 10 - (critical_count * 3 + high_count * 2 + medium_count * 1))
            grounding_score = max(0, 10 - len(hallucination_flags) * 2)
            quality_score = max(0, 10 - len(quality_issues) * 1.5)
            overall_score = (security_score + grounding_score + quality_score) / 3
            
            metrics = SafetyMetrics(
                overall_risk_score=overall_score,
                security_score=security_score,
                grounding_score=grounding_score,
                quality_score=quality_score,
                total_findings=total_findings,
                critical_findings=critical_count,
                high_findings=high_count,
                medium_findings=medium_count,
                low_findings=low_count,
                auto_fixable_count=0  # Standalone tools don't provide auto-fixes yet
            )
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Create response
            response = SafetyAnalysisResponse(
                request_id=f"standalone-{request.session_id}-{hash(request.code)}",
                session_id=request.session_id,
                timestamp=datetime.utcnow(),
                analysis_duration_ms=duration_ms,
                security_findings=security_findings,
                hallucination_flags=hallucination_flags,
                quality_issues=quality_issues,
                agent_recommendations=recommendations,
                auto_fix_suggestions=[],
                safety_metrics=metrics,
                crew_type="standalone_analysis",
                agents_involved=["security_analyzer", "hallucination_detector", "quality_analyzer"]
            )
            
            # Cache results
            if self.cache_manager and cache_key:
                try:
                    await self.cache_manager.set(
                        cache_key,
                        response.model_dump_json(),
                        ttl=300  # 5 minutes
                    )
                except Exception:
                    pass
            
            return response
            
        except Exception as e:
            # Return error response
            duration_ms = int((time.time() - start_time) * 1000)
            
            from .models import SafetyMetrics
            
            return SafetyAnalysisResponse(
                request_id=f"error-{request.session_id}",
                session_id=request.session_id,
                timestamp=datetime.utcnow(),
                analysis_duration_ms=duration_ms,
                security_findings=[],
                hallucination_flags=[],
                quality_issues=[],
                agent_recommendations=[{
                    "agent_name": "SystemError",
                    "recommendation": f"Analysis failed: {str(e)}",
                    "priority": "high",
                    "rationale": "System error prevented analysis",
                    "action_items": ["Check system configuration", "Try again"]
                }],
                auto_fix_suggestions=[],
                safety_metrics=SafetyMetrics(
                    overall_risk_score=0.0,
                    security_score=0.0,
                    grounding_score=0.0,
                    quality_score=0.0,
                    total_findings=0,
                    critical_findings=0,
                    high_findings=0,
                    medium_findings=0,
                    low_findings=0,
                    auto_fixable_count=0
                ),
                crew_type="standalone_analysis",
                agents_involved=["error"]
            )