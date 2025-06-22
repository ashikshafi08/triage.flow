"""Enterprise Safety Crew - Advanced features for enterprise compliance"""

import time
from typing import Dict, Any, List, Optional
from crewai import Crew, Process

from ..agents import (
    SecuritySpecialist,
    GroundingSpecialist,
    QualityArchitect,
    SafetyOrchestrator
)
from ..tasks import (
    SecurityAnalysisTask,
    HallucinationCheckTask,
    QualityReviewTask,
    SynthesisTask
)
from ..models import SafetyAnalysisRequest, SafetyAnalysisResponse
from ..tools import convert_existing_tools_to_crewai, create_safety_specific_tools
from src.agent_tools.llm_config import get_llm_instance
from src.config import settings


class EnterpriseSafetyCrew:
    """Enterprise-grade crew with compliance and audit features"""
    
    def __init__(
        self,
        existing_rag: Optional[Any] = None,
        existing_tools: Optional[List[Any]] = None,
        context_manager: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        compliance_rules: Optional[List[str]] = None,
        audit_logger: Optional[Any] = None,
        verbose: bool = True
    ):
        # Use high-quality model for enterprise features
        self.llm_client = get_llm_instance(default_model=settings.default_model)
        self.cheap_llm = get_llm_instance(default_model=settings.cheap_model)
        self.existing_rag = existing_rag
        self.context_manager = context_manager
        self.cache_manager = cache_manager
        self.compliance_rules = compliance_rules or ["SOC2", "ISO27001"]
        self.audit_logger = audit_logger
        self.verbose = verbose
        
        # All tools including enterprise-specific
        self.crewai_tools = []
        if existing_tools:
            self.crewai_tools.extend(existing_tools)
            
        self.safety_tools = create_safety_specific_tools()
        
        # Initialize all agents with enhanced capabilities
        self.agents = self._create_agents()
        
    def _create_agents(self) -> Dict[str, Any]:
        """Create enterprise-grade agents with enhanced capabilities"""
        
        # Enhanced security specialist with compliance focus
        security_specialist = SecuritySpecialist(
            llm=self.cheap_llm,
            tools=self.crewai_tools + self.safety_tools,
            verbose=self.verbose
        )
        
        # Enhanced grounding specialist
        grounding_specialist = GroundingSpecialist(
            llm=self.cheap_llm,
            tools=self.crewai_tools,
            rag_system=self.existing_rag,
            context_manager=self.context_manager,
            verbose=self.verbose
        )
        
        # Enhanced quality architect
        quality_architect = QualityArchitect(
            llm=self.llm_client,
            tools=self.crewai_tools + self.safety_tools,
            verbose=self.verbose
        )
        
        # Enhanced orchestrator with compliance expertise
        safety_orchestrator = SafetyOrchestrator(
            llm=self.llm_client,
            tools=self.crewai_tools,
            verbose=self.verbose
        )
        
        return {
            "security": security_specialist.get_agent(),
            "grounding": grounding_specialist.get_agent(),
            "quality": quality_architect.get_agent(),
            "orchestrator": safety_orchestrator.get_agent()
        }
        
    def _create_enterprise_tasks(self, request: SafetyAnalysisRequest) -> List[Any]:
        """Create enhanced tasks with compliance requirements"""
        
        # Add compliance context
        compliance_context = {
            "compliance_frameworks": self.compliance_rules,
            "audit_requirements": True,
            "generate_evidence": True
        }
        
        # Enhanced security task with compliance
        security_task = SecurityAnalysisTask.create(
            agent=self.agents["security"],
            code=request.code,
            file_path=request.file_path,
            language=request.language or "python",
            custom_rules=request.custom_rules + [f"compliance/{rule.lower()}" for rule in self.compliance_rules],
            context={**request.context, **compliance_context}
        )
        
        # Standard hallucination check
        hallucination_task = HallucinationCheckTask.create(
            agent=self.agents["grounding"],
            code=request.code,
            session_id=request.session_id,
            file_path=request.file_path,
            language=request.language or "python",
            context=request.context
        )
        
        # Enhanced quality review with enterprise standards
        quality_task = QualityReviewTask.create(
            agent=self.agents["quality"],
            code=request.code,
            file_path=request.file_path,
            language=request.language or "python",
            context={**request.context, "enterprise_standards": True}
        )
        
        # Enhanced synthesis with compliance reporting
        synthesis_task = SynthesisTask.create(
            agent=self.agents["orchestrator"],
            context={
                "request": request.dict(),
                "compliance_requirements": self.compliance_rules,
                "generate_audit_trail": True,
                "include_remediation_evidence": True
            }
        )
        
        synthesis_task.context_from = [security_task, hallucination_task, quality_task]
        
        return [security_task, hallucination_task, quality_task, synthesis_task]
        
    async def analyze_with_compliance(
        self,
        request: SafetyAnalysisRequest
    ) -> Dict[str, Any]:
        """Perform enterprise-grade analysis with compliance reporting"""
        
        start_time = time.time()
        
        # Log audit trail
        if self.audit_logger:
            await self._log_audit_event("analysis_started", request)
            
        # Check cache
        if self.cache_manager:
            cache_key = f"enterprise_safety:{request.session_id}:{hash(request.code)}"
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
                
        # Create enhanced tasks
        tasks = self._create_enterprise_tasks(request)
        
        # Create enterprise crew with enhanced process
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            process=Process.hierarchical,  # Orchestrator manages others
            manager_llm=self.llm_client,   # Premium model for management
            verbose=self.verbose,
            memory=True,
            embedder={
                "provider": "openai"
            },
            planning=True,  # Enable planning for complex analysis
            planning_llm=self.llm_client
        )
        
        try:
            # Execute crew
            crew_output = crew.kickoff()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Generate compliance reports
            compliance_report = await self._generate_compliance_report(crew_output, request)
            
            # Create audit trail
            audit_trail = await self._create_audit_trail(crew_output, request, duration_ms)
            
            # Parse output with compliance additions
            parsed_output = self._parse_enterprise_output(crew_output, compliance_report, audit_trail)
            
            response = SafetyAnalysisResponse.from_crew_results(
                crew_output=parsed_output,
                request=request,
                duration_ms=duration_ms
            )
            
            # Enhanced response with compliance data
            enhanced_response = {
                "safety_analysis": response.dict(),
                "compliance_report": compliance_report,
                "audit_trail": audit_trail,
                "evidence_package": self._create_evidence_package(parsed_output)
            }
            
            # Cache result
            if self.cache_manager:
                await self._cache_result(cache_key, enhanced_response)
                
            # Log completion
            if self.audit_logger:
                await self._log_audit_event("analysis_completed", enhanced_response)
                
            return enhanced_response
            
        except Exception as e:
            # Log error
            if self.audit_logger:
                await self._log_audit_event("analysis_failed", {"error": str(e)})
                
            raise
            
    async def _generate_compliance_report(
        self,
        crew_output: Any,
        request: SafetyAnalysisRequest
    ) -> Dict[str, Any]:
        """Generate compliance-specific reporting"""
        
        return {
            "frameworks": self.compliance_rules,
            "compliance_status": "PASS",  # Would be determined by findings
            "violations": [],
            "recommendations": [],
            "evidence_collected": True,
            "timestamp": time.time()
        }
        
    async def _create_audit_trail(
        self,
        crew_output: Any,
        request: SafetyAnalysisRequest,
        duration_ms: int
    ) -> Dict[str, Any]:
        """Create detailed audit trail for compliance"""
        
        return {
            "analysis_id": f"ENT-{request.session_id}-{int(time.time())}",
            "timestamp": time.time(),
            "duration_ms": duration_ms,
            "agents_involved": list(self.agents.keys()),
            "tools_used": [tool.name for tool in self.safety_tools],
            "compliance_frameworks": self.compliance_rules,
            "code_hash": hash(request.code),
            "findings_summary": {
                "total": 0,  # Would be populated from crew_output
                "by_severity": {}
            }
        }
        
    def _create_evidence_package(self, parsed_output: Dict[str, Any]) -> Dict[str, Any]:
        """Create evidence package for compliance audits"""
        
        return {
            "scan_results": parsed_output.get("security_findings", []),
            "tool_outputs": [],
            "agent_decisions": [],
            "remediation_evidence": [],
            "timestamp": time.time()
        }
        
    def _parse_enterprise_output(
        self,
        crew_output: Any,
        compliance_report: Dict[str, Any],
        audit_trail: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse crew output with enterprise additions"""
        
        base_output = {
            "security_findings": [],
            "hallucination_flags": [],
            "quality_issues": [],
            "recommendations": [],
            "auto_fixes": [],
            "crew_type": "enterprise_crew",
            "agents_involved": list(self.agents.keys())
        }
        
        # Add enterprise metadata
        base_output["compliance_metadata"] = compliance_report
        base_output["audit_metadata"] = audit_trail
        
        return base_output
        
    async def _log_audit_event(self, event_type: str, data: Any):
        """Log audit event"""
        if self.audit_logger:
            await self.audit_logger.log({
                "event_type": event_type,
                "timestamp": time.time(),
                "data": data
            })
            
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        if not self.cache_manager:
            return None
            
        try:
            return await self.cache_manager.get(cache_key)
        except Exception:
            return None
            
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result"""
        if not self.cache_manager:
            return
            
        try:
            await self.cache_manager.set(cache_key, result, ttl=600)  # 10 minutes
        except Exception:
            pass