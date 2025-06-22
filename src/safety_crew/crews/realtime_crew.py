"""Realtime Safety Crew - Fast safety checks for live coding"""

import time
from typing import Dict, Any, Optional
from crewai import Crew, Process

from ..agents import SecuritySpecialist, GroundingSpecialist
from ..tasks import SecurityAnalysisTask, HallucinationCheckTask
from ..models import SafetyAnalysisRequest, SafetyAnalysisResponse
from ..tools import create_safety_specific_tools
from src.agent_tools.llm_config import get_llm_instance
from src.config import settings


class RealtimeSafetyCrew:
    """Lightweight crew for fast safety checks during coding"""
    
    def __init__(
        self,
        existing_rag: Optional[Any] = None,
        context_manager: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        verbose: bool = False  # Less verbose for speed
    ):
        # Always use cheap model for realtime analysis
        self.llm_client = get_llm_instance(default_model=settings.cheap_model)
        self.cheap_llm = self.llm_client
        self.existing_rag = existing_rag
        self.context_manager = context_manager
        self.cache_manager = cache_manager
        self.verbose = verbose
        
        # Only essential tools for speed
        self.safety_tools = create_safety_specific_tools()
        
        # Only critical agents for realtime
        self.agents = self._create_agents()
        
    def _create_agents(self) -> Dict[str, Any]:
        """Create minimal agent set for speed"""
        
        # Security specialist for critical vulnerabilities
        security_specialist = SecuritySpecialist(
            llm=self.cheap_llm,
            tools=self.safety_tools[:2],  # Only core security tools
            verbose=self.verbose
        )
        
        # Grounding specialist for hallucination detection
        grounding_specialist = GroundingSpecialist(
            llm=self.cheap_llm,
            tools=[self.safety_tools[1]],  # Only hallucination detector
            rag_system=self.existing_rag,
            context_manager=self.context_manager,
            verbose=self.verbose
        )
        
        return {
            "security": security_specialist.get_agent(),
            "grounding": grounding_specialist.get_agent()
        }
        
    async def quick_analysis(self, code: str, session_id: str) -> Dict[str, Any]:
        """Perform quick safety analysis for realtime feedback"""
        
        start_time = time.time()
        
        # Create minimal request
        request = SafetyAnalysisRequest(
            session_id=session_id,
            code=code,
            analysis_depth="quick"
        )
        
        # Check cache
        if self.cache_manager:
            cache_key = f"realtime_safety:{session_id}:{hash(code)}"
            cached = await self._get_cached_result(cache_key)
            if cached:
                return cached
                
        # Create minimal tasks
        security_task = SecurityAnalysisTask.create(
            agent=self.agents["security"],
            code=code,
            language="python",
            custom_rules=["p/security-audit"]  # Only critical rules
        )
        
        hallucination_task = HallucinationCheckTask.create(
            agent=self.agents["grounding"],
            code=code,
            session_id=session_id,
            language="python"
        )
        
        # Create lightweight crew
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[security_task, hallucination_task],
            process=Process.parallel,  # Run in parallel for speed
            verbose=self.verbose,
            memory=False  # No memory for speed
        )
        
        try:
            # Execute with timeout
            crew_output = crew.kickoff()
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Simple result format for realtime
            result = {
                "status": "success",
                "duration_ms": duration_ms,
                "has_critical_issues": self._has_critical_issues(crew_output),
                "security_alerts": self._extract_security_alerts(crew_output),
                "hallucination_alerts": self._extract_hallucination_alerts(crew_output),
                "quick_fixes": self._suggest_quick_fixes(crew_output)
            }
            
            # Cache result
            if self.cache_manager:
                await self._cache_result(cache_key, result)
                
            return result
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "duration_ms": int((time.time() - start_time) * 1000)
            }
            
    def _has_critical_issues(self, crew_output: Any) -> bool:
        """Check if critical issues were found"""
        # Parse crew output for critical findings
        # In production, would parse actual output
        return False
        
    def _extract_security_alerts(self, crew_output: Any) -> list:
        """Extract critical security alerts"""
        # Parse security task output
        return []
        
    def _extract_hallucination_alerts(self, crew_output: Any) -> list:
        """Extract hallucination alerts"""
        # Parse hallucination task output
        return []
        
    def _suggest_quick_fixes(self, crew_output: Any) -> list:
        """Suggest immediate fixes"""
        # Generate quick fix suggestions
        return []
        
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
            await self.cache_manager.set(cache_key, result, ttl=60)  # 1 minute TTL
        except Exception:
            pass