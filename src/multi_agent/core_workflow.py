"""
Core Workflow for Multi-Agent Codebase Intelligence

This workflow orchestrates the research, analysis, and implementation agents
to provide comprehensive software development assistance.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    Context,
    step
)

from .events import (
    ResearchPlanEvent,
    ResearchResultEvent,
    ImplementationPlanEvent,
    ValidationEvent,
    StreamingEvent
)
from .agents import ResearchPlannerAgent, CodeAnalysisAgent, ImplementationAgent
from .validators import CodeValidator, SafetyGate
from ..agentic_tools import AgenticCodebaseExplorer

logger = logging.getLogger(__name__)


class CodebaseIntelligenceWorkflow(Workflow):
    """
    Main workflow that coordinates multi-agent research and implementation
    for intelligent software development assistance.
    """
    
    def __init__(
        self,
        session_id: str,
        repo_path: str,
        issue_rag_system=None,
        timeout: int = 300,
        **kwargs
    ):
        super().__init__(timeout=timeout, **kwargs)
        
        self.session_id = session_id
        self.repo_path = repo_path
        
        # Initialize the core codebase explorer
        self.explorer = AgenticCodebaseExplorer(
            session_id=session_id,
            repo_path=repo_path,
            issue_rag_system=issue_rag_system
        )
        
        # Initialize agents
        self.research_planner = ResearchPlannerAgent(self.explorer)
        self.code_analyst = CodeAnalysisAgent(self.explorer)
        self.implementation_agent = ImplementationAgent(self.explorer)
        
        # Initialize validators
        self.code_validator = CodeValidator()
        self.safety_gate = SafetyGate()
        
        # Performance tracking
        self.start_time = None
        self.stage_times = {}
    
    @step
    async def plan_research(self, ctx: Context, ev: StartEvent) -> ResearchPlanEvent:
        """
        First step: Plan the research approach based on the user's query
        """
        self.start_time = time.time()
        stage_start = time.time()
        
        try:
            query = ev.get("query")
            context = ev.get("context", {})
            
            # Store query in context for later steps
            await ctx.set("original_query", query)
            await ctx.set("start_time", self.start_time)
            
            # Create research plan
            research_plan = await self.research_planner.create_research_plan(query, context)
            
            self.stage_times["planning"] = time.time() - stage_start
            
            return ResearchPlanEvent(
                query=query,
                research_tasks=research_plan.get("research_tasks", []),
                priority=research_plan.get("priority", "normal"),
                estimated_complexity=research_plan.get("estimated_complexity", 5)
            )
            
        except Exception as e:
            logger.error(f"Research planning failed: {e}")
            # Create fallback plan
            return ResearchPlanEvent(
                query=query,
                research_tasks=[{
                    "task_id": "fallback_analysis",
                    "task_type": "code_analysis",
                    "description": f"Basic analysis for: {query}",
                    "agent_type": "CodeAnalysisAgent",
                    "parameters": {"query": query}
                }],
                priority="normal",
                estimated_complexity=3
            )
    
    @step(num_workers=3)  # Parallel execution of research tasks
    async def execute_research(self, ctx: Context, ev: ResearchPlanEvent) -> ResearchResultEvent:
        """
        Second step: Execute research tasks in parallel using multiple workers
        """
        stage_start = time.time()
        
        try:
            # Execute research tasks
            research_results = []
            
            # Create tasks for parallel execution
            tasks = []
            for research_task in ev.research_tasks:
                task_coro = self.code_analyst.execute_research_task(research_task)
                tasks.append(task_coro)
            
            # Execute tasks
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Research task {i} failed: {result}")
                        # Create error result
                        result = {
                            "task_id": ev.research_tasks[i].get("task_id", f"task_{i}"),
                            "analysis_type": "error",
                            "error": str(result),
                            "confidence": 0.0
                        }
                    
                    research_results.append(result)
            
            self.stage_times["research"] = time.time() - stage_start
            
            # Calculate overall confidence
            confidences = [r.get("confidence", 0.0) for r in research_results if isinstance(r, dict)]
            overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Organize results by type
            organized_results = self._organize_research_results(research_results)
            
            return ResearchResultEvent(
                original_query=ev.query,
                code_analysis=organized_results.get("code_analysis", {}),
                issue_analysis=organized_results.get("issue_analysis", {}),
                pattern_analysis=organized_results.get("pattern_detection", {}),
                historical_context=organized_results.get("historical_analysis", {}),
                confidence_score=overall_confidence
            )
            
        except Exception as e:
            logger.error(f"Research execution failed: {e}")
            return ResearchResultEvent(
                original_query=ev.query,
                code_analysis={"error": str(e)},
                issue_analysis={},
                pattern_analysis={},
                historical_context={},
                confidence_score=0.0
            )
    
    @step
    async def create_implementation(self, ctx: Context, ev: ResearchResultEvent) -> ImplementationPlanEvent:
        """
        Third step: Create implementation plan based on research results
        """
        stage_start = time.time()
        
        try:
            # Prepare research summary for implementation agent
            research_summary = {
                "research_results": [
                    {"analysis_type": "code_analysis", "results": ev.code_analysis, "confidence": ev.confidence_score},
                    {"analysis_type": "issue_analysis", "results": ev.issue_analysis, "confidence": ev.confidence_score},
                    {"analysis_type": "pattern_detection", "results": ev.pattern_analysis, "confidence": ev.confidence_score},
                    {"analysis_type": "historical_analysis", "results": ev.historical_context, "confidence": ev.confidence_score}
                ]
            }
            
            # Create implementation plan
            implementation_plan = await self.implementation_agent.create_implementation_plan(
                query=ev.original_query,
                research_results=research_summary
            )
            
            self.stage_times["implementation"] = time.time() - stage_start
            
            # Extract code components from implementation plan
            code_components = implementation_plan.get("code_components", [])
            code_examples = code_components if code_components else []
            
            return ImplementationPlanEvent(
                original_query=ev.original_query,
                research_summary=research_summary,
                implementation_strategy=implementation_plan.get("implementation_strategy", {}),
                code_examples=code_examples,
                validation_steps=implementation_plan.get("validation_steps", []),
                risk_assessment=implementation_plan.get("risk_assessment", {})
            )
            
        except Exception as e:
            logger.error(f"Implementation planning failed: {e}")
            return ImplementationPlanEvent(
                original_query=ev.original_query,
                research_summary={"error": str(e)},
                implementation_strategy={"error": "Implementation planning failed"},
                code_examples=[],
                validation_steps=["Manual implementation required"],
                risk_assessment={"risk_level": "HIGH", "reason": "Planning failed"}
            )
    
    @step
    async def validate_implementation(self, ctx: Context, ev: ImplementationPlanEvent) -> ValidationEvent:
        """
        Fourth step: Validate the implementation plan for safety and quality
        """
        stage_start = time.time()
        
        try:
            validation_results = {}
            
            # Validate generated code
            if ev.code_examples:
                first_example = ev.code_examples[0] if ev.code_examples else {}
                code_validation = self.code_validator.validate_code(
                    code=first_example.get("code", ""),
                    language=first_example.get("language", "python")
                )
                validation_results["code_validation"] = code_validation
            
            # Assess safety risks
            first_example = ev.code_examples[0] if ev.code_examples else {}
            implementation_plan_dict = {
                "original_query": ev.original_query,
                "main_implementation": first_example.get("code", ""),
                "implementation_steps": []  # Simplified for validation
            }
            
            risk_assessment = self.safety_gate.assess_implementation_risk(implementation_plan_dict)
            validation_results["risk_assessment"] = risk_assessment
            
            # Determine if implementation is approved
            code_valid = validation_results.get("code_validation", {}).get("valid", False)
            risk_acceptable = not risk_assessment.get("requires_approval", True)
            
            approved = code_valid and risk_acceptable
            
            # Collect specific feedback and required changes
            feedback = []
            required_changes = []
            
            if not code_valid:
                code_validation_result = validation_results.get("code_validation", {})
                code_errors = code_validation_result.get("errors", [])
                code_warnings = code_validation_result.get("warnings", [])
                code_suggestions = code_validation_result.get("suggestions", [])
                
                # Add specific code validation feedback
                for error in code_errors:
                    feedback.append(f"❌ Syntax Error: {error}")
                    required_changes.append(f"Fix syntax error: {error}")
                
                for warning in code_warnings:
                    feedback.append(f"⚠️ Security Warning: {warning}")
                    required_changes.append(f"Review security concern: {warning}")
                
                for suggestion in code_suggestions:
                    feedback.append(f"💡 Suggestion: {suggestion}")
            
            # Add risk-specific feedback
            if risk_assessment.get("requires_approval", False):
                risk_level = risk_assessment.get("risk_level", "UNKNOWN")
                approval_reason = risk_assessment.get("approval_reason", "Unknown risk")
                feedback.append(f"🔒 High Risk ({risk_level}): {approval_reason}")
                required_changes.append(f"Address {risk_level.lower()} risk factors")
                
                # Add specific mitigation suggestions
                mitigations = risk_assessment.get("mitigation_suggestions", {})
                for risk_type, suggestions in mitigations.items():
                    if suggestions:
                        feedback.append(f"🛡️ {risk_type.replace('_', ' ').title()} Mitigations: {', '.join(suggestions[:2])}")
            
            # Add performance feedback if validation took time
            validation_time = time.time() - stage_start
            if validation_time > 1.0:
                feedback.append(f"⏱️ Validation completed in {validation_time:.2f}s")
            
            # Add positive feedback when things go well
            if code_valid and not risk_assessment.get("requires_approval", False):
                feedback.append("✅ Code validation passed")
                feedback.append("✅ Risk assessment approved")
            
            # Ensure we have some feedback even if everything passes
            if not feedback:
                feedback.append("🔍 Implementation completed but requires manual review")
                required_changes.append("Perform manual testing and integration")
            
            self.stage_times["validation"] = time.time() - stage_start
            
            return ValidationEvent(
                implementation_plan=implementation_plan_dict,
                validation_results=validation_results,
                approved=approved,
                feedback=feedback,
                required_changes=required_changes
            )
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return ValidationEvent(
                implementation_plan={},
                validation_results={"error": str(e)},
                approved=False,
                feedback=[f"Validation error: {str(e)}"],
                required_changes=["Manual validation required"]
            )
    
    @step
    async def finalize_result(self, ctx: Context, ev: ValidationEvent) -> StopEvent:
        """
        Final step: Package and return the complete result
        """
        try:
            # Get original query and timing information
            original_query = await ctx.get("original_query", "")
            start_time = await ctx.get("start_time", time.time())
            total_time = time.time() - start_time
            
            # Create comprehensive result
            result = {
                "query": original_query,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "total_execution_time": total_time,
                "stage_times": self.stage_times,
                
                # Implementation details
                "implementation_plan": ev.implementation_plan,
                "validation_results": ev.validation_results,
                "approved": ev.approved,
                "feedback": ev.feedback,
                "required_changes": ev.required_changes,
                
                # Summary metrics
                "summary": {
                    "status": "approved" if ev.approved else "requires_review",
                    "risk_level": ev.validation_results.get("risk_assessment", {}).get("risk_level", "UNKNOWN"),
                    "code_valid": ev.validation_results.get("code_validation", {}).get("valid", False),
                    "total_feedback_items": len(ev.feedback),
                    "performance": {
                        "total_time_seconds": total_time,
                        "stages": self.stage_times
                    }
                },
                
                # Next steps
                "next_steps": self._generate_next_steps(ev),
                
                # Workflow metadata
                "workflow_version": "1.0.0",
                "agents_used": ["ResearchPlanner", "CodeAnalysis", "Implementation", "Validator"]
            }
            
            logger.info(f"Workflow completed for query: '{original_query}' in {total_time:.2f}s")
            
            return StopEvent(result=result)
            
        except Exception as e:
            logger.error(f"Result finalization failed: {e}")
            return StopEvent(result={
                "error": f"Workflow failed: {str(e)}",
                "query": original_query,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat()
            })
    
    def _organize_research_results(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Organize research results by analysis type"""
        
        organized = {}
        
        for result in results:
            if not isinstance(result, dict):
                continue
                
            analysis_type = result.get("analysis_type", "unknown")
            
            if analysis_type not in organized:
                organized[analysis_type] = {}
            
            # Merge results of the same type
            if "results" in result:
                organized[analysis_type].update(result["results"])
            
            # Keep metadata
            organized[analysis_type]["_metadata"] = {
                "task_id": result.get("task_id"),
                "confidence": result.get("confidence", 0.0),
                "execution_time": result.get("execution_time", 0.0),
                "completed_at": result.get("completed_at")
            }
        
        return organized
    
    def _generate_next_steps(self, validation_event: ValidationEvent) -> List[str]:
        """Generate recommended next steps based on validation results"""
        
        next_steps = []
        
        if validation_event.approved:
            next_steps.extend([
                "Implementation has been approved and can proceed",
                "Run the generated code in a development environment",
                "Execute validation steps as outlined in the plan",
                "Monitor for any runtime issues or edge cases"
            ])
        else:
            next_steps.extend([
                "Implementation requires review before proceeding",
                "Address the required changes listed in feedback",
                "Consider manual review of high-risk components"
            ])
            
            # Add specific next steps based on required changes
            if validation_event.required_changes:
                next_steps.append("Required changes:")
                next_steps.extend([f"  - {change}" for change in validation_event.required_changes])
        
        # Always suggest testing
        next_steps.extend([
            "Create comprehensive tests for the implementation",
            "Document the implementation for future reference",
            "Consider code review by team members"
        ])
        
        return next_steps
    
    # Convenience methods for external usage
    async def process_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Convenience method to process a query and return results
        """
        handler = self.run(query=query, context=context or {})
        result = await handler
        return result
    
    async def stream_process_query(self, query: str, context: Dict[str, Any] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the workflow execution with progress updates
        """
        handler = self.run(query=query, context=context or {})
        
        async for event in handler.stream_events():
            if isinstance(event, StreamingEvent):
                yield {
                    "type": "progress",
                    "agent": event.agent_name,
                    "stage": event.stage,
                    "progress": event.progress_percentage,
                    "task": event.current_task,
                    "details": event.details
                }
        
        # Return final result
        result = await handler
        yield {
            "type": "result",
            "data": result
        } 