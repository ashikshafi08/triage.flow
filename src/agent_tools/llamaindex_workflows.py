"""
LlamaIndex AgentWorkflow integration for triage.flow

Following patterns from:
https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/

This module implements three main workflow patterns:
1. Linear Swarm: Automatic handoffs with minimal configuration
2. Orchestrator: Central agent managing sub-agents as tools
3. Custom Planner: Maximum flexibility with manual control
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from ..config import settings
from .llm_config import get_llm_instance
from .tool_registry import create_all_tools
from .context_manager import ContextManager
from .context_aware_tools import ContextAwareToolFactory
from ..agentic_rag import QueryComplexity
from .llamaindex_comprehensive_workflow import run_comprehensive_analysis

logger = logging.getLogger(__name__)


def calculate_dynamic_iterations(query: str, base_iterations: int = None, manual_override: int = None) -> int:
    """
    Calculate dynamic iteration count based on query complexity analysis.
    Uses the same complexity analysis as the composite RAG system.
    
    Args:
        query: The query string to analyze
        base_iterations: Base number of iterations (defaults to settings)
        manual_override: Manual override for iterations (bypasses complexity analysis)
    """
    # Manual override takes precedence for repository analysis tasks
    if manual_override is not None:
        logger.info(f"Using manual iteration override: {manual_override}")
        return manual_override
    
    if base_iterations is None:
        base_iterations = settings.AGENTIC_BASE_ITERATIONS
    
    # Use the same analysis logic as CompositeAgenticRetriever
    query_lower = query.lower()
    word_count = len(query.split())
    
    # Determine complexity using same logic as agentic_rag.py
    if word_count > 20:
        complexity = QueryComplexity.COMPLEX
    elif word_count > 10:
        complexity = QueryComplexity.MODERATE
    else:
        complexity = QueryComplexity.SIMPLE
    
    # Check for agentic patterns that indicate complex reasoning needed
    agentic_patterns = [
        "explain", "analyze", "how does", "implement", "create", "find all",
        "comprehensive", "detailed", "step by step", "security", "vulnerability",
        "architecture", "design pattern", "best practices", "refactor", "review"
    ]
    has_agentic_patterns = any(pattern in query_lower for pattern in agentic_patterns)
    
    # For comprehensive repository analysis, use much higher iterations
    repository_analysis_patterns = [
        "comprehensive", "security", "performance", "review", "analyze", "vulnerabilities",
        "entire repository", "codebase", "architecture", "full analysis"
    ]
    is_repo_analysis = any(pattern in query_lower for pattern in repository_analysis_patterns)
    
    if is_repo_analysis:
        # Repository analysis needs much more iterations
        iterations = 100  # Start with 100 for comprehensive analysis
        logger.info(f"Repository analysis detected - using {iterations} iterations")
    elif complexity == QueryComplexity.COMPLEX or has_agentic_patterns:
        iterations = int(base_iterations * settings.AGENTIC_COMPLEXITY_MULTIPLIER)
    elif complexity == QueryComplexity.MODERATE:
        iterations = int(base_iterations * 1.5)
    else:
        iterations = base_iterations
    
    # Apply bounds from settings (but allow higher for repo analysis)
    if not is_repo_analysis:
        iterations = max(settings.AGENTIC_MIN_ITERATIONS, iterations)
        iterations = min(settings.AGENTIC_MAX_ITERATIONS, iterations)
    
    logger.info(f"Dynamic iterations calculated: {iterations} (complexity: {complexity.value}, patterns: {has_agentic_patterns}, repo_analysis: {is_repo_analysis})")
    return iterations


class WorkflowType(Enum):
    """Types of workflow patterns supported"""
    LINEAR_SWARM = "linear_swarm"
    ORCHESTRATOR = "orchestrator"
    CUSTOM_PLANNER = "custom_planner"


@dataclass
class AgentConfig:
    """Configuration for individual agents in workflow"""
    id: str
    name: str
    role: str
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)
    system_prompt: Optional[str] = None
    max_iterations: int = None  # Will be set dynamically based on query complexity
    
    def __post_init__(self):
        """Initialize max_iterations if not explicitly set"""
        if self.max_iterations is None:
            self.max_iterations = settings.AGENTIC_BASE_ITERATIONS
    can_handoff_to: List[str] = field(default_factory=list)
    specialization: Optional[str] = None


@dataclass
class WorkflowConfig:
    """Configuration for workflow instances"""
    id: str
    name: str
    type: WorkflowType
    agents: List[AgentConfig]
    entry_agent: Optional[str] = None
    max_workflow_iterations: int = 20
    enable_memory_sharing: bool = True
    enable_handoff_reasoning: bool = True
    parallel_execution: bool = False


# Workflow Events (inheriting from LlamaIndex Event)
class AgentActivationEvent(Event):
    """Event when an agent is activated"""
    agent_id: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


class AgentCompletionEvent(Event):
    """Event when an agent completes its task"""
    agent_id: str
    message: str
    result: Any = None
    confidence: float = 0.0
    next_agent: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class HandoffEvent(Event):
    """Event for agent handoffs"""
    from_agent: str
    to_agent: str
    reason: str
    message: str
    handoff_data: Dict[str, Any] = field(default_factory=dict)


class WorkflowCompletionEvent(Event):
    """Event when entire workflow completes"""
    final_result: Any
    execution_summary: Dict[str, Any]


class TriageFlowAgentWorkflow(Workflow):
    """Base workflow class for triage.flow multi-agent systems"""
    
    def __init__(
        self,
        session_id: str,
        repo_path: str,
        config: WorkflowConfig,
        llm: Optional[LLM] = None,
        tools: Optional[List[BaseTool]] = None,
        explorer: Optional[Any] = None
    ):
        super().__init__(timeout=300.0, verbose=True)
        
        self.session_id = session_id
        self.repo_path = Path(repo_path)
        self.config = config
        self.llm = llm or get_llm_instance()
        self.explorer = explorer
        
        # Initialize context management
        self.context_manager = ContextManager(session_id, self.repo_path)
        
        # Create context-aware tools if explorer is provided
        if self.explorer:
            context_tool_factory = ContextAwareToolFactory(self.context_manager)
            self.tools = context_tool_factory.create_context_aware_tools(self.explorer)
        else:
            self.tools = tools or []
        
        # Agent instances
        self.agents: Dict[str, FunctionCallingAgentWorker] = {}
        
        # Workflow state
        self.execution_history: List[Dict[str, Any]] = []
        self.shared_memory: Dict[str, Any] = {}
        self.current_agent: Optional[str] = None
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info(f"Initialized TriageFlowAgentWorkflow: {config.name} for session {session_id}")
    
    def _initialize_agents(self):
        """Initialize all agents defined in configuration"""
        for agent_config in self.config.agents:
            agent = self._create_agent(agent_config)
            self.agents[agent_config.id] = agent
            logger.debug(f"Created agent: {agent_config.id} ({agent_config.role})")
    
    def _create_agent(self, config: AgentConfig) -> FunctionCallingAgentWorker:
        """Create individual agent from configuration"""
        # Get agent-specific tools
        agent_tools = self._get_tools_for_agent(config)
        
        # Create system prompt
        system_prompt = config.system_prompt or self._generate_system_prompt(config)
        
        # Create memory buffer
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        
        # Create agent using FunctionCallingAgentWorker
        agent = FunctionCallingAgentWorker.from_tools(
            tools=agent_tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
            allow_parallel_tool_calls=True
        )
        
        return agent
    
    def set_query_and_context(self, query: str, context: Dict[str, Any], manual_iterations: int = None):
        """Set query and context for workflow execution"""
        self._query = query
        self._context = context
        
        # Update agent max_iterations based on query complexity or manual override
        self._update_agent_iterations_for_query(query, manual_override=manual_iterations)
    
    def _update_agent_iterations_for_query(self, query: str, manual_override: int = None):
        """Update all agents' max_iterations based on query complexity"""
        # Note: FunctionCallingAgentWorker doesn't use max_iterations in the same way as ReActAgent
        # It manages iterations through the task lifecycle instead
        if not settings.AGENTIC_DYNAMIC_ITERATIONS and manual_override is None:
            return
        
        dynamic_iterations = calculate_dynamic_iterations(query, manual_override=manual_override)
        logger.info(f"Dynamic iterations calculated: {dynamic_iterations} (FunctionCallingAgentWorker manages iterations internally)")
    
    def increase_iterations_by(self, increment: int = 100):
        """Increase max_iterations for all agents by the specified increment"""
        # Note: FunctionCallingAgentWorker doesn't expose max_iterations the same way
        # It uses internal task management instead
        logger.info(f"Iteration increase requested by {increment} (FunctionCallingAgentWorker manages iterations internally)")
    
    def _get_tools_for_agent(self, config: AgentConfig) -> List[BaseTool]:
        """Get tools specific to an agent's role"""
        if not config.tools:
            return self.tools  # Use all tools if none specified
        
        # Filter tools based on agent configuration
        agent_tools = []
        for tool in self.tools:
            if tool.metadata.name in config.tools:
                agent_tools.append(tool)
        
        return agent_tools
    
    def _generate_system_prompt(self, config: AgentConfig) -> str:
        """Generate system prompt for agent based on configuration"""
        base_prompt = f"""You are {config.name}, a {config.role} specialist.

Your primary goal: {config.goal}

Background: {config.backstory}

You are working in a multi-agent workflow for repository analysis and issue resolution.
Current repository: {self.repo_path}
Session ID: {self.session_id}

Guidelines:
1. Focus on your specialization: {config.specialization or config.role}
2. Collaborate effectively with other agents
3. Provide clear, actionable insights
4. When your analysis is complete, clearly state your findings
5. If you need to handoff to another agent, specify which agent and why

Available tools: {[tool.metadata.name for tool in self._get_tools_for_agent(config)]}
"""
        
        if config.can_handoff_to:
            base_prompt += f"\nYou can handoff to these agents: {', '.join(config.can_handoff_to)}"
        
        return base_prompt
    
    @step
    async def start_workflow(self, ctx: Context, ev: StartEvent) -> AgentActivationEvent:
        """Entry point for workflow execution"""
        logger.info(f"Starting workflow: {self.config.name}")
        
        # Get query from workflow instance (set by the API)
        query = getattr(self, "_query", "Begin analysis")
        context = getattr(self, "_context", {})
        
        # CRITICAL: Start execution context for context-aware tools
        if self.context_manager:
            execution_context = self.context_manager.start_execution(query)
            logger.info(f"Started execution context for workflow query: {query[:100]}...")
        
        # Determine entry agent
        entry_agent = self.config.entry_agent or self.config.agents[0].id
        self.current_agent = entry_agent
        
        # Log workflow start
        self.execution_history.append({
            "type": "workflow_start",
            "timestamp": datetime.now().isoformat(),
            "entry_agent": entry_agent,
            "query": query
        })
        
        return AgentActivationEvent(
            agent_id=entry_agent,
            message=query,
            context={"workflow_type": self.config.type.value, **context}
        )
    
    @step
    async def process_agent_activation(
        self, 
        ctx: Context, 
        ev: AgentActivationEvent
    ) -> Union[AgentCompletionEvent, HandoffEvent, WorkflowCompletionEvent]:
        """Process agent activation and execution"""
        logger.info(f"Activating agent: {ev.agent_id}")
        
        agent = self.agents.get(ev.agent_id)
        if not agent:
            logger.error(f"Agent {ev.agent_id} not found")
            return WorkflowCompletionEvent(
                final_result=f"Error: Agent {ev.agent_id} not found",
                execution_summary={"error": True, "agent_not_found": ev.agent_id}
            )
        
        try:
            # Execute agent using FunctionCallingAgentWorker API
            start_time = datetime.now()
            task = agent.create_task(ev.message)
            response = await agent.arun(task)
            end_time = datetime.now()
            
            # Log execution
            execution_log = {
                "type": "agent_execution",
                "agent_id": ev.agent_id,
                "timestamp": start_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "input": ev.message,
                "output": str(response),
                "context": ev.context
            }
            self.execution_history.append(execution_log)
            
            # Analyze response for handoff or completion
            handoff_info = await self._analyze_response_for_handoff(ev.agent_id, str(response))
            
            if handoff_info:
                return HandoffEvent(
                    agent_id=ev.agent_id,
                    message=str(response),
                    from_agent=ev.agent_id,
                    to_agent=handoff_info["to_agent"],
                    reason=handoff_info["reason"],
                    handoff_data=handoff_info.get("data", {})
                )
            else:
                # Check if workflow should complete
                if await self._should_complete_workflow(ev.agent_id, str(response)):
                    return WorkflowCompletionEvent(
                        final_result=str(response),
                        execution_summary=self._generate_execution_summary()
                    )
                else:
                    return AgentCompletionEvent(
                        agent_id=ev.agent_id,
                        message=str(response),
                        result=str(response),
                        confidence=0.8  # TODO: Calculate actual confidence
                    )
        
        except Exception as e:
            logger.error(f"Error executing agent {ev.agent_id}: {e}")
            return WorkflowCompletionEvent(
                final_result=f"Error in agent execution: {str(e)}",
                execution_summary={"error": True, "exception": str(e)}
            )
    
    @step
    async def process_handoff(
        self, 
        ctx: Context, 
        ev: HandoffEvent
    ) -> AgentActivationEvent:
        """Process agent handoff"""
        logger.info(f"Processing handoff from {ev.from_agent} to {ev.to_agent}: {ev.reason}")
        
        # Update current agent
        self.current_agent = ev.to_agent
        
        # Prepare handoff message
        handoff_message = f"""Previous agent ({ev.from_agent}) has completed their analysis:

{ev.message}

Handoff reason: {ev.reason}

Please continue the analysis based on this information."""
        
        # Add to shared memory if enabled
        if self.config.enable_memory_sharing:
            self.shared_memory[f"handoff_{ev.from_agent}_to_{ev.to_agent}"] = {
                "timestamp": datetime.now().isoformat(),
                "message": ev.message,
                "reason": ev.reason,
                "data": ev.handoff_data
            }
        
        return AgentActivationEvent(
            agent_id=ev.to_agent,
            message=handoff_message,
            context={
                "handoff_from": ev.from_agent,
                "handoff_reason": ev.reason,
                **ev.context
            }
        )
    
    @step
    async def process_completion(
        self, 
        ctx: Context, 
        ev: AgentCompletionEvent
    ) -> Union[WorkflowCompletionEvent, AgentActivationEvent]:
        """Process agent completion"""
        logger.info(f"Agent {ev.agent_id} completed with confidence {ev.confidence}")
        
        # Check if this is the final completion
        if ev.next_agent:
            return AgentActivationEvent(
                agent_id=ev.next_agent,
                message=f"Continuing from {ev.agent_id}: {ev.message}",
                context={"previous_agent": ev.agent_id}
            )
        else:
            return WorkflowCompletionEvent(
                final_result=ev.result,
                execution_summary=self._generate_execution_summary()
            )

    @step
    async def finalize_workflow(
        self, 
        ctx: Context, 
        ev: WorkflowCompletionEvent
    ) -> StopEvent:
        """Final step that properly terminates the workflow"""
        logger.info("Workflow completed, finalizing...")
        
        # Log final completion
        self.execution_history.append({
            "type": "workflow_complete",
            "timestamp": datetime.now().isoformat(),
            "final_result": ev.final_result,
            "execution_summary": ev.execution_summary
        })
        
        # Return StopEvent with the final result
        return StopEvent(result={
            "final_result": ev.final_result,
            "execution_summary": ev.execution_summary,
            "execution_history": self.execution_history,
            "total_duration": self._calculate_total_duration(),
            "agents_used": list(self.agents.keys()),
            "workflow_type": self.config.type.value,
            "session_id": self.session_id
        })
    
    async def _analyze_response_for_handoff(
        self, 
        agent_id: str, 
        response: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze agent response to determine if handoff is needed"""
        # Simple keyword-based handoff detection
        # TODO: Implement more sophisticated LLM-based analysis
        
        agent_config = next(
            (a for a in self.config.agents if a.id == agent_id), 
            None
        )
        
        if not agent_config or not agent_config.can_handoff_to:
            return None
        
        # Check for handoff patterns in response
        handoff_keywords = {
            "security": ["security", "vulnerability", "exploit", "unsafe"],
            "quality": ["quality", "standards", "best practices", "code review"],
            "testing": ["test", "testing", "validation", "qa"],
            "issue_resolution": ["issue", "bug", "problem", "resolution"]
        }
        
        response_lower = response.lower()
        
        for target_agent in agent_config.can_handoff_to:
            if target_agent in handoff_keywords:
                keywords = handoff_keywords[target_agent]
                if any(keyword in response_lower for keyword in keywords):
                    return {
                        "to_agent": target_agent,
                        "reason": f"Detected {target_agent} related content",
                        "data": {"keywords_found": keywords}
                    }
        
        return None
    
    async def _should_complete_workflow(self, agent_id: str, response: str) -> bool:
        """Determine if workflow should complete"""
        # Simple completion detection
        completion_indicators = [
            "analysis complete",
            "investigation finished",
            "no further action needed",
            "final recommendation",
            "conclusion:"
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in completion_indicators)
    
    def _generate_execution_summary(self) -> Dict[str, Any]:
        """Generate summary of workflow execution"""
        return {
            "workflow_id": self.config.id,
            "session_id": self.session_id,
            "execution_history": self.execution_history,
            "agents_used": list(set(log["agent_id"] for log in self.execution_history if "agent_id" in log)),
            "total_duration": self._calculate_total_duration(),
            "handoffs_count": len([log for log in self.execution_history if log.get("type") == "handoff"]),
            "shared_memory": self.shared_memory
        }
    
    def _calculate_total_duration(self) -> float:
        """Calculate total workflow execution time"""
        if not self.execution_history:
            return 0.0
        
        start_time = datetime.fromisoformat(self.execution_history[0]["timestamp"])
        end_time = datetime.now()
        
        return (end_time - start_time).total_seconds()
    
    async def _should_use_comprehensive_workflow(self, query: str) -> bool:
        """Determine if query should use the comprehensive analysis workflow"""
        comprehensive_patterns = [
            "comprehensive", "full", "complete", "entire", "repository", "codebase",
            "security and performance", "review", "audit", "analysis", "assess",
            "vulnerabilities", "bottlenecks", "quality", "architecture"
        ]
        
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in comprehensive_patterns)
    
    async def _execute_workflow_directly(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow directly without LlamaIndex event system
        This bypasses StartEvent issues with certain LlamaIndex versions
        Includes automatic retry with increased iterations if max iterations reached
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                logger.info(f"Starting direct workflow execution (attempt {retry_count + 1}): {self.config.name}")
                
                # Check if we should use the comprehensive analysis workflow
                if await self._should_use_comprehensive_workflow(query):
                    logger.info("Using LlamaIndex AgentWorkflow for comprehensive analysis")
                    
                    # Extract focus areas from query
                    focus_areas = []
                    query_lower = query.lower()
                    if any(word in query_lower for word in ["security", "vulnerabilities", "secure"]):
                        focus_areas.append("security")
                    if any(word in query_lower for word in ["performance", "speed", "bottleneck", "optimization"]):
                        focus_areas.append("performance")
                    if any(word in query_lower for word in ["quality", "architecture", "design", "code"]):
                        focus_areas.append("quality")
                    if any(word in query_lower for word in ["dependencies", "requirements", "packages"]):
                        focus_areas.append("dependencies")
                    
                    # If no specific focus areas, analyze everything
                    if not focus_areas:
                        focus_areas = ["structure", "dependencies", "security", "performance", "quality"]
                    
                    try:
                        result = await run_comprehensive_analysis(
                            session_id=self.session_id,
                            repo_path=str(self.repo_path),
                            query=query,
                            focus_areas=focus_areas,
                            context_manager=self.context_manager,
                            llm=self.llm,
                            tools=self.tools
                        )
                        
                        # If comprehensive workflow succeeds, return immediately
                        if result and not result.get("error"):
                            logger.info("Comprehensive analysis completed successfully")
                            return {
                                "final_result": result,
                                "execution_summary": {
                                    "workflow_type": "llamaindex_comprehensive",
                                    "session_id": self.session_id,
                                    "success": True
                                },
                                "execution_history": [{
                                    "type": "comprehensive_analysis",
                                    "timestamp": datetime.now().isoformat(),
                                    "query": query,
                                    "focus_areas": focus_areas
                                }],
                                "total_duration": 0.0,
                                "agents_used": ["comprehensive_workflow"],
                                "workflow_type": self.config.type.value,
                                "retries_used": 0
                            }
                        else:
                            logger.warning(f"Comprehensive workflow returned error, falling back to linear workflow")
                    except Exception as e:
                        logger.warning(f"Comprehensive workflow error: {e}, falling back to agent workflow")
                
                # CRITICAL: Start execution context for context-aware tools
                if self.context_manager:
                    execution_context = self.context_manager.start_execution(query)
                    logger.info(f"Started execution context for direct workflow: {query[:100]}...")
                
                # Initialize workflow
                start_time = datetime.now()
                entry_agent = self.config.entry_agent or self.config.agents[0].id
                
                # Log workflow start
                self.execution_history.append({
                    "type": "workflow_start",
                    "timestamp": start_time.isoformat(),
                    "entry_agent": entry_agent,
                    "query": query,
                    "context": context,
                    "retry_attempt": retry_count + 1
                })
                
                # Execute the workflow based on type
                if self.config.type == WorkflowType.LINEAR_SWARM:
                    result = await self._execute_linear_swarm(query, context)
                elif self.config.type == WorkflowType.ORCHESTRATOR:
                    result = await self._execute_orchestrator(query, context)
                else:
                    result = await self._execute_custom_workflow(query, context)
                
                # Log completion
                end_time = datetime.now()
                self.execution_history.append({
                    "type": "workflow_complete",
                    "timestamp": end_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "result": result,
                    "retry_attempt": retry_count + 1
                })
                
                # If we reach here, execution was successful
                break
                
            except Exception as e:
                error_message = str(e)
                if "Reached max iterations" in error_message and retry_count < max_retries:
                    # Increase iterations and retry
                    increment = 100
                    logger.warning(f"Max iterations reached on attempt {retry_count + 1}, increasing by {increment} and retrying...")
                    self.increase_iterations_by(increment)
                    
                    # Log the retry
                    self.execution_history.append({
                        "type": "workflow_retry",
                        "timestamp": datetime.now().isoformat(),
                        "reason": "max_iterations_reached",
                        "retry_attempt": retry_count + 1,
                        "iterations_increased_by": increment
                    })
                    
                    retry_count += 1
                    continue
                else:
                    # Re-raise the exception if it's not a max iterations error or we've exceeded retries
                    raise e
        
        # Calculate final duration
        final_end_time = datetime.now()
        if self.execution_history:
            first_start = next((log for log in self.execution_history if log.get("type") == "workflow_start"), None)
            if first_start:
                total_duration = (final_end_time - datetime.fromisoformat(first_start["timestamp"])).total_seconds()
            else:
                total_duration = 0.0
        else:
            total_duration = 0.0
            
        return {
            "final_result": result,
            "execution_summary": self._generate_execution_summary(),
            "execution_history": self.execution_history,
            "total_duration": total_duration,
            "agents_used": list(self.agents.keys()),
            "workflow_type": self.config.type.value,
            "session_id": self.session_id,
            "retries_used": retry_count
        }
        
    async def _execute_workflow_directly_fallback(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback execution method in case of critical errors"""
        try:
            logger.error(f"Direct workflow execution failed after retries, using fallback")
            self.execution_history.append({
                "type": "workflow_error",
                "timestamp": datetime.now().isoformat(),
                "error": "Max retries exceeded for iteration limit"
            })
            return {
                "final_result": f"Workflow exceeded maximum retries due to iteration limits. Consider breaking down the query into smaller parts.",
                "execution_summary": {"error": True, "exception": "Max iteration retries exceeded"},
                "execution_history": self.execution_history
            }
        except Exception as e:
            logger.error(f"Fallback execution also failed: {e}")
            return {
                "final_result": f"Workflow failed: {str(e)}",
                "execution_summary": {"error": True, "exception": str(e)},
                "execution_history": self.execution_history
            }
    
    async def _execute_linear_swarm(self, query: str, context: Dict[str, Any]) -> str:
        """Execute linear swarm pattern: code_analysis → issue_resolution → testing_qa"""
        agent_sequence = ["code_analysis", "issue_resolution", "testing"]
        current_message = query
        
        for agent_id in agent_sequence:
            if agent_id not in self.agents:
                logger.warning(f"Agent {agent_id} not found, skipping")
                continue
            
            agent = self.agents[agent_id]
            logger.info(f"Executing agent: {agent_id}")
            
            try:
                start_time = datetime.now()
                task = agent.create_task(current_message)
                response = await agent.arun(task)
                end_time = datetime.now()
                
                # Log execution
                self.execution_history.append({
                    "type": "agent_execution",
                    "agent_id": agent_id,
                    "timestamp": start_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "input": current_message,
                    "output": str(response)
                })
                
                # Prepare message for next agent
                current_message = f"""Previous analysis from {agent_id}:

{str(response)}

Please continue the analysis building on this information."""
                
            except Exception as e:
                error_message = str(e)
                logger.error(f"Error executing agent {agent_id}: {e}")
                # Re-raise max iterations errors to be handled by the retry logic
                if "Reached max iterations" in error_message:
                    raise e
                return f"Error in {agent_id}: {str(e)}"
        
        return current_message
    
    async def _execute_orchestrator(self, query: str, context: Dict[str, Any]) -> str:
        """Execute orchestrator pattern: orchestrator manages sub-agents"""
        orchestrator = self.agents.get("orchestrator")
        if not orchestrator:
            return "Error: Orchestrator agent not found"
        
        try:
            start_time = datetime.now()
            task = orchestrator.create_task(query)
            response = await orchestrator.arun(task)
            end_time = datetime.now()
            
            # Log execution
            self.execution_history.append({
                "type": "agent_execution",
                "agent_id": "orchestrator",
                "timestamp": start_time.isoformat(),
                "duration": (end_time - start_time).total_seconds(),
                "input": query,
                "output": str(response)
            })
            
            return str(response)
            
        except Exception as e:
            logger.error(f"Error executing orchestrator: {e}")
            return f"Error in orchestrator: {str(e)}"
    
    async def _execute_custom_workflow(self, query: str, context: Dict[str, Any]) -> str:
        """Execute custom workflow pattern"""
        # For now, just execute all agents in sequence
        results = []
        
        for agent_config in self.config.agents:
            agent = self.agents.get(agent_config.id)
            if not agent:
                continue
            
            try:
                start_time = datetime.now()
                task = agent.create_task(query)
                response = await agent.arun(task)
                end_time = datetime.now()
                
                # Log execution
                self.execution_history.append({
                    "type": "agent_execution",
                    "agent_id": agent_config.id,
                    "timestamp": start_time.isoformat(),
                    "duration": (end_time - start_time).total_seconds(),
                    "input": query,
                    "output": str(response)
                })
                
                results.append(f"**{agent_config.name}:**\n{str(response)}")
                
            except Exception as e:
                logger.error(f"Error executing agent {agent_config.id}: {e}")
                results.append(f"**{agent_config.name}:** Error - {str(e)}")
        
        return "\n\n".join(results)


class LinearSwarmWorkflow(TriageFlowAgentWorkflow):
    """Implementation of Linear Swarm pattern with automatic handoffs"""
    
    def __init__(self, session_id: str, repo_path: str, explorer: Optional[Any] = None, **kwargs):
        # Create default linear swarm configuration
        config = WorkflowConfig(
            id=f"linear_swarm_{session_id}",
            name="Linear Swarm Analysis",
            type=WorkflowType.LINEAR_SWARM,
            agents=[
                AgentConfig(
                    id="code_analysis",
                    name="Code Analysis Specialist",
                    role="Code Analyzer",
                    goal="Analyze code structure and understand implementation",
                    backstory="Expert in code analysis and understanding complex codebases",
                    can_handoff_to=["issue_resolution", "testing"],
                    specialization="code_analysis"
                ),
                AgentConfig(
                    id="issue_resolution",
                    name="Issue Resolution Specialist", 
                    role="Problem Solver",
                    goal="Identify and propose solutions for issues",
                    backstory="Experienced in debugging and issue resolution",
                    can_handoff_to=["testing"],
                    specialization="issue_resolution"
                ),
                AgentConfig(
                    id="testing",
                    name="Testing & QA Specialist",
                    role="Quality Assurance",
                    goal="Ensure code quality and proper testing",
                    backstory="Expert in testing strategies and quality assurance",
                    can_handoff_to=[],
                    specialization="testing"
                )
            ],
            entry_agent="code_analysis"
        )
        
        super().__init__(session_id, repo_path, config, explorer=explorer, **kwargs)


class OrchestratorWorkflow(TriageFlowAgentWorkflow):
    """Implementation of Orchestrator pattern with central coordination"""
    
    def __init__(self, session_id: str, repo_path: str, explorer: Optional[Any] = None, **kwargs):
        # Create orchestrator configuration
        config = WorkflowConfig(
            id=f"orchestrator_{session_id}",
            name="Orchestrated Analysis",
            type=WorkflowType.ORCHESTRATOR,
            agents=[
                AgentConfig(
                    id="orchestrator",
                    name="Analysis Orchestrator",
                    role="Orchestrator",
                    goal="Coordinate analysis across multiple specialized agents",
                    backstory="Expert in managing complex analysis workflows",
                    can_handoff_to=["code_analysis", "security", "quality"],
                    specialization="orchestration"
                ),
                AgentConfig(
                    id="code_analysis",
                    name="Code Analysis Agent",
                    role="Code Analyzer",
                    goal="Deep code analysis and understanding",
                    backstory="Specialized in code structure analysis",
                    can_handoff_to=["orchestrator"],
                    specialization="code_analysis"
                ),
                AgentConfig(
                    id="security",
                    name="Security Analysis Agent",
                    role="Security Specialist",
                    goal="Identify security vulnerabilities and risks",
                    backstory="Expert in security analysis and threat detection",
                    can_handoff_to=["orchestrator"],
                    specialization="security"
                ),
                AgentConfig(
                    id="quality",
                    name="Quality Analysis Agent",
                    role="Quality Specialist",
                    goal="Assess code quality and best practices",
                    backstory="Expert in code quality and best practices",
                    can_handoff_to=["orchestrator"],
                    specialization="quality"
                )
            ],
            entry_agent="orchestrator"
        )
        
        super().__init__(session_id, repo_path, config, explorer=explorer, **kwargs)


class CustomPlannerWorkflow(TriageFlowAgentWorkflow):
    """Implementation of Custom Planner pattern with maximum flexibility"""
    
    def __init__(
        self, 
        session_id: str, 
        repo_path: str, 
        custom_config: WorkflowConfig,
        explorer: Optional[Any] = None,
        **kwargs
    ):
        super().__init__(session_id, repo_path, custom_config, explorer=explorer, **kwargs) 