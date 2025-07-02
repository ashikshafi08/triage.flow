"""
Workflow Management API Endpoints
Part of LlamaIndex Workflow Integration for triage.flow
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...agent_tools.llamaindex_workflows import (
    LinearSwarmWorkflow, 
    OrchestratorWorkflow, 
    CustomPlannerWorkflow,
    WorkflowType,
    WorkflowConfig,
    AgentConfig,
    TriageFlowAgentWorkflow
)
from ...agent_tools.workflow_state import (
    WorkflowStateManager,
    WorkflowEventManager, 
    WorkflowStatus,
    workflow_state_manager,
    workflow_event_manager
)
from ...agent_tools.workflow_agents import WorkflowAgentFactory
from ..dependencies import get_session, session_manager, logger, settings

router = APIRouter(prefix="/assistant/sessions", tags=["workflows"])

# Pydantic models for API
class CreateWorkflowRequest(BaseModel):
    workflow_type: str = Field(..., description="Type of workflow: linear_swarm, orchestrator, custom_planner")
    name: Optional[str] = Field(None, description="Custom name for the workflow")
    agents: Optional[List[Dict[str, Any]]] = Field(None, description="Custom agent configurations")
    entry_agent: Optional[str] = Field(None, description="ID of the entry agent")
    config: Optional[Dict[str, Any]] = Field(None, description="Additional workflow configuration")

class ExecuteWorkflowRequest(BaseModel):
    query: str = Field(..., description="Query to execute in the workflow")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for execution")

class WorkflowResponse(BaseModel):
    workflow_id: str
    session_id: str
    status: str
    created_at: str
    updated_at: str
    config: Dict[str, Any]
    execution_history: List[Dict[str, Any]] = []
    result: Optional[Any] = None
    error_message: Optional[str] = None

class WorkflowStatusResponse(BaseModel):
    workflow_id: str
    status: str
    current_agent: Optional[str] = None
    progress: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    last_update: str


# Active workflow instances for WebSocket management
active_workflows: Dict[str, TriageFlowAgentWorkflow] = {}
websocket_connections: Dict[str, List[WebSocket]] = {}


@router.post("/{session_id}/workflows/create")
async def create_workflow(
    session_id: str,
    request: CreateWorkflowRequest,
    session: Dict[str, Any] = Depends(get_session)
) -> Dict[str, Any]:
    """Create a new workflow instance"""
    try:
        # Generate workflow ID
        workflow_id = f"workflow_{session_id}_{uuid.uuid4().hex[:8]}"
        
        # Get repository path from session
        repo_path = session.get("repo_path", "/unknown")
        
        # Create workflow configuration
        config = await _create_workflow_config(
            workflow_id, 
            session_id,
            request.workflow_type,
            request.name,
            request.agents,
            request.entry_agent,
            request.config
        )
        
        # Create workflow state
        workflow_state = await workflow_state_manager.create_workflow(
            workflow_id=workflow_id,
            session_id=session_id,
            config=config
        )
        
        # Initialize workflow instance
        workflow_instance = await _create_workflow_instance(
            session_id=session_id,
            repo_path=repo_path,
            config=config
        )
        
        # Store in active workflows
        active_workflows[workflow_id] = workflow_instance
        
        logger.info(f"Created workflow {workflow_id} of type {request.workflow_type}")
        
        return {
            "workflow_id": workflow_id,
            "status": "created",
            "type": request.workflow_type,
            "config": config.__dict__ if hasattr(config, '__dict__') else str(config),
            "created_at": workflow_state.created_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to create workflow: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")


@router.post("/{session_id}/workflows/{workflow_id}/execute")
async def execute_workflow(
    session_id: str,
    workflow_id: str,
    request: ExecuteWorkflowRequest,
    background_tasks: BackgroundTasks,
    session: Dict[str, Any] = Depends(get_session)
) -> Dict[str, Any]:
    """Execute a workflow with the given query"""
    try:
        # Get workflow instance
        workflow_instance = active_workflows.get(workflow_id)
        if not workflow_instance:
            # Try to recreate from state
            workflow_state = await workflow_state_manager.load_workflow_state(workflow_id)
            if not workflow_state:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            # Recreate workflow instance
            repo_path = session.get("repo_path", "/unknown")
            workflow_instance = await _create_workflow_instance(
                session_id=session_id,
                repo_path=repo_path,
                config=workflow_state.config
            )
            active_workflows[workflow_id] = workflow_instance
        
        # Update workflow status to running
        await workflow_state_manager.update_workflow_status(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING
        )
        
        # Execute workflow in background
        background_tasks.add_task(
            _execute_workflow_background,
            workflow_id,
            workflow_instance,
            request.query,
            request.context or {}
        )
        
        # Publish workflow start event
        await workflow_event_manager.publish_workflow_event(
            workflow_id=workflow_id,
            event_type="workflow_started",
            data={
                "query": request.query,
                "context": request.context,
                "workflow_type": workflow_instance.config.type.value
            }
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "running",
            "message": "Workflow execution started",
            "query": request.query
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to execute workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute workflow: {str(e)}")


@router.get("/{session_id}/workflows/{workflow_id}/status")
async def get_workflow_status(
    session_id: str,
    workflow_id: str,
    session: Dict[str, Any] = Depends(get_session)
) -> WorkflowStatusResponse:
    """Get current workflow status"""
    try:
        # Load workflow state
        workflow_state = await workflow_state_manager.load_workflow_state(workflow_id)
        if not workflow_state:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Check if workflow is active
        workflow_instance = active_workflows.get(workflow_id)
        current_agent = None
        if workflow_instance:
            current_agent = workflow_instance.current_agent
        
        # Calculate progress
        progress = _calculate_workflow_progress(workflow_state)
        
        return WorkflowStatusResponse(
            workflow_id=workflow_id,
            status=workflow_state.status.value,
            current_agent=current_agent,
            progress=progress,
            context=workflow_state.shared_memory.get('execution_context', {}),
            last_update=workflow_state.updated_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow status: {str(e)}")


@router.post("/{session_id}/workflows/{workflow_id}/pause")
async def pause_workflow(
    session_id: str,
    workflow_id: str,
    session: Dict[str, Any] = Depends(get_session)
) -> Dict[str, Any]:
    """Pause workflow execution"""
    try:
        # Update workflow status
        success = await workflow_state_manager.update_workflow_status(
            workflow_id=workflow_id,
            status=WorkflowStatus.PAUSED
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Publish pause event
        await workflow_event_manager.publish_workflow_event(
            workflow_id=workflow_id,
            event_type="workflow_paused",
            data={"timestamp": datetime.now().isoformat()}
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "paused",
            "message": "Workflow paused successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to pause workflow: {str(e)}")


@router.post("/{session_id}/workflows/{workflow_id}/resume")
async def resume_workflow(
    session_id: str,
    workflow_id: str,
    background_tasks: BackgroundTasks,
    session: Dict[str, Any] = Depends(get_session)
) -> Dict[str, Any]:
    """Resume paused workflow execution"""
    try:
        # Load workflow state
        workflow_state = await workflow_state_manager.load_workflow_state(workflow_id)
        if not workflow_state:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        if workflow_state.status != WorkflowStatus.PAUSED:
            raise HTTPException(status_code=400, detail="Workflow is not paused")
        
        # Update status to running
        await workflow_state_manager.update_workflow_status(
            workflow_id=workflow_id,
            status=WorkflowStatus.RUNNING
        )
        
        # Get or recreate workflow instance
        workflow_instance = active_workflows.get(workflow_id)
        if not workflow_instance:
            repo_path = session.get("repo_path", "/unknown")
            workflow_instance = await _create_workflow_instance(
                session_id=session_id,
                repo_path=repo_path,
                config=workflow_state.config
            )
            active_workflows[workflow_id] = workflow_instance
        
        # Publish resume event
        await workflow_event_manager.publish_workflow_event(
            workflow_id=workflow_id,
            event_type="workflow_resumed",
            data={"timestamp": datetime.now().isoformat()}
        )
        
        return {
            "workflow_id": workflow_id,
            "status": "running",
            "message": "Workflow resumed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume workflow {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to resume workflow: {str(e)}")


@router.get("/{session_id}/workflows")
async def list_workflows(
    session_id: str,
    status: Optional[str] = Query(None, description="Filter by status"),
    session: Dict[str, Any] = Depends(get_session)
) -> List[Dict[str, Any]]:
    """List all workflows for a session"""
    try:
        # Get workflow IDs for session
        workflow_ids = await workflow_state_manager.get_session_workflows(session_id)
        
        workflows = []
        for workflow_id in workflow_ids:
            workflow_state = await workflow_state_manager.load_workflow_state(workflow_id)
            if workflow_state:
                # Filter by status if specified
                if status and workflow_state.status.value != status:
                    continue
                
                workflows.append({
                    "workflow_id": workflow_id,
                    "status": workflow_state.status.value,
                    "type": workflow_state.config.type.value,
                    "name": workflow_state.config.name,
                    "created_at": workflow_state.created_at.isoformat(),
                    "updated_at": workflow_state.updated_at.isoformat(),
                    "completed_at": workflow_state.completed_at.isoformat() if workflow_state.completed_at else None
                })
        
        return workflows
        
    except Exception as e:
        logger.error(f"Failed to list workflows for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list workflows: {str(e)}")


@router.get("/{session_id}/workflows/{workflow_id}")
async def get_workflow_details(
    session_id: str,
    workflow_id: str,
    include_history: bool = Query(True, description="Include execution history"),
    session: Dict[str, Any] = Depends(get_session)
) -> WorkflowResponse:
    """Get detailed workflow information"""
    try:
        # Load workflow state
        workflow_state = await workflow_state_manager.load_workflow_state(workflow_id)
        if not workflow_state:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        # Convert config to dict for response
        config_dict = {
            "id": workflow_state.config.id,
            "name": workflow_state.config.name,
            "type": workflow_state.config.type.value,
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "role": agent.role,
                    "specialization": agent.specialization
                }
                for agent in workflow_state.config.agents
            ],
            "entry_agent": workflow_state.config.entry_agent
        }
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            session_id=session_id,
            status=workflow_state.status.value,
            created_at=workflow_state.created_at.isoformat(),
            updated_at=workflow_state.updated_at.isoformat(),
            config=config_dict,
            execution_history=workflow_state.execution_history if include_history else [],
            result=workflow_state.result,
            error_message=workflow_state.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow details {workflow_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get workflow details: {str(e)}")


@router.websocket("/{session_id}/workflows/{workflow_id}/ws")
async def workflow_websocket(
    websocket: WebSocket,
    session_id: str,
    workflow_id: str
):
    """WebSocket endpoint for real-time workflow updates"""
    await websocket.accept()
    
    # Add to connections
    if workflow_id not in websocket_connections:
        websocket_connections[workflow_id] = []
    websocket_connections[workflow_id].append(websocket)
    
    # Subscribe to workflow events
    async def event_handler(event: Dict[str, Any]):
        try:
            await websocket.send_json(event)
        except Exception as e:
            logger.error(f"Failed to send WebSocket event: {e}")
    
    workflow_event_manager.subscribe_to_workflow(workflow_id, event_handler)
    
    try:
        # Send initial status
        workflow_state = await workflow_state_manager.load_workflow_state(workflow_id)
        if workflow_state:
            await websocket.send_json({
                "type": "workflow_status",
                "workflow_id": workflow_id,
                "status": workflow_state.status.value,
                "timestamp": datetime.now().isoformat()
            })
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages (keep-alive)
                message = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle client messages
                if message == "ping":
                    await websocket.send_text("pong")
                    
            except asyncio.TimeoutError:
                # Send keep-alive ping
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for workflow {workflow_id}")
    except Exception as e:
        logger.error(f"WebSocket error for workflow {workflow_id}: {e}")
    finally:
        # Clean up
        if workflow_id in websocket_connections:
            try:
                websocket_connections[workflow_id].remove(websocket)
                if not websocket_connections[workflow_id]:
                    del websocket_connections[workflow_id]
            except ValueError:
                pass
        
        workflow_event_manager.unsubscribe_from_workflow(workflow_id, event_handler)


# Helper functions

async def _create_workflow_config(
    workflow_id: str,
    session_id: str,
    workflow_type: str,
    name: Optional[str],
    agents: Optional[List[Dict[str, Any]]],
    entry_agent: Optional[str],
    config: Optional[Dict[str, Any]]
) -> WorkflowConfig:
    """Create workflow configuration from request"""
    
    # Parse workflow type
    try:
        wf_type = WorkflowType(workflow_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid workflow type: {workflow_type}")
    
    # Create default configurations for standard types
    if wf_type == WorkflowType.LINEAR_SWARM and not agents:
        return WorkflowConfig(
            id=workflow_id,
            name=name or "Linear Swarm Analysis",
            type=wf_type,
            agents=[
                AgentConfig(
                    id="code_analysis",
                    name="Code Analysis Specialist",
                    role="Code Analyzer",
                    goal="Analyze code structure and understand implementation",
                    backstory="Expert in code analysis and understanding complex codebases",
                    can_handoff_to=["issue_resolution", "testing_qa"],
                    specialization="code_analysis",
                    max_iterations=20
                ),
                AgentConfig(
                    id="issue_resolution",
                    name="Issue Resolution Specialist",
                    role="Problem Solver", 
                    goal="Identify and propose solutions for issues",
                    backstory="Experienced in debugging and issue resolution",
                    can_handoff_to=["testing_qa"],
                    specialization="issue_resolution",
                    max_iterations=15
                ),
                AgentConfig(
                    id="testing_qa",
                    name="Testing & QA Specialist",
                    role="Quality Assurance",
                    goal="Ensure code quality and proper testing",
                    backstory="Expert in testing strategies and quality assurance",
                    can_handoff_to=[],
                    specialization="testing_qa",
                    max_iterations=15
                )
            ],
            entry_agent=entry_agent or "code_analysis"
        )
    
    elif wf_type == WorkflowType.ORCHESTRATOR and not agents:
        return WorkflowConfig(
            id=workflow_id,
            name=name or "Orchestrated Analysis",
            type=wf_type,
            agents=[
                AgentConfig(
                    id="orchestrator",
                    name="Analysis Orchestrator",
                    role="Orchestrator",
                    goal="Coordinate analysis across multiple specialized agents",
                    backstory="Expert in managing complex analysis workflows",
                    can_handoff_to=["code_analysis", "security_specialist", "testing_qa"],
                    specialization="orchestration",
                    max_iterations=25
                ),
                AgentConfig(
                    id="code_analysis",
                    name="Code Analysis Agent",
                    role="Code Analyzer",
                    goal="Deep code analysis and understanding",
                    backstory="Specialized in code structure analysis",
                    can_handoff_to=["orchestrator"],
                    specialization="code_analysis",
                    max_iterations=20
                ),
                AgentConfig(
                    id="security_specialist",
                    name="Security Analysis Agent",
                    role="Security Specialist",
                    goal="Identify security vulnerabilities and risks",
                    backstory="Expert in security analysis and threat detection",
                    can_handoff_to=["orchestrator"],
                    specialization="security",
                    max_iterations=15
                ),
                AgentConfig(
                    id="testing_qa",
                    name="Quality Analysis Agent",
                    role="Quality Specialist",
                    goal="Assess code quality and best practices",
                    backstory="Expert in code quality and best practices",
                    can_handoff_to=["orchestrator"],
                    specialization="testing_qa",
                    max_iterations=15
                )
            ],
            entry_agent=entry_agent or "orchestrator"
        )
    
    # For custom configurations
    if agents:
        agent_configs = []
        for agent_data in agents:
            agent_config = AgentConfig(**agent_data)
            agent_configs.append(agent_config)
    else:
        raise HTTPException(status_code=400, detail="Custom workflow requires agent configurations")
    
    return WorkflowConfig(
        id=workflow_id,
        name=name or f"Custom {workflow_type} Workflow",
        type=wf_type,
        agents=agent_configs,
        entry_agent=entry_agent
    )


async def _create_workflow_instance(
    session_id: str,
    repo_path: str,
    config: WorkflowConfig
) -> TriageFlowAgentWorkflow:
    """Create workflow instance from configuration"""
    
    # Use the dependency system to get AgenticRAG which handles session loading
    from ..dependencies import get_agentic_rag
    
    try:
        # Use the dependency system which handles AgenticRAG creation/caching properly
        session = await session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        # Use get_agentic_rag dependency which handles the full initialization
        agentic_rag = await get_agentic_rag(session_id, session)
        
        if agentic_rag and hasattr(agentic_rag, 'agentic_explorer'):
            # Use the explorer from the agentic RAG system
            explorer = agentic_rag.agentic_explorer
            tools = None  # Will be created from explorer in workflow
            logger.info(f"Successfully retrieved AgenticRAG explorer for workflow {session_id}")
        else:
            # This should not happen with get_agentic_rag, but keep as fallback
            factory = WorkflowAgentFactory(session_id=session_id, repo_path=repo_path)
            tools = factory.tools
            explorer = None
            logger.warning(f"get_agentic_rag returned None for session {session_id}, using factory tools")
        
    except Exception as e:
        logger.error(f"Error getting AgenticRAG for workflow {session_id}: {e}")
        # Fallback to factory
        factory = WorkflowAgentFactory(session_id=session_id, repo_path=repo_path)
        tools = factory.tools
        explorer = None
    
    # Create workflow instance based on type
    if config.type == WorkflowType.LINEAR_SWARM:
        return LinearSwarmWorkflow(
            session_id=session_id,
            repo_path=repo_path,
            explorer=explorer,
            tools=tools
        )
    elif config.type == WorkflowType.ORCHESTRATOR:
        return OrchestratorWorkflow(
            session_id=session_id,
            repo_path=repo_path,
            explorer=explorer,
            tools=tools
        )
    elif config.type == WorkflowType.CUSTOM_PLANNER:
        return CustomPlannerWorkflow(
            session_id=session_id,
            repo_path=repo_path,
            custom_config=config,
            explorer=explorer,
            tools=tools
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported workflow type: {config.type}")


async def _execute_workflow_background(
    workflow_id: str,
    workflow_instance: TriageFlowAgentWorkflow,
    query: str,
    context: Dict[str, Any]
):
    """Execute workflow in background task"""
    try:
        logger.info(f"Starting background execution of workflow {workflow_id}")
        
        # Publish agent activation events
        async def event_publisher(event_data):
            await workflow_event_manager.publish_workflow_event(
                workflow_id=workflow_id,
                event_type=event_data.get("type", "workflow_update"),
                data=event_data
            )
        
        # Use event_publisher for workflow events
        await event_publisher({
            "type": "workflow_execution_started",
            "timestamp": datetime.now().isoformat(),
            "query": query
        })
        
        # Store execution context in workflow shared memory for status API
        await workflow_state_manager.update_shared_memory(
            workflow_id=workflow_id,
            key='execution_context',
            value=context
        )
        
        # Check if this should use comprehensive analysis workflow
        query_lower = query.lower()
        comprehensive_patterns = [
            "comprehensive", "security", "vulnerabilities", "analyze", "review",
            "entire repository", "codebase", "architecture", "full analysis",
            "performance", "quality", "dependencies"
        ]
        use_comprehensive = any(pattern in query_lower for pattern in comprehensive_patterns)
        
        if use_comprehensive:
            # Use the comprehensive analysis workflow
            from ...agent_tools.llamaindex_comprehensive_workflow import run_comprehensive_analysis
            from ...agent_tools.context_manager import ContextManager
            
            # Create context manager
            context_manager = ContextManager(None, None)
            
            # Determine focus areas from query
            focus_areas = []
            if any(word in query_lower for word in ["security", "vulnerabilities", "secure"]):
                focus_areas.append("security")
            if any(word in query_lower for word in ["dependencies", "deps", "requirements"]):
                focus_areas.append("dependencies")
            if any(word in query_lower for word in ["performance", "speed", "optimization"]):
                focus_areas.append("performance")
            if any(word in query_lower for word in ["quality", "code quality", "best practices"]):
                focus_areas.append("quality")
            
            # Default to security and dependencies if none specified
            if not focus_areas:
                focus_areas = ["security", "dependencies"]
            
            logger.info(f"Using comprehensive analysis workflow with focus areas: {focus_areas}")
            
            # Get session and repo_path
            workflow_state = await workflow_state_manager.load_workflow_state(workflow_id)
            session_id = workflow_state.session_id if workflow_state else 'unknown'
            
            # Try to get repo_path from session
            try:
                session = await session_manager.get_session(session_id)
                repo_path = session.get('repo_path', '/unknown') if session else '/unknown'
            except Exception:
                repo_path = '/unknown'
            
            result = await run_comprehensive_analysis(
                session_id=workflow_state.session_id if workflow_state else workflow_id,
                repo_path=repo_path,
                query=query,
                focus_areas=focus_areas,
                context_manager=context_manager
            )
        else:
            # Use standard workflow execution
            # Execute workflow using direct method call instead of run() to bypass StartEvent issues
            # Use set_query_and_context to set query/context AND update dynamic iterations
            # Start with 100 iterations for comprehensive repository analysis
            workflow_instance.set_query_and_context(query, context, manual_iterations=100)
            
            # Execute workflow directly without LlamaIndex's event system for now
            result = await workflow_instance._execute_workflow_directly(query, context)
        
        # Update workflow status on completion
        # Store the complete result as JSON for comprehensive analysis
        result_to_store = result if use_comprehensive else str(result)
        await workflow_state_manager.update_workflow_status(
            workflow_id=workflow_id,
            status=WorkflowStatus.COMPLETED,
            result=result_to_store
        )
        
        # Publish completion event
        await workflow_event_manager.publish_workflow_event(
            workflow_id=workflow_id,
            event_type="workflow_completed",
            data={
                "result": str(result),
                "execution_summary": workflow_instance._generate_execution_summary()
            }
        )
        
        logger.info(f"Completed workflow {workflow_id}")
        
    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")
        
        # Update workflow status on failure
        await workflow_state_manager.update_workflow_status(
            workflow_id=workflow_id,
            status=WorkflowStatus.FAILED,
            error_message=str(e)
        )
        
        # Publish failure event
        await workflow_event_manager.publish_workflow_event(
            workflow_id=workflow_id,
            event_type="workflow_failed",
            data={
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    finally:
        # Clean up active workflow
        if workflow_id in active_workflows:
            del active_workflows[workflow_id]


def _calculate_workflow_progress(workflow_state) -> Dict[str, Any]:
    """Calculate workflow progress based on execution history"""
    if not workflow_state.execution_history:
        return {"progress": 0, "current_step": "Starting"}
    
    total_agents = len(workflow_state.config.agents)
    executed_agents = len(set(
        entry.get("agent_id") 
        for entry in workflow_state.execution_history 
        if entry.get("type") == "agent_execution"
    ))
    
    progress = min(100, (executed_agents / total_agents) * 100) if total_agents > 0 else 0
    
    latest_entry = workflow_state.execution_history[-1]
    current_step = latest_entry.get("agent_id", "Unknown")
    
    return {
        "progress": progress,
        "current_step": current_step,
        "executed_agents": executed_agents,
        "total_agents": total_agents
    } 