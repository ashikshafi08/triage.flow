"""
Workflow state management with Redis persistence
Part of LlamaIndex Workflow Integration for triage.flow
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

from ..config import settings
from .llamaindex_workflows import WorkflowConfig, AgentConfig

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowState:
    """State of a workflow instance"""
    workflow_id: str
    session_id: str
    config: WorkflowConfig
    status: WorkflowStatus
    current_agent: Optional[str] = None
    execution_history: List[Dict[str, Any]] = None
    shared_memory: Dict[str, Any] = None
    created_at: datetime = None
    updated_at: datetime = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result: Optional[Any] = None
    
    def __post_init__(self):
        if self.execution_history is None:
            self.execution_history = []
        if self.shared_memory is None:
            self.shared_memory = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class WorkflowStateManager:
    """Manages workflow state with Redis persistence and in-memory fallback"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self._memory_store: Dict[str, WorkflowState] = {}
        self._session_index: Dict[str, set] = {}
        self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection"""
        if not self.redis_client:
            try:
                # Try to get Redis client from existing cache system
                from ..cache.redis_cache_manager import get_redis_client
                self.redis_client = get_redis_client()
                logger.info("Workflow state manager connected to Redis")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for workflow state: {e}")
                self.redis_client = None
    
    async def create_workflow(
        self, 
        workflow_id: str, 
        session_id: str, 
        config: WorkflowConfig
    ) -> WorkflowState:
        """Create a new workflow state"""
        state = WorkflowState(
            workflow_id=workflow_id,
            session_id=session_id,
            config=config,
            status=WorkflowStatus.CREATED
        )
        
        await self.save_workflow_state(state)
        logger.info(f"Created workflow state: {workflow_id}")
        return state
    
    async def save_workflow_state(self, state: WorkflowState) -> None:
        """Save workflow state to Redis with in-memory fallback"""
        state.updated_at = datetime.now()
        
        # Always save to memory for immediate access
        self._memory_store[state.workflow_id] = state
        if state.session_id not in self._session_index:
            self._session_index[state.session_id] = set()
        self._session_index[state.session_id].add(state.workflow_id)
        
        if not self.redis_client:
            logger.debug(f"Saved workflow state to memory: {state.workflow_id}")
            return
        
        try:
            # Convert to JSON-serializable format
            state_data = self._serialize_state(state)
            
            # Save to Redis with TTL
            key = f"workflow_state:{state.workflow_id}"
            ttl = 86400 * 7  # 7 days
            
            await self.redis_client.setex(
                key,
                ttl,
                json.dumps(state_data, default=str)
            )
            
            # Also maintain session index
            session_key = f"session_workflows:{state.session_id}"
            await self.redis_client.sadd(session_key, state.workflow_id)
            await self.redis_client.expire(session_key, ttl)
            
            logger.debug(f"Saved workflow state to Redis: {state.workflow_id}")
            
        except Exception as e:
            logger.error(f"Failed to save workflow state to Redis {state.workflow_id}: {e}")
            logger.debug(f"Using in-memory fallback for {state.workflow_id}")
    
    async def load_workflow_state(self, workflow_id: str) -> Optional[WorkflowState]:
        """Load workflow state from Redis with in-memory fallback"""
        # First check memory store
        if workflow_id in self._memory_store:
            logger.debug(f"Loaded workflow state from memory: {workflow_id}")
            return self._memory_store[workflow_id]
        
        if not self.redis_client:
            logger.debug(f"Workflow state not found in memory: {workflow_id}")
            return None
        
        try:
            key = f"workflow_state:{workflow_id}"
            data = await self.redis_client.get(key)
            
            if not data:
                logger.debug(f"Workflow state not found in Redis: {workflow_id}")
                return None
            
            state_data = json.loads(data)
            state = self._deserialize_state(state_data)
            
            # Cache in memory for faster access
            self._memory_store[workflow_id] = state
            if state.session_id not in self._session_index:
                self._session_index[state.session_id] = set()
            self._session_index[state.session_id].add(workflow_id)
            
            logger.debug(f"Loaded workflow state from Redis: {workflow_id}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load workflow state from Redis {workflow_id}: {e}")
            return None
    
    async def update_workflow_status(
        self, 
        workflow_id: str, 
        status: WorkflowStatus,
        error_message: Optional[str] = None,
        result: Optional[Any] = None
    ) -> bool:
        """Update workflow status"""
        state = await self.load_workflow_state(workflow_id)
        if not state:
            logger.error(f"Cannot update status, workflow not found: {workflow_id}")
            return False
        
        state.status = status
        if error_message:
            state.error_message = error_message
        if result is not None:
            state.result = result
        if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
            state.completed_at = datetime.now()
        
        await self.save_workflow_state(state)
        return True
    
    async def add_execution_entry(
        self, 
        workflow_id: str, 
        entry: Dict[str, Any]
    ) -> bool:
        """Add entry to workflow execution history"""
        state = await self.load_workflow_state(workflow_id)
        if not state:
            return False
        
        entry["timestamp"] = datetime.now().isoformat()
        state.execution_history.append(entry)
        
        await self.save_workflow_state(state)
        return True
    
    async def update_shared_memory(
        self, 
        workflow_id: str, 
        key: str, 
        value: Any
    ) -> bool:
        """Update shared memory for workflow"""
        state = await self.load_workflow_state(workflow_id)
        if not state:
            return False
        
        state.shared_memory[key] = value
        await self.save_workflow_state(state)
        return True
    
    async def get_session_workflows(self, session_id: str) -> List[str]:
        """Get all workflow IDs for a session"""
        # First check memory store
        memory_workflows = list(self._session_index.get(session_id, set()))
        
        if not self.redis_client:
            return memory_workflows
        
        try:
            session_key = f"session_workflows:{session_id}"
            workflow_ids = await self.redis_client.smembers(session_key)
            redis_workflows = list(workflow_ids) if workflow_ids else []
            
            # Combine memory and Redis workflows, removing duplicates
            all_workflows = list(set(memory_workflows + redis_workflows))
            return all_workflows
        except Exception as e:
            logger.error(f"Failed to get session workflows from Redis {session_id}: {e}")
            return memory_workflows
    
    async def cleanup_expired_workflows(self) -> int:
        """Clean up expired workflow states"""
        if not self.redis_client:
            return 0
        
        try:
            # Find all workflow state keys
            pattern = "workflow_state:*"
            keys = await self.redis_client.keys(pattern)
            
            cleaned = 0
            for key in keys:
                ttl = await self.redis_client.ttl(key)
                if ttl == -1:  # No expiry set
                    # Set default expiry of 7 days
                    await self.redis_client.expire(key, 86400 * 7)
                elif ttl == -2:  # Key doesn't exist
                    cleaned += 1
            
            logger.info(f"Cleaned up {cleaned} expired workflow states")
            return cleaned
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired workflows: {e}")
            return 0
    
    def _serialize_state(self, state: WorkflowState) -> Dict[str, Any]:
        """Convert WorkflowState to JSON-serializable format"""
        # Convert dataclass to dict
        state_dict = asdict(state)
        
        # Handle enum serialization
        state_dict["status"] = state.status.value
        
        # Handle config serialization (WorkflowConfig is also a dataclass)
        if state.config:
            state_dict["config"] = asdict(state.config)
            # Handle nested enum in config
            if state.config.type:
                state_dict["config"]["type"] = state.config.type.value
        
        return state_dict
    
    def _deserialize_state(self, state_data: Dict[str, Any]) -> WorkflowState:
        """Convert JSON data back to WorkflowState"""
        from .llamaindex_workflows import WorkflowType
        
        # Handle status deserialization
        if "status" in state_data:
            state_data["status"] = WorkflowStatus(state_data["status"])
        
        # Handle config deserialization
        if "config" in state_data and state_data["config"]:
            config_data = state_data["config"]
            
            # Handle WorkflowType enum
            if "type" in config_data:
                config_data["type"] = WorkflowType(config_data["type"])
            
            # Convert agents list back to AgentConfig objects
            if "agents" in config_data:
                agents = []
                for agent_data in config_data["agents"]:
                    agent = AgentConfig(**agent_data)
                    agents.append(agent)
                config_data["agents"] = agents
            
            # Create WorkflowConfig object
            state_data["config"] = WorkflowConfig(**config_data)
        
        # Handle datetime deserialization
        for field in ["created_at", "updated_at", "completed_at"]:
            if field in state_data and state_data[field]:
                if isinstance(state_data[field], str):
                    state_data[field] = datetime.fromisoformat(state_data[field])
        
        return WorkflowState(**state_data)


class WorkflowEventManager:
    """Manages workflow events for real-time updates"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
        self._init_redis()
        self.event_subscribers: Dict[str, List[Callable]] = {}
    
    def _init_redis(self):
        """Initialize Redis connection"""
        if not self.redis_client:
            try:
                from ..cache.redis_cache_manager import get_redis_client
                self.redis_client = get_redis_client()
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for events: {e}")
                self.redis_client = None
    
    async def publish_workflow_event(
        self, 
        workflow_id: str, 
        event_type: str, 
        data: Dict[str, Any]
    ) -> None:
        """Publish workflow event"""
        event = {
            "workflow_id": workflow_id,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Publish to Redis if available
        if self.redis_client:
            try:
                channel = f"workflow_events:{workflow_id}"
                await self.redis_client.publish(
                    channel,
                    json.dumps(event, default=str)
                )
            except Exception as e:
                logger.error(f"Failed to publish workflow event: {e}")
        
        # Call local subscribers
        subscribers = self.event_subscribers.get(workflow_id, [])
        for callback in subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")
    
    def subscribe_to_workflow(
        self, 
        workflow_id: str, 
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Subscribe to workflow events"""
        if workflow_id not in self.event_subscribers:
            self.event_subscribers[workflow_id] = []
        
        self.event_subscribers[workflow_id].append(callback)
    
    def unsubscribe_from_workflow(
        self, 
        workflow_id: str, 
        callback: Callable[[Dict[str, Any]], None]
    ) -> None:
        """Unsubscribe from workflow events"""
        if workflow_id in self.event_subscribers:
            try:
                self.event_subscribers[workflow_id].remove(callback)
            except ValueError:
                pass
    
    async def get_workflow_events(
        self, 
        workflow_id: str, 
        since: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get recent workflow events"""
        # This is a simplified implementation
        # In production, you might want to store events in Redis with timestamps
        return []


# Global instances
workflow_state_manager = WorkflowStateManager()
workflow_event_manager = WorkflowEventManager() 