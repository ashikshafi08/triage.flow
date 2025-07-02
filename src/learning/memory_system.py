"""
Memory System for Self-Improving Agents

Manages hierarchical memory including episodic memory for learning from interactions.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from .performance_tracker import PerformanceMetrics
from .pattern_storage import PatternStorage, LearnedPattern

logger = logging.getLogger(__name__)

@dataclass
class Episode:
    """Represents a complete interaction episode for learning"""
    episode_id: str
    session_id: str
    timestamp: datetime
    query: str
    context: Dict[str, Any]
    actions_taken: List[Dict[str, Any]]
    response: str
    performance_metrics: PerformanceMetrics
    outcome_quality: float
    learned_insights: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['performance_metrics'] = self.performance_metrics.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """Create from dictionary"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['performance_metrics'] = PerformanceMetrics.from_dict(data['performance_metrics'])
        return cls(**data)

class MemorySystem:
    """
    Hierarchical memory system that manages different types of agent memory
    """
    
    def __init__(self, session_id: str, redis_client=None):
        self.session_id = session_id
        self.redis_client = redis_client
        self.pattern_storage = PatternStorage(redis_client, session_id)
        
        # Local caches
        self.episode_cache = []
        self.working_memory = {}
        
    async def record_episode(self, episode: Episode) -> bool:
        """Record a complete interaction episode"""
        try:
            # Store episode
            await self._store_episode(episode)
            
            # Extract patterns if successful
            if episode.performance_metrics.success and episode.outcome_quality > 0.7:
                pattern = await self.pattern_storage.create_pattern_from_interaction(
                    query=episode.query,
                    response=episode.response,
                    tools_used=episode.actions_taken,
                    success=True,
                    context=episode.context
                )
                
                if pattern:
                    logger.info(f"Created pattern {pattern.pattern_id} from episode {episode.episode_id}")
            
            # Update working memory with insights
            await self._update_working_memory(episode)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record episode {episode.episode_id}: {e}")
            return False
    
    async def _store_episode(self, episode: Episode):
        """Store episode in Redis and local cache"""
        # Add to local cache
        self.episode_cache.append(episode)
        
        # Keep only last 50 episodes in cache
        if len(self.episode_cache) > 50:
            self.episode_cache = self.episode_cache[-50:]
        
        # Store in Redis if available
        if self.redis_client:
            key = f"episode:{self.session_id}:{episode.episode_id}"
            data = json.dumps(episode.to_dict())
            await self.redis_client.setex(key, 86400 * 7, data)  # 7 days TTL
            
            # Add to session episode index
            session_key = f"session_episodes:{self.session_id}"
            await self.redis_client.lpush(session_key, episode.episode_id)
            await self.redis_client.ltrim(session_key, 0, 199)  # Keep last 200
            await self.redis_client.expire(session_key, 86400 * 14)  # 14 days
    
    async def _update_working_memory(self, episode: Episode):
        """Update working memory with insights from the episode"""
        # Track successful strategies
        if episode.performance_metrics.success:
            strategy_key = f"successful_strategy_{len(episode.actions_taken)}_tools"
            if strategy_key not in self.working_memory:
                self.working_memory[strategy_key] = []
            
            self.working_memory[strategy_key].append({
                "tools": [action.get("tool", "unknown") for action in episode.actions_taken],
                "query_type": self._classify_query(episode.query),
                "success_rate": episode.outcome_quality,
                "timestamp": episode.timestamp.isoformat()
            })
            
            # Keep only recent strategies
            if len(self.working_memory[strategy_key]) > 10:
                self.working_memory[strategy_key] = self.working_memory[strategy_key][-10:]
        
        # Track error patterns
        if not episode.performance_metrics.success:
            error_key = "error_patterns"
            if error_key not in self.working_memory:
                self.working_memory[error_key] = []
            
            self.working_memory[error_key].append({
                "query": episode.query,
                "actions": episode.actions_taken,
                "error_count": episode.performance_metrics.error_count,
                "timestamp": episode.timestamp.isoformat()
            })
            
            if len(self.working_memory[error_key]) > 20:
                self.working_memory[error_key] = self.working_memory[error_key][-20:]
    
    def _classify_query(self, query: str) -> str:
        """Classify query type for pattern matching"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["search", "find", "locate"]):
            return "search"
        elif any(word in query_lower for word in ["analyze", "understand", "explain"]):
            return "analysis"
        elif any(word in query_lower for word in ["create", "generate", "write"]):
            return "generation"
        elif any(word in query_lower for word in ["fix", "debug", "solve"]):
            return "debugging"
        elif any(word in query_lower for word in ["compare", "difference", "versus"]):
            return "comparison"
        else:
            return "general"
    
    async def get_relevant_memories(self, 
                                   query: str,
                                   context: Dict[str, Any] = None,
                                   limit: int = 5) -> Dict[str, Any]:
        """Get relevant memories for the current query"""
        memories = {
            "similar_patterns": [],
            "successful_strategies": [],
            "error_patterns": [],
            "working_memory_insights": {}
        }
        
        try:
            # Get similar patterns
            similar_patterns = await self.pattern_storage.find_similar_patterns(
                query=query,
                context=context,
                limit=limit
            )
            memories["similar_patterns"] = [p.to_dict() for p in similar_patterns]
            
            # Get successful strategies from working memory
            query_type = self._classify_query(query)
            for key, strategies in self.working_memory.items():
                if "successful_strategy" in key:
                    relevant_strategies = [
                        s for s in strategies 
                        if s.get("query_type") == query_type
                    ]
                    if relevant_strategies:
                        memories["successful_strategies"].extend(relevant_strategies[-3:])
            
            # Get relevant error patterns to avoid
            if "error_patterns" in self.working_memory:
                recent_errors = self.working_memory["error_patterns"][-5:]
                relevant_errors = [
                    error for error in recent_errors
                    if self._queries_similar(query, error.get("query", ""))
                ]
                memories["error_patterns"] = relevant_errors
            
            # Include current working memory insights
            memories["working_memory_insights"] = {
                k: v for k, v in self.working_memory.items()
                if not k.startswith("successful_strategy") and k != "error_patterns"
            }
            
        except Exception as e:
            logger.error(f"Failed to get relevant memories: {e}")
        
        return memories
    
    def _queries_similar(self, query1: str, query2: str) -> bool:
        """Check if two queries are similar"""
        words1 = set(query1.lower().split())
        words2 = set(query2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        similarity = overlap / len(words1.union(words2))
        
        return similarity > 0.3
    
    async def extract_session_patterns(self) -> List[LearnedPattern]:
        """Extract all patterns learned in this session"""
        patterns = []
        
        try:
            # Get all successful episodes from this session
            successful_episodes = [
                ep for ep in self.episode_cache
                if ep.performance_metrics.success and ep.outcome_quality > 0.6
            ]
            
            # Create patterns from successful episodes
            for episode in successful_episodes:
                pattern = await self.pattern_storage.create_pattern_from_interaction(
                    query=episode.query,
                    response=episode.response,
                    tools_used=episode.actions_taken,
                    success=True,
                    context=episode.context
                )
                if pattern:
                    patterns.append(pattern)
            
        except Exception as e:
            logger.error(f"Failed to extract session patterns: {e}")
        
        return patterns
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about what the agent has learned"""
        insights = {
            "total_episodes": len(self.episode_cache),
            "success_rate": 0.0,
            "common_patterns": [],
            "improvement_areas": [],
            "learning_velocity": 0.0
        }
        
        try:
            if self.episode_cache:
                # Calculate success rate
                successful = sum(1 for ep in self.episode_cache if ep.performance_metrics.success)
                insights["success_rate"] = successful / len(self.episode_cache)
                
                # Identify common patterns
                tool_usage = {}
                for episode in self.episode_cache:
                    if episode.performance_metrics.success:
                        tools = [action.get("tool", "unknown") for action in episode.actions_taken]
                        tool_sequence = " -> ".join(tools)
                        tool_usage[tool_sequence] = tool_usage.get(tool_sequence, 0) + 1
                
                common_patterns = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]
                insights["common_patterns"] = [
                    {"pattern": pattern, "usage_count": count}
                    for pattern, count in common_patterns
                ]
                
                # Identify improvement areas
                failed_episodes = [ep for ep in self.episode_cache if not ep.performance_metrics.success]
                if failed_episodes:
                    error_types = {}
                    for episode in failed_episodes:
                        query_type = self._classify_query(episode.query)
                        error_types[query_type] = error_types.get(query_type, 0) + 1
                    
                    insights["improvement_areas"] = [
                        {"area": area, "failure_count": count}
                        for area, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)
                    ]
                
                # Calculate learning velocity (improvement over time)
                if len(self.episode_cache) >= 10:
                    recent_success = sum(1 for ep in self.episode_cache[-5:] if ep.performance_metrics.success) / 5
                    older_success = sum(1 for ep in self.episode_cache[-10:-5] if ep.performance_metrics.success) / 5
                    insights["learning_velocity"] = recent_success - older_success
            
        except Exception as e:
            logger.error(f"Failed to get learning insights: {e}")
        
        return insights
    
    async def cleanup_old_memories(self, days_threshold: int = 14):
        """Clean up old episodic memories"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            # Clean local cache
            self.episode_cache = [
                ep for ep in self.episode_cache
                if ep.timestamp >= cutoff_date
            ]
            
            # Clean Redis episodes
            if self.redis_client:
                session_key = f"session_episodes:{self.session_id}"
                episode_ids = await self.redis_client.lrange(session_key, 0, -1)
                
                removed_count = 0
                for episode_id in episode_ids:
                    if isinstance(episode_id, bytes):
                        episode_id = episode_id.decode()
                    
                    key = f"episode:{self.session_id}:{episode_id}"
                    episode_data = await self.redis_client.get(key)
                    
                    if episode_data:
                        if isinstance(episode_data, bytes):
                            episode_data = episode_data.decode()
                        episode_dict = json.loads(episode_data)
                        episode_time = datetime.fromisoformat(episode_dict['timestamp'])
                        
                        if episode_time < cutoff_date:
                            await self.redis_client.delete(key)
                            await self.redis_client.lrem(session_key, 1, episode_id)
                            removed_count += 1
                
                logger.info(f"Cleaned up {removed_count} old episodes")
            
            # Clean up pattern storage
            await self.pattern_storage.cleanup_old_patterns(days_threshold)
            
        except Exception as e:
            logger.error(f"Failed to cleanup old memories: {e}")