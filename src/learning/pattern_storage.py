"""
Pattern Storage System for Self-Improving Agents

Stores and retrieves learned patterns that enable agents to improve over time.
"""

import json
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class LearnedPattern:
    """Represents a learned pattern that can be reused"""
    pattern_id: str
    pattern_type: str  # "query_pattern", "solution_pattern", "tool_pattern", "code_pattern"
    description: str
    context: Dict[str, Any]  # Repository, query type, etc.
    solution: Dict[str, Any]  # What worked
    success_rate: float
    usage_count: int
    created_at: datetime
    last_used: datetime
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.last_used, str):
            self.last_used = datetime.fromisoformat(self.last_used)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['last_used'] = self.last_used.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearnedPattern':
        """Create from dictionary"""
        return cls(**data)
    
    def update_usage(self, success: bool):
        """Update usage statistics"""
        self.usage_count += 1
        self.last_used = datetime.utcnow()
        
        # Update success rate using exponential moving average
        alpha = 0.1  # Learning rate
        new_success = 1.0 if success else 0.0
        self.success_rate = (1 - alpha) * self.success_rate + alpha * new_success

class PatternStorage:
    """
    Stores and retrieves learned patterns using Redis and vector similarity
    """
    
    def __init__(self, redis_client=None, session_id: str = None):
        self.redis_client = redis_client
        self.session_id = session_id
        self.local_cache = {}
        
    async def store_pattern(self, pattern: LearnedPattern) -> bool:
        """Store a learned pattern"""
        try:
            # Store in Redis if available
            if self.redis_client:
                key = f"pattern:{pattern.pattern_id}"
                data = json.dumps(pattern.to_dict())
                self.redis_client.setex(key, 86400 * 30, data)  # 30 days TTL
                
                # Add to pattern index by type
                type_key = f"patterns_by_type:{pattern.pattern_type}"
                self.redis_client.sadd(type_key, pattern.pattern_id)
                self.redis_client.expire(type_key, 86400 * 30)
                
                # Add to session patterns if session_id provided
                if self.session_id:
                    session_key = f"session_patterns:{self.session_id}"
                    self.redis_client.sadd(session_key, pattern.pattern_id)
                    self.redis_client.expire(session_key, 86400 * 7)
            
            # Store in local cache
            self.local_cache[pattern.pattern_id] = pattern
            
            logger.info(f"Stored pattern {pattern.pattern_id} of type {pattern.pattern_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store pattern {pattern.pattern_id}: {e}")
            return False
    
    async def get_pattern(self, pattern_id: str) -> Optional[LearnedPattern]:
        """Retrieve a specific pattern"""
        # Check local cache first
        if pattern_id in self.local_cache:
            return self.local_cache[pattern_id]
        
        # Check Redis
        if self.redis_client:
            try:
                key = f"pattern:{pattern_id}"
                data = self.redis_client.get(key)
                if data:
                    if isinstance(data, bytes):
                        data = data.decode()
                    pattern_data = json.loads(data)
                    pattern = LearnedPattern.from_dict(pattern_data)
                    
                    # Cache locally
                    self.local_cache[pattern_id] = pattern
                    return pattern
            except Exception as e:
                logger.error(f"Failed to retrieve pattern {pattern_id}: {e}")
        
        return None
    
    async def find_similar_patterns(self, 
                                   query: str,
                                   pattern_type: Optional[str] = None,
                                   context: Dict[str, Any] = None,
                                   limit: int = 5) -> List[LearnedPattern]:
        """Find patterns similar to the given query and context"""
        patterns = []
        
        try:
            # Get all patterns of the specified type
            if pattern_type and self.redis_client:
                type_key = f"patterns_by_type:{pattern_type}"
                pattern_ids = self.redis_client.smembers(type_key)
            else:
                # Get all patterns (this could be optimized with better indexing)
                pattern_ids = await self._get_all_pattern_ids()
            
            # Score and rank patterns
            scored_patterns = []
            for pattern_id in pattern_ids:
                if isinstance(pattern_id, bytes):
                    pattern_id = pattern_id.decode()
                
                pattern = await self.get_pattern(pattern_id)
                if pattern:
                    similarity_score = self._calculate_similarity(
                        query, pattern, context
                    )
                    if similarity_score > 0.3:  # Minimum similarity threshold
                        scored_patterns.append((similarity_score, pattern))
            
            # Sort by similarity and return top results
            scored_patterns.sort(key=lambda x: x[0], reverse=True)
            patterns = [pattern for _, pattern in scored_patterns[:limit]]
            
        except Exception as e:
            logger.error(f"Failed to find similar patterns: {e}")
        
        return patterns
    
    def _calculate_similarity(self, 
                            query: str, 
                            pattern: LearnedPattern,
                            context: Dict[str, Any] = None) -> float:
        """Calculate similarity between query/context and pattern"""
        similarity = 0.0
        
        # Text similarity (simple word overlap for now)
        query_words = set(query.lower().split())
        pattern_desc_words = set(pattern.description.lower().split())
        
        if query_words and pattern_desc_words:
            overlap = len(query_words.intersection(pattern_desc_words))
            text_similarity = overlap / len(query_words.union(pattern_desc_words))
            similarity += text_similarity * 0.4
        
        # Context similarity
        if context and pattern.context:
            context_similarity = self._calculate_context_similarity(
                context, pattern.context
            )
            similarity += context_similarity * 0.3
        
        # Success rate influence
        similarity += pattern.success_rate * 0.2
        
        # Recency bonus (patterns used recently are more relevant)
        days_since_use = (datetime.utcnow() - pattern.last_used).days
        recency_bonus = max(0, 1 - days_since_use / 30) * 0.1
        similarity += recency_bonus
        
        return min(similarity, 1.0)
    
    def _calculate_context_similarity(self, 
                                    context1: Dict[str, Any],
                                    context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        if not context1 or not context2:
            return 0.0
        
        similarity = 0.0
        total_keys = set(context1.keys()).union(set(context2.keys()))
        
        if not total_keys:
            return 0.0
        
        matching_keys = 0
        for key in total_keys:
            if key in context1 and key in context2:
                val1, val2 = context1[key], context2[key]
                
                if val1 == val2:
                    matching_keys += 1
                elif isinstance(val1, str) and isinstance(val2, str):
                    # Partial string matching
                    if val1.lower() in val2.lower() or val2.lower() in val1.lower():
                        matching_keys += 0.5
        
        return matching_keys / len(total_keys)
    
    async def _get_all_pattern_ids(self) -> List[str]:
        """Get all pattern IDs (for when we don't have type filtering)"""
        pattern_ids = []
        
        if self.redis_client:
            try:
                # This is not efficient for large datasets - should use better indexing
                keys = self.redis_client.keys("pattern:*")
                pattern_ids = [key.decode().split(":", 1)[1] if isinstance(key, bytes) 
                              else key.split(":", 1)[1] for key in keys]
            except Exception as e:
                logger.error(f"Failed to get all pattern IDs: {e}")
        
        # Add local cache patterns
        pattern_ids.extend(self.local_cache.keys())
        
        return list(set(pattern_ids))  # Remove duplicates
    
    async def create_pattern_from_interaction(self,
                                            query: str,
                                            response: str,
                                            tools_used: List[Dict[str, Any]],
                                            success: bool,
                                            context: Dict[str, Any] = None) -> Optional[LearnedPattern]:
        """Create a pattern from a successful interaction"""
        if not success:
            return None  # Only learn from successful interactions
        
        try:
            # Generate pattern ID
            pattern_data = f"{query}_{json.dumps(tools_used, sort_keys=True)}"
            pattern_id = hashlib.md5(pattern_data.encode()).hexdigest()
            
            # Determine pattern type
            pattern_type = self._determine_pattern_type(query, tools_used)
            
            # Extract solution information
            solution = {
                "response": response,
                "tools_used": tools_used,
                "approach": self._extract_approach(tools_used),
                "key_insights": self._extract_insights(response)
            }
            
            pattern = LearnedPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                description=self._generate_pattern_description(query, tools_used),
                context=context or {},
                solution=solution,
                success_rate=1.0,  # Start with 100% since it worked
                usage_count=1,
                created_at=datetime.utcnow(),
                last_used=datetime.utcnow(),
                metadata={
                    "query_complexity": self._calculate_query_complexity(query),
                    "tools_count": len(tools_used),
                    "response_length": len(response)
                }
            )
            
            await self.store_pattern(pattern)
            return pattern
            
        except Exception as e:
            logger.error(f"Failed to create pattern from interaction: {e}")
            return None
    
    def _determine_pattern_type(self, query: str, tools_used: List[Dict[str, Any]]) -> str:
        """Determine the type of pattern based on query and tools"""
        query_lower = query.lower()
        
        if any("search" in tool.get("tool", "") for tool in tools_used):
            return "search_pattern"
        elif any("generate" in tool.get("tool", "") or "create" in tool.get("tool", "") for tool in tools_used):
            return "generation_pattern"
        elif "analyze" in query_lower or "understand" in query_lower:
            return "analysis_pattern"
        elif "fix" in query_lower or "debug" in query_lower:
            return "debugging_pattern"
        else:
            return "general_pattern"
    
    def _extract_approach(self, tools_used: List[Dict[str, Any]]) -> str:
        """Extract the approach used from tools"""
        if not tools_used:
            return "direct_response"
        
        tool_sequence = " -> ".join([tool.get("tool", "unknown") for tool in tools_used])
        return tool_sequence
    
    def _extract_insights(self, response: str) -> List[str]:
        """Extract key insights from the response"""
        # Simple keyword-based extraction
        insights = []
        
        if "found" in response.lower():
            insights.append("successful_search")
        if "error" in response.lower():
            insights.append("error_handling")
        if "performance" in response.lower():
            insights.append("performance_consideration")
        if "security" in response.lower():
            insights.append("security_consideration")
        
        return insights
    
    def _generate_pattern_description(self, query: str, tools_used: List[Dict[str, Any]]) -> str:
        """Generate a description for the pattern"""
        tools_str = ", ".join([tool.get("tool", "unknown") for tool in tools_used])
        return f"Query pattern: '{query[:50]}...' using tools: {tools_str}"
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate complexity score for pattern metadata"""
        complexity = 0.0
        
        # Word count
        word_count = len(query.split())
        complexity += min(word_count / 20.0, 1.0) * 0.5
        
        # Complex keywords
        complex_words = ['analyze', 'comprehensive', 'architecture', 'relationship', 'dependency']
        complexity += sum(0.1 for word in complex_words if word in query.lower())
        
        return min(complexity, 1.0)
    
    async def get_success_patterns(self, 
                                  pattern_type: Optional[str] = None,
                                  min_success_rate: float = 0.7,
                                  min_usage_count: int = 2) -> List[LearnedPattern]:
        """Get patterns with high success rates"""
        all_patterns = await self.find_similar_patterns(
            "", pattern_type=pattern_type, limit=100
        )
        
        successful_patterns = [
            pattern for pattern in all_patterns
            if (pattern.success_rate >= min_success_rate and 
                pattern.usage_count >= min_usage_count)
        ]
        
        # Sort by success rate and usage count
        successful_patterns.sort(
            key=lambda p: (p.success_rate, p.usage_count), 
            reverse=True
        )
        
        return successful_patterns
    
    async def cleanup_old_patterns(self, days_threshold: int = 30):
        """Remove patterns that haven't been used in a while"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            # Get all patterns and check their last usage
            all_pattern_ids = await self._get_all_pattern_ids()
            
            removed_count = 0
            for pattern_id in all_pattern_ids:
                pattern = await self.get_pattern(pattern_id)
                if pattern and pattern.last_used < cutoff_date:
                    # Remove from Redis
                    if self.redis_client:
                        self.redis_client.delete(f"pattern:{pattern_id}")
                        
                        # Remove from type index
                        type_key = f"patterns_by_type:{pattern.pattern_type}"
                        self.redis_client.srem(type_key, pattern_id)
                    
                    # Remove from local cache
                    self.local_cache.pop(pattern_id, None)
                    removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old patterns")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old patterns: {e}")