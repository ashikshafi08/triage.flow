"""
Learning Loop for Self-Improving Agents

Implements the continuous learning loop that analyzes performance and improves agent behavior.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from .performance_tracker import PerformanceTracker
from .memory_system import MemorySystem
from .pattern_storage import PatternStorage

logger = logging.getLogger(__name__)

class LearningLoop:
    """
    Manages the continuous learning process for self-improving agents
    """
    
    def __init__(self, 
                 session_id: str,
                 performance_tracker: PerformanceTracker,
                 memory_system: MemorySystem,
                 redis_client=None):
        self.session_id = session_id
        self.performance_tracker = performance_tracker
        self.memory_system = memory_system
        self.redis_client = redis_client
        
        self.learning_active = False
        self.learning_task = None
        self.learning_interval = 300  # 5 minutes
        
        # Learning configuration
        self.config = {
            "min_interactions_for_learning": 5,
            "performance_threshold": 0.7,
            "pattern_reuse_threshold": 0.8,
            "learning_trigger_conditions": [
                "low_success_rate",
                "declining_performance", 
                "high_response_time",
                "frequent_errors"
            ]
        }
    
    async def start_learning(self):
        """Start the continuous learning loop"""
        if self.learning_active:
            logger.warning(f"Learning loop already active for session {self.session_id}")
            return
        
        self.learning_active = True
        self.learning_task = asyncio.create_task(self._learning_loop())
        logger.info(f"Started learning loop for session {self.session_id}")
    
    async def stop_learning(self):
        """Stop the continuous learning loop"""
        self.learning_active = False
        if self.learning_task:
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped learning loop for session {self.session_id}")
    
    async def _learning_loop(self):
        """Main learning loop that runs continuously"""
        while self.learning_active:
            try:
                await asyncio.sleep(self.learning_interval)
                
                if not self.learning_active:
                    break
                
                # Check if learning should be triggered
                should_learn = await self._should_trigger_learning()
                
                if should_learn:
                    logger.info(f"Triggering learning for session {self.session_id}")
                    await self._perform_learning_cycle()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning loop for session {self.session_id}: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _should_trigger_learning(self) -> bool:
        """Determine if learning should be triggered"""
        try:
            # Get recent performance summary
            summary = await self.performance_tracker.get_performance_summary(days=1)
            
            if summary["total_interactions"] < self.config["min_interactions_for_learning"]:
                return False
            
            # Check trigger conditions
            triggers = []
            
            # Low success rate
            if summary["success_rate"] < self.config["performance_threshold"]:
                triggers.append("low_success_rate")
            
            # Declining performance
            if summary["improvement_trend"] == "declining":
                triggers.append("declining_performance")
            
            # High response time
            if summary["avg_response_time"] > 30.0:
                triggers.append("high_response_time")
            
            # Check if any trigger conditions are met
            active_triggers = [t for t in triggers if t in self.config["learning_trigger_conditions"]]
            
            if active_triggers:
                logger.info(f"Learning triggered by: {', '.join(active_triggers)}")
                return True
            
            # Also trigger learning periodically even if performance is good
            # to discover new patterns
            last_learning = await self._get_last_learning_time()
            if last_learning:
                time_since_learning = datetime.now() - last_learning
                if time_since_learning > timedelta(hours=6):  # Learn every 6 hours
                    logger.info("Periodic learning triggered")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking learning triggers: {e}")
            return False
    
    async def _perform_learning_cycle(self):
        """Perform a complete learning cycle"""
        try:
            # 1. Analyze recent performance
            performance_analysis = await self._analyze_performance()
            
            # 2. Extract new patterns from recent interactions
            new_patterns = await self._extract_new_patterns()
            
            # 3. Identify improvement opportunities
            improvements = await self._identify_improvements(performance_analysis)
            
            # 4. Generate optimization strategies
            strategies = await self._generate_optimization_strategies(improvements)
            
            # 5. Update agent recommendations
            recommendations = await self._create_agent_recommendations(strategies)
            
            # 6. Store learning results
            await self._store_learning_results({
                "timestamp": datetime.now().isoformat(),
                "performance_analysis": performance_analysis,
                "new_patterns_count": len(new_patterns),
                "improvements_identified": improvements,
                "strategies_generated": strategies,
                "recommendations": recommendations
            })
            
            logger.info(f"Learning cycle completed for session {self.session_id}: "
                       f"{len(new_patterns)} patterns, {len(improvements)} improvements")
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {e}")
    
    async def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze recent performance to identify patterns"""
        analysis = {
            "overall_performance": {},
            "tool_effectiveness": {},
            "query_type_performance": {},
            "time_based_patterns": {},
            "error_analysis": {}
        }
        
        try:
            # Get recent metrics
            metrics = await self.performance_tracker.get_recent_metrics(days=3)
            
            if not metrics:
                return analysis
            
            # Overall performance
            total = len(metrics)
            successful = sum(1 for m in metrics if m.success)
            analysis["overall_performance"] = {
                "success_rate": successful / total,
                "avg_response_time": sum(m.response_time for m in metrics) / total,
                "avg_tokens": sum(m.tokens_used for m in metrics) / total,
                "total_interactions": total
            }
            
            # Tool effectiveness
            tool_stats = {}
            for metric in metrics:
                for tool_call in metric.tool_calls:
                    tool_name = tool_call.get("tool", "unknown")
                    if tool_name not in tool_stats:
                        tool_stats[tool_name] = {"total": 0, "successful": 0, "avg_duration": 0}
                    
                    tool_stats[tool_name]["total"] += 1
                    if tool_call.get("success", False):
                        tool_stats[tool_name]["successful"] += 1
                    tool_stats[tool_name]["avg_duration"] += tool_call.get("duration", 0)
            
            for tool, stats in tool_stats.items():
                if stats["total"] > 0:
                    stats["success_rate"] = stats["successful"] / stats["total"]
                    stats["avg_duration"] = stats["avg_duration"] / stats["total"]
            
            analysis["tool_effectiveness"] = tool_stats
            
            # Query type performance (simplified classification)
            query_performance = {}
            for metric in metrics:
                query_type = self._classify_query_simple(metric.query)
                if query_type not in query_performance:
                    query_performance[query_type] = {"total": 0, "successful": 0}
                
                query_performance[query_type]["total"] += 1
                if metric.success:
                    query_performance[query_type]["successful"] += 1
            
            for query_type, stats in query_performance.items():
                if stats["total"] > 0:
                    stats["success_rate"] = stats["successful"] / stats["total"]
            
            analysis["query_type_performance"] = query_performance
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
        
        return analysis
    
    def _classify_query_simple(self, query: str) -> str:
        """Simple query classification"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["search", "find"]):
            return "search"
        elif any(word in query_lower for word in ["analyze", "understand"]):
            return "analysis"
        elif any(word in query_lower for word in ["create", "generate"]):
            return "generation"
        elif any(word in query_lower for word in ["fix", "debug"]):
            return "debugging"
        else:
            return "general"
    
    async def _extract_new_patterns(self) -> List[Dict[str, Any]]:
        """Extract new patterns from recent successful interactions"""
        patterns = []
        
        try:
            # Get recent successful metrics
            metrics = await self.performance_tracker.get_recent_metrics(days=1)
            successful_metrics = [m for m in metrics if m.success and m.goal_achievement > 0.7]
            
            for metric in successful_metrics:
                # Check if this interaction represents a new pattern
                pattern_candidate = {
                    "query": metric.query,
                    "tools_used": [call.get("tool") for call in metric.tool_calls],
                    "response_time": metric.response_time,
                    "complexity": metric.query_complexity,
                    "success_indicators": {
                        "goal_achievement": metric.goal_achievement,
                        "low_error_rate": metric.error_count == 0,
                        "efficient_execution": metric.response_time < 15.0
                    }
                }
                
                # Only add if it's a novel or highly effective pattern
                if await self._is_novel_pattern(pattern_candidate):
                    patterns.append(pattern_candidate)
            
        except Exception as e:
            logger.error(f"Error extracting new patterns: {e}")
        
        return patterns
    
    async def _is_novel_pattern(self, pattern_candidate: Dict[str, Any]) -> bool:
        """Check if a pattern candidate is novel or significantly better"""
        try:
            # Look for similar existing patterns
            similar_patterns = await self.memory_system.pattern_storage.find_similar_patterns(
                query=pattern_candidate["query"],
                limit=3
            )
            
            if not similar_patterns:
                return True  # Novel pattern
            
            # Check if this pattern is significantly better than existing ones
            for existing_pattern in similar_patterns:
                if (existing_pattern.success_rate > 0.8 and 
                    existing_pattern.usage_count > 2):
                    return False  # We already have a good pattern for this
            
            return True  # Better than existing patterns
            
        except Exception as e:
            logger.error(f"Error checking pattern novelty: {e}")
            return False
    
    async def _identify_improvements(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific areas for improvement"""
        improvements = []
        
        try:
            # Check overall performance
            overall = analysis.get("overall_performance", {})
            if overall.get("success_rate", 1.0) < 0.7:
                improvements.append({
                    "type": "success_rate",
                    "current_value": overall.get("success_rate", 0),
                    "target_value": 0.8,
                    "priority": "high",
                    "description": "Success rate below acceptable threshold"
                })
            
            # Check response time
            if overall.get("avg_response_time", 0) > 20.0:
                improvements.append({
                    "type": "response_time",
                    "current_value": overall.get("avg_response_time", 0),
                    "target_value": 15.0,
                    "priority": "medium",
                    "description": "Response time too high"
                })
            
            # Check tool effectiveness
            tool_effectiveness = analysis.get("tool_effectiveness", {})
            for tool_name, stats in tool_effectiveness.items():
                if stats.get("success_rate", 1.0) < 0.6 and stats.get("total", 0) > 3:
                    improvements.append({
                        "type": "tool_effectiveness",
                        "tool": tool_name,
                        "current_value": stats.get("success_rate", 0),
                        "target_value": 0.8,
                        "priority": "medium",
                        "description": f"Tool {tool_name} has low success rate"
                    })
            
            # Check query type performance
            query_performance = analysis.get("query_type_performance", {})
            for query_type, stats in query_performance.items():
                if stats.get("success_rate", 1.0) < 0.6 and stats.get("total", 0) > 2:
                    improvements.append({
                        "type": "query_type_performance", 
                        "query_type": query_type,
                        "current_value": stats.get("success_rate", 0),
                        "target_value": 0.8,
                        "priority": "medium",
                        "description": f"Poor performance on {query_type} queries"
                    })
            
        except Exception as e:
            logger.error(f"Error identifying improvements: {e}")
        
        return improvements
    
    async def _generate_optimization_strategies(self, improvements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate strategies to address identified improvements"""
        strategies = []
        
        for improvement in improvements:
            strategy = None
            
            if improvement["type"] == "success_rate":
                strategy = {
                    "type": "increase_validation",
                    "description": "Add more validation steps and error handling",
                    "actions": [
                        "Use more reliable tools for critical operations",
                        "Add fallback strategies for failed operations",
                        "Increase validation of tool outputs"
                    ]
                }
            
            elif improvement["type"] == "response_time":
                strategy = {
                    "type": "optimize_performance",
                    "description": "Reduce response time through optimization",
                    "actions": [
                        "Prioritize faster tools for simple queries",
                        "Use parallel tool execution where possible",
                        "Cache frequent query results"
                    ]
                }
            
            elif improvement["type"] == "tool_effectiveness":
                strategy = {
                    "type": "improve_tool_usage",
                    "description": f"Improve effectiveness of {improvement.get('tool', 'unknown')} tool",
                    "actions": [
                        f"Analyze failure patterns for {improvement.get('tool', 'unknown')}",
                        "Add pre-conditions before using this tool",
                        "Consider alternative tools for similar tasks"
                    ]
                }
            
            elif improvement["type"] == "query_type_performance":
                strategy = {
                    "type": "specialize_for_query_type",
                    "description": f"Improve handling of {improvement.get('query_type', 'unknown')} queries",
                    "actions": [
                        f"Create specialized tool sequences for {improvement.get('query_type', 'unknown')} queries",
                        "Add query-specific validation",
                        "Use patterns from successful similar queries"
                    ]
                }
            
            if strategy:
                strategy["priority"] = improvement["priority"]
                strategy["target_improvement"] = improvement
                strategies.append(strategy)
        
        return strategies
    
    async def _create_agent_recommendations(self, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create actionable recommendations for the agent"""
        recommendations = {
            "immediate_actions": [],
            "strategy_adjustments": [],
            "tool_preferences": {},
            "query_handling_updates": {}
        }
        
        try:
            for strategy in strategies:
                if strategy["priority"] == "high":
                    recommendations["immediate_actions"].extend(strategy["actions"])
                else:
                    recommendations["strategy_adjustments"].extend(strategy["actions"])
                
                # Create tool preferences based on effectiveness
                if strategy["type"] == "improve_tool_usage":
                    tool_name = strategy["target_improvement"].get("tool")
                    if tool_name:
                        recommendations["tool_preferences"][tool_name] = "use_with_caution"
                
                # Create query handling updates
                if strategy["type"] == "specialize_for_query_type":
                    query_type = strategy["target_improvement"].get("query_type")
                    if query_type:
                        recommendations["query_handling_updates"][query_type] = {
                            "needs_improvement": True,
                            "suggested_approach": strategy["actions"][0] if strategy["actions"] else "review_patterns"
                        }
            
        except Exception as e:
            logger.error(f"Error creating recommendations: {e}")
        
        return recommendations
    
    async def _store_learning_results(self, results: Dict[str, Any]):
        """Store learning results for future reference"""
        try:
            if self.redis_client:
                key = f"learning_results:{self.session_id}:{results['timestamp']}"
                data = json.dumps(results)
                self.redis_client.setex(key, 86400 * 14, data)  # 14 days TTL
                
                # Update last learning time
                self.redis_client.setex(
                    f"last_learning:{self.session_id}",
                    86400 * 30,
                    results['timestamp']
                )
                
        except Exception as e:
            logger.error(f"Error storing learning results: {e}")
    
    async def _get_last_learning_time(self) -> Optional[datetime]:
        """Get the timestamp of the last learning cycle"""
        try:
            if self.redis_client:
                timestamp_str = self.redis_client.get(f"last_learning:{self.session_id}")
                if timestamp_str:
                    if isinstance(timestamp_str, bytes):
                        timestamp_str = timestamp_str.decode()
                    return datetime.fromisoformat(timestamp_str)
        except Exception as e:
            logger.error(f"Error getting last learning time: {e}")
        
        return None
    
    async def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning status and statistics"""
        status = {
            "learning_active": self.learning_active,
            "last_learning": None,
            "total_patterns_learned": 0,
            "recent_improvements": [],
            "next_learning_scheduled": None
        }
        
        try:
            # Get last learning time
            last_learning = await self._get_last_learning_time()
            if last_learning:
                status["last_learning"] = last_learning.isoformat()
                
                # Calculate next scheduled learning
                next_learning = last_learning + timedelta(seconds=self.learning_interval)
                status["next_learning_scheduled"] = next_learning.isoformat()
            
            # Get pattern count
            patterns = await self.memory_system.pattern_storage.get_success_patterns(min_usage_count=1)
            status["total_patterns_learned"] = len(patterns)
            
            # Get recent learning insights
            insights = await self.memory_system.get_learning_insights()
            status["learning_insights"] = insights
            
        except Exception as e:
            logger.error(f"Error getting learning status: {e}")
        
        return status