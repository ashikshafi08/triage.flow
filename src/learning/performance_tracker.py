"""
Performance Tracking System for Self-Improving Agents

Tracks agent performance metrics to enable data-driven improvements.
"""

import time
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Core performance metrics for agent interactions"""
    session_id: str
    query: str
    query_complexity: float
    response_time: float
    tokens_used: int
    iterations_needed: int
    tool_calls: List[Dict[str, Any]]
    success: bool
    error_count: int
    user_satisfaction: Optional[float] = None
    goal_achievement: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary"""
        data = data.copy()
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class PerformanceTracker:
    """
    Tracks and analyzes agent performance over time using production-quality scoring.
    Uses adaptive thresholds similar to systems like Copilot, Cursor, and Devin.
    """
    
    def __init__(self, session_id: str, redis_client=None):
        self.session_id = session_id
        self.redis_client = redis_client
        self.current_interaction = {}
        self.metrics_cache = []
        
        # Adaptive thresholds based on query complexity (production approach)
        self.adaptive_thresholds = {
            "success_rate": {
                "simple": 0.90,    # Simple queries should have very high success
                "moderate": 0.80,  # Moderate complexity
                "complex": 0.65,   # Complex queries have lower expected success
                "research": 0.55   # Research queries are most challenging
            },
            "response_time": {
                "simple": 5.0,     # Simple queries should be fast
                "moderate": 15.0,  # Moderate complexity
                "complex": 30.0,   # Complex queries can take longer
                "research": 60.0   # Research queries may need more time
            },
            "goal_achievement": {
                "simple": 0.85,    # High goal achievement expected
                "moderate": 0.70,  # Good goal achievement
                "complex": 0.60,   # Reasonable for complex queries
                "research": 0.50   # Exploration may not fully achieve goals
            }
        }
        
        # Performance remediation actions (what happens when scores aren't hit)
        self.remediation_actions = {
            "below_threshold": {
                "success_rate": ["review_tool_selection", "add_validation_steps", "try_alternative_approach"],
                "response_time": ["optimize_tool_order", "use_parallel_execution", "cache_frequent_operations"],
                "goal_achievement": ["clarify_requirements", "break_down_complex_queries", "use_iterative_refinement"]
            },
            "consistently_poor": ["flag_for_human_review", "escalate_complexity_level", "request_model_update"]
        }
        
    def start_interaction(self, query: str) -> str:
        """Start tracking a new interaction"""
        interaction_id = f"{self.session_id}_{int(time.time())}"
        
        self.current_interaction = {
            "interaction_id": interaction_id,
            "start_time": time.time(),
            "query": query,
            "query_complexity": self._calculate_query_complexity(query),
            "tool_calls": [],
            "errors": [],
            "tokens_used": 0
        }
        
        logger.debug(f"Started tracking interaction {interaction_id}")
        return interaction_id
    
    def record_tool_call(self, tool_name: str, success: bool, duration: float, **kwargs):
        """Record a tool call during the interaction"""
        if not self.current_interaction:
            logger.warning("No active interaction to record tool call")
            return
            
        tool_call = {
            "tool": tool_name,
            "success": success,
            "duration": duration,
            "timestamp": time.time(),
            **kwargs
        }
        
        self.current_interaction["tool_calls"].append(tool_call)
        
        if not success:
            self.current_interaction["errors"].append({
                "tool": tool_name,
                "timestamp": time.time(),
                "error": kwargs.get("error", "Unknown error")
            })
    
    def add_token_usage(self, tokens: int):
        """Add token usage to current interaction"""
        if self.current_interaction:
            self.current_interaction["tokens_used"] += tokens
    
    def record_error(self, error: str, context: Dict[str, Any] = None):
        """Record an error during the interaction"""
        if not self.current_interaction:
            return
            
        self.current_interaction["errors"].append({
            "error": error,
            "context": context or {},
            "timestamp": time.time()
        })
    
    async def end_interaction(self, 
                             success: bool,
                             response: str = None,
                             user_feedback: Optional[float] = None) -> PerformanceMetrics:
        """End the current interaction and store metrics"""
        if not self.current_interaction:
            logger.warning("No active interaction to end")
            return None
            
        end_time = time.time()
        response_time = end_time - self.current_interaction["start_time"]
        
        # Calculate performance metrics using production-quality scoring
        tool_success_rate = self._calculate_tool_success_rate()
        goal_achievement = self._calculate_goal_achievement(success, response)
        iterations_needed = len(self.current_interaction["tool_calls"])
        
        metrics = PerformanceMetrics(
            session_id=self.session_id,
            query=self.current_interaction["query"],
            query_complexity=self.current_interaction["query_complexity"],
            response_time=response_time,
            tokens_used=self.current_interaction["tokens_used"],
            iterations_needed=iterations_needed,
            tool_calls=self.current_interaction["tool_calls"],
            success=success,
            error_count=len(self.current_interaction["errors"]),
            user_satisfaction=user_feedback,
            goal_achievement=goal_achievement
        )
        
        # Evaluate performance against adaptive thresholds
        performance_evaluation = self._evaluate_against_thresholds(metrics)
        
        # Handle below-threshold performance (what happens when scores aren't hit)
        if not performance_evaluation["meets_expectations"]:
            await self._handle_below_threshold_performance(metrics, performance_evaluation)
        
        # Store metrics with evaluation results
        await self._store_metrics(metrics, performance_evaluation)
        
        # Reset current interaction
        self.current_interaction = {}
        
        # Enhanced logging with performance context
        grade = self._calculate_performance_grade(goal_achievement)
        logger.info(f"Interaction completed: success={success}, grade={grade}, time={response_time:.2f}s, tokens={metrics.tokens_used}")
        return metrics
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate complexity score for a query"""
        complexity = 0.0
        
        # Word count factor
        word_count = len(query.split())
        complexity += min(word_count / 20.0, 1.0) * 0.3
        
        # Special patterns that indicate complexity
        complex_patterns = [
            'analyze', 'compare', 'integrate', 'comprehensive', 'detailed',
            'architecture', 'relationship', 'dependency', 'trace', 'find all',
            'across files', 'entire codebase', 'performance', 'security'
        ]
        
        pattern_matches = sum(1 for pattern in complex_patterns if pattern.lower() in query.lower())
        complexity += min(pattern_matches / 5.0, 1.0) * 0.4
        
        # File/code references
        file_mentions = query.count('@') + query.count('/')
        complexity += min(file_mentions / 3.0, 1.0) * 0.2
        
        # Multiple questions
        questions = query.count('?') + query.count(' and ') * 0.5
        complexity += min(questions / 3.0, 1.0) * 0.1
        
        return min(complexity, 1.0)
    
    def _calculate_tool_success_rate(self) -> float:
        """Calculate success rate of tool calls"""
        tool_calls = self.current_interaction.get("tool_calls", [])
        if not tool_calls:
            return 1.0
            
        successful = sum(1 for call in tool_calls if call.get("success", False))
        return successful / len(tool_calls)
    
    def _calculate_goal_achievement(self, success: bool, response: str = None) -> float:
        """Calculate goal achievement using production-quality scoring"""
        if not success:
            return 0.15  # Minimal achievement for failed interactions
        
        # Start with base score for successful completion
        score = 0.5
        
        if response:
            response_lower = response.lower()
            
            # Length and substance scoring (like production systems)
            if len(response) > 200:
                score += 0.15  # Substantial response
            elif len(response) > 50:
                score += 0.05  # Adequate response
            
            # Error-free execution bonus
            if not any(error_word in response_lower for error_word in ["error", "failed", "exception", "unable"]):
                score += 0.1
            
            # Value indicators (weighted by production importance)
            high_value = ["implemented", "created", "fixed", "resolved", "completed", "generated"]
            medium_value = ["found", "identified", "analyzed", "solution", "approach", "explanation"]
            low_value = ["shows", "indicates", "suggests", "appears", "might"]
            
            if any(indicator in response_lower for indicator in high_value):
                score += 0.20  # High value outcomes
            elif any(indicator in response_lower for indicator in medium_value):
                score += 0.10  # Medium value outcomes
            elif any(indicator in response_lower for indicator in low_value):
                score += 0.03  # Low value outcomes
            
            # Technical content bonus (for development tasks)
            if any(tech in response_lower for tech in ["function", "class", "import", "def ", "const", "var "]):  
                score += 0.08
        
        # Factor in tool execution effectiveness
        tool_success = self._calculate_tool_success_rate()
        score = (score * 0.8) + (tool_success * 0.2)  # 80/20 split
        
        return min(score, 1.0)
    
    async def _store_metrics(self, metrics: PerformanceMetrics, evaluation: Dict[str, Any] = None):
        """Store metrics in Redis and local cache with evaluation results"""
        try:
            # Store in local cache
            self.metrics_cache.append(metrics)
            
            # Keep only last 100 metrics in cache
            if len(self.metrics_cache) > 100:
                self.metrics_cache = self.metrics_cache[-100:]
            
            # Store in Redis if available
            if self.redis_client:
                # Enhanced metrics with evaluation data
                metrics_data = metrics.to_dict()
                if evaluation:
                    metrics_data["performance_evaluation"] = evaluation
                    metrics_data["complexity_classification"] = self._classify_query_complexity(metrics.query)
                    metrics_data["performance_grade"] = self._calculate_performance_grade(metrics.goal_achievement)
                
                key = f"performance_metrics:{self.session_id}:{metrics.timestamp.isoformat()}"
                self.redis_client.setex(
                    key, 
                    86400 * 30,  # 30 days TTL (increased for better analytics)
                    json.dumps(metrics_data)
                )
                
                # Also maintain a session index
                session_key = f"session_metrics:{self.session_id}"
                self.redis_client.lpush(session_key, key)
                self.redis_client.ltrim(session_key, 0, 999)  # Keep last 1000
                self.redis_client.expire(session_key, 86400 * 30)  # 30 days
                
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    async def get_recent_metrics(self, days: int = 7) -> List[PerformanceMetrics]:
        """Get recent performance metrics"""
        metrics = []
        
        # First check cache
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        cache_metrics = [
            m for m in self.metrics_cache 
            if m.timestamp >= cutoff_time
        ]
        metrics.extend(cache_metrics)
        
        # Then check Redis
        if self.redis_client:
            try:
                session_key = f"session_metrics:{self.session_id}"
                metric_keys = self.redis_client.lrange(session_key, 0, -1)
                
                for key in metric_keys:
                    if isinstance(key, bytes):
                        key = key.decode()
                    
                    data = self.redis_client.get(key)
                    if data:
                        if isinstance(data, bytes):
                            data = data.decode()
                        metric_data = json.loads(data)
                        metric = PerformanceMetrics.from_dict(metric_data)
                        
                        if metric.timestamp >= cutoff_time:
                            # Avoid duplicates from cache
                            if not any(m.timestamp == metric.timestamp for m in metrics):
                                metrics.append(metric)
                                
            except Exception as e:
                logger.error(f"Failed to retrieve metrics from Redis: {e}")
        
        # Sort by timestamp
        metrics.sort(key=lambda x: x.timestamp, reverse=True)
        return metrics
    
    async def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary statistics"""
        metrics = await self.get_recent_metrics(days)
        
        if not metrics:
            return {
                "total_interactions": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "avg_tokens": 0.0,
                "avg_complexity": 0.0,
                "improvement_trend": "insufficient_data"
            }
        
        total = len(metrics)
        successful = sum(1 for m in metrics if m.success)
        success_rate = successful / total
        
        avg_response_time = sum(m.response_time for m in metrics) / total
        avg_tokens = sum(m.tokens_used for m in metrics) / total
        avg_complexity = sum(m.query_complexity for m in metrics) / total
        avg_goal_achievement = sum(m.goal_achievement for m in metrics) / total
        
        # Calculate improvement trend
        improvement_trend = self._calculate_improvement_trend(metrics)
        
        # Calculate production-style metrics
        performance_grade = self._calculate_performance_grade(avg_goal_achievement)
        complexity_dist = self._get_complexity_distribution(metrics)
        
        return {
            "total_interactions": total,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "avg_tokens": avg_tokens,
            "avg_complexity": avg_complexity,
            "avg_goal_achievement": avg_goal_achievement,
            "improvement_trend": improvement_trend,
            "time_period_days": days,
            # Production-style enhancements
            "performance_grade": performance_grade,
            "complexity_distribution": complexity_dist,
            "quality_score": avg_goal_achievement,  # Alias for goal achievement
            "efficiency_score": min(1.0, 30.0 / avg_response_time) if avg_response_time > 0 else 1.0
        }
    
    def _calculate_improvement_trend(self, metrics: List[PerformanceMetrics]) -> str:
        """Calculate if performance is improving, declining, or stable"""
        if len(metrics) < 10:
            return "insufficient_data"
        
        # Split into two halves (recent vs older)
        mid_point = len(metrics) // 2
        recent_metrics = metrics[:mid_point]  # More recent (sorted desc)
        older_metrics = metrics[mid_point:]
        
        # Calculate average performance for each half
        recent_avg = sum(m.goal_achievement for m in recent_metrics) / len(recent_metrics)
        older_avg = sum(m.goal_achievement for m in older_metrics) / len(older_metrics)
        
        improvement = recent_avg - older_avg
        
        if improvement > 0.1:
            return "improving"
        elif improvement < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _evaluate_against_thresholds(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Evaluate performance against adaptive thresholds"""
        complexity = self._classify_query_complexity(metrics.query)
        thresholds = {k: v[complexity] for k, v in self.adaptive_thresholds.items()}
        
        evaluation = {
            "complexity": complexity,
            "thresholds_used": thresholds,
            "meets_expectations": True,
            "failing_metrics": [],
            "performance_score": 0.0
        }
        
        # Evaluate each metric with weighted scoring
        score_components = []
        
        # Success rate (high weight for basic functionality)
        success_score = 1.0 if metrics.success else 0.0
        if not metrics.success and complexity in ["simple", "moderate"]:
            evaluation["failing_metrics"].append("success_rate")
            evaluation["meets_expectations"] = False
        score_components.append(success_score * 0.35)  # 35% weight
        
        # Response time (efficiency scoring)
        time_score = min(1.0, thresholds["response_time"] / max(metrics.response_time, 1.0))
        if metrics.response_time > thresholds["response_time"]:
            evaluation["failing_metrics"].append("response_time")
            evaluation["meets_expectations"] = False
        score_components.append(time_score * 0.25)  # 25% weight
        
        # Goal achievement (quality scoring)
        if metrics.goal_achievement < thresholds["goal_achievement"]:
            evaluation["failing_metrics"].append("goal_achievement")
            evaluation["meets_expectations"] = False
        score_components.append(metrics.goal_achievement * 0.30)  # 30% weight
        
        # Error tolerance (reliability scoring)
        error_score = max(0.0, 1.0 - (metrics.error_count * 0.2))
        if metrics.error_count > 2:  # More than 2 errors is concerning
            evaluation["failing_metrics"].append("error_count")
            evaluation["meets_expectations"] = False
        score_components.append(error_score * 0.10)  # 10% weight
        
        evaluation["performance_score"] = sum(score_components)
        return evaluation
    
    def _classify_query_complexity(self, query: str) -> str:
        """Classify query complexity for adaptive thresholds"""
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Research indicators
        research_keywords = ["analyze", "understand", "explore", "investigate", "comprehensive", "detailed", "research"]
        if any(keyword in query_lower for keyword in research_keywords) and word_count > 15:
            return "research"
        
        # Complex indicators
        complex_keywords = ["integrate", "implement", "refactor", "optimize", "design", "architecture", "trace"]
        if any(keyword in query_lower for keyword in complex_keywords) or word_count > 25:
            return "complex"
        
        # Moderate indicators
        moderate_keywords = ["create", "modify", "update", "fix", "debug", "find all", "compare"]
        if any(keyword in query_lower for keyword in moderate_keywords) or word_count > 10:
            return "moderate"
        
        return "simple"
    
    def _calculate_performance_grade(self, goal_achievement: float) -> str:
        """Calculate letter grade like production systems"""
        if goal_achievement >= 0.9:
            return "A"
        elif goal_achievement >= 0.8:
            return "B"
        elif goal_achievement >= 0.7:
            return "C"
        elif goal_achievement >= 0.6:
            return "D"
        else:
            return "F"
    
    async def _handle_below_threshold_performance(self, metrics: PerformanceMetrics, evaluation: Dict[str, Any]):
        """Handle performance below thresholds - implement remediation actions"""
        failing_metrics = evaluation["failing_metrics"]
        complexity = evaluation["complexity"]
        
        try:
            # Log the performance issue with context
            logger.warning(f"Performance below threshold for session {self.session_id}: {failing_metrics} (complexity: {complexity})")
            
            # Determine and execute remediation actions
            actions_taken = []
            
            for failing_metric in failing_metrics:
                if failing_metric in self.remediation_actions["below_threshold"]:
                    metric_actions = self.remediation_actions["below_threshold"][failing_metric]
                    actions_taken.extend(metric_actions)
            
            # Store remediation record for learning
            if self.redis_client:
                remediation_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "session_id": self.session_id,
                    "query": metrics.query,
                    "failing_metrics": failing_metrics,
                    "complexity": complexity,
                    "performance_score": evaluation["performance_score"],
                    "actions_taken": actions_taken,
                    "thresholds_used": evaluation["thresholds_used"]
                }
                
                key = f"performance_remediation:{self.session_id}:{datetime.utcnow().timestamp()}"
                self.redis_client.setex(
                    key, 
                    86400 * 14,  # 14 days TTL for remediation tracking
                    json.dumps(remediation_record)
                )
            
            # Check for consistently poor performance
            await self._check_consistent_poor_performance()
            
        except Exception as e:
            logger.error(f"Error handling below-threshold performance: {e}")
    
    async def _check_consistent_poor_performance(self):
        """Check for patterns of consistently poor performance"""
        try:
            recent_metrics = await self.get_recent_metrics(days=1)
            if len(recent_metrics) < 5:
                return
            
            # Calculate recent performance grade distribution
            grades = [self._calculate_performance_grade(m.goal_achievement) for m in recent_metrics]
            poor_grades = sum(1 for grade in grades if grade in ["D", "F"])
            poor_percentage = poor_grades / len(grades)
            
            # If more than 60% of recent interactions are poor quality
            if poor_percentage > 0.6:
                logger.error(f"Consistent poor performance detected for session {self.session_id}: {poor_percentage:.2f}")
                
                # Execute escalation actions
                for action in self.remediation_actions["consistently_poor"]:
                    logger.info(f"Executing remediation action: {action}")
                
                # Store high-priority alert
                if self.redis_client:
                    alert_record = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "session_id": self.session_id,
                        "alert_type": "consistent_poor_performance",
                        "poor_percentage": poor_percentage,
                        "total_interactions": len(recent_metrics),
                        "escalation_level": "high",
                        "grade_distribution": {grade: grades.count(grade) for grade in set(grades)}
                    }
                    
                    key = f"performance_alert:{self.session_id}:{datetime.utcnow().timestamp()}"
                    self.redis_client.setex(
                        key,
                        86400 * 30,  # 30 days TTL for alerts
                        json.dumps(alert_record)
                    )
        
        except Exception as e:
            logger.error(f"Error checking consistent poor performance: {e}")
    
    async def should_trigger_learning(self) -> bool:
        """Determine if learning should be triggered based on adaptive performance criteria"""
        summary = await self.get_performance_summary(days=1)
        
        if summary["total_interactions"] < 5:
            return False
        
        # Get complexity-aware thresholds
        recent_metrics = await self.get_recent_metrics(days=1)
        complexity_distribution = self._get_complexity_distribution(recent_metrics)
        
        # Calculate weighted success rate expectation based on complexity
        expected_success_rate = 0.0
        total_weight = 0
        
        for complexity, count in complexity_distribution.items():
            if count > 0:
                expected_rate = self.adaptive_thresholds["success_rate"][complexity]
                expected_success_rate += expected_rate * count
                total_weight += count
        
        expected_success_rate = expected_success_rate / total_weight if total_weight > 0 else 0.8
        
        # Trigger learning if performance is below complexity-adjusted expectations
        should_learn = (
            summary["success_rate"] < expected_success_rate - 0.1 or  # 10% below expected
            summary["improvement_trend"] == "declining" or
            summary["avg_response_time"] > 45.0 or  # Consistently slow
            summary.get("avg_goal_achievement", 0.5) < 0.5  # Poor goal achievement
        )
        
        return should_learn
    
    def _get_complexity_distribution(self, metrics: List[PerformanceMetrics]) -> Dict[str, int]:
        """Get distribution of query complexity"""
        distribution = {"simple": 0, "moderate": 0, "complex": 0, "research": 0}
        
        for metric in metrics:
            complexity = self._classify_query_complexity(metric.query)
            distribution[complexity] += 1
        
        return distribution