"""
Early Termination System for Agent Optimization

This module implements intelligent early termination patterns to prevent
unnecessary agent iterations and reduce LLM API calls while maintaining quality.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TerminationSignal:
    """Represents a signal indicating the agent should terminate early"""
    should_terminate: bool
    confidence: float
    reason: str
    iteration_count: int
    response_quality: float = 0.0

class EarlyTerminationDetector:
    """
    Detects when an agent should terminate early based on response patterns,
    completion indicators, and quality metrics.
    """
    
    def __init__(self):
        self.iteration_history = []
        self.response_history = []
        self.consecutive_no_progress = 0
        self.max_no_progress = 2
        
        # Completion pattern indicators
        self.completion_patterns = [
            r"\b(?:I have found|I found|Found)\b.*(?:answer|solution|information|result)",
            r"\b(?:The analysis is|Analysis is)\s+(?:complete|finished|done)",
            r"\b(?:Based on my|From my)\s+(?:investigation|analysis|search|exploration)",
            r"\b(?:In summary|To summarize|In conclusion)",
            r"\b(?:Here is|Here are)\s+(?:the|your)\s+(?:answer|result|information|solution)",
            r"\b(?:I can see|I can confirm|I've identified)\s+(?:that|the|how)",
            r"\b(?:The (?:file|function|class|issue|problem))\s+(?:is|appears to be|seems to be)",
            r"\b(?:This (?:shows|indicates|suggests|reveals))\b",
            r"\b(?:Final|Ultimate|Complete)\s+(?:answer|result|analysis)"
        ]
        
        # Patterns indicating the agent is stuck or repeating
        self.stuck_patterns = [
            r"\b(?:I need to|Let me|I should)\s+(?:search|look|find|check)\s+(?:for|in|at)\b",
            r"\b(?:I'll|I will)\s+(?:try|attempt|search|look)\b",
            r"\b(?:Let me try|Let me search|Let me look)\b",
            r"\b(?:I'm still|I am still)\s+(?:looking|searching|trying)",
            r"\b(?:I couldn't find|I can't find|I cannot find|I was unable to find)\b",
            r"\b(?:No results|No matches|Nothing found)\b"
        ]
        
        # High-quality response indicators
        self.quality_patterns = [
            r"\b(?:function|class|method|variable)\s+\w+\s+(?:is|are)\s+(?:defined|located|implemented)",
            r"\b(?:Line \d+|Lines \d+-\d+)\b",
            r"\b(?:File|Path):\s*[\w/.-]+\.(py|js|ts|java|cpp|c|go|rs|jsx|tsx)",
            r"\b(?:commit|SHA|hash):\s*[a-f0-9]{7,40}\b",
            r"\b(?:Issue|PR|Pull Request)\s*#?\d+\b",
            r"\b(?:Author|Developer|Contributor):\s*\w+",
            r"\b(?:Added|Modified|Changed|Updated|Fixed)\s+in\s+commit\b"
        ]
    
    def should_terminate(self, current_response: str, iteration: int, total_iterations: int) -> TerminationSignal:
        """
        Determine if the agent should terminate early based on current response and history.
        
        Args:
            current_response: The current agent response
            iteration: Current iteration number
            total_iterations: Total planned iterations
            
        Returns:
            TerminationSignal indicating whether to terminate and why
        """
        # Add to history
        self.response_history.append(current_response)
        self.iteration_history.append(iteration)
        
        # Check for completion patterns
        completion_signal = self._check_completion_patterns(current_response, iteration)
        if completion_signal.should_terminate:
            return completion_signal
        
        # Check for stuck/repetitive behavior
        stuck_signal = self._check_stuck_patterns(current_response, iteration)
        if stuck_signal.should_terminate:
            return stuck_signal
        
        # Check for high-quality answer that appears complete
        quality_signal = self._check_response_quality(current_response, iteration)
        if quality_signal.should_terminate:
            return quality_signal
        
        # Check for repetitive responses
        repetition_signal = self._check_repetitive_responses(iteration)
        if repetition_signal.should_terminate:
            return repetition_signal
        
        # No termination signal detected
        return TerminationSignal(
            should_terminate=False,
            confidence=0.0,
            reason="No termination signal detected",
            iteration_count=iteration
        )
    
    def _check_completion_patterns(self, response: str, iteration: int) -> TerminationSignal:
        """Check for explicit completion indicators"""
        response_lower = response.lower()
        
        for pattern in self.completion_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                # Additional validation: check if response has substance
                if len(response.strip()) > 50:  # Minimum substantial response
                    confidence = 0.9 if iteration > 2 else 0.7  # Higher confidence after some iterations
                    return TerminationSignal(
                        should_terminate=True,
                        confidence=confidence,
                        reason=f"Completion pattern detected: {pattern}",
                        iteration_count=iteration,
                        response_quality=self._calculate_response_quality(response)
                    )
        
        return TerminationSignal(should_terminate=False, confidence=0.0, reason="", iteration_count=iteration)
    
    def _check_stuck_patterns(self, response: str, iteration: int) -> TerminationSignal:
        """Check if the agent appears stuck or unable to progress"""
        if iteration < 3:  # Don't terminate too early
            return TerminationSignal(should_terminate=False, confidence=0.0, reason="", iteration_count=iteration)
        
        stuck_indicators = 0
        for pattern in self.stuck_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                stuck_indicators += 1
        
        # If multiple stuck indicators and low quality response
        if stuck_indicators >= 2 and len(response.strip()) < 100:
            self.consecutive_no_progress += 1
            if self.consecutive_no_progress >= self.max_no_progress:
                return TerminationSignal(
                    should_terminate=True,
                    confidence=0.8,
                    reason=f"Agent appears stuck with {stuck_indicators} stuck indicators",
                    iteration_count=iteration,
                    response_quality=0.2
                )
        else:
            self.consecutive_no_progress = 0
        
        return TerminationSignal(should_terminate=False, confidence=0.0, reason="", iteration_count=iteration)
    
    def _check_response_quality(self, response: str, iteration: int) -> TerminationSignal:
        """Check if response quality is high enough to justify early termination"""
        quality_score = self._calculate_response_quality(response)
        
        # High quality response with specific details can justify early termination
        if quality_score >= 0.8 and iteration >= 2:
            # Additional check: does response contain actionable information?
            actionable_indicators = [
                r"\b(?:file|line|function|class|method)\s+\w+",
                r"\b(?:located|found|defined)\s+(?:in|at)\b",
                r"\b(?:you can|you should|to fix|to solve)\b",
                r"\bcommit\s+[a-f0-9]{7,}\b"
            ]
            
            actionable_count = sum(1 for pattern in actionable_indicators 
                                 if re.search(pattern, response, re.IGNORECASE))
            
            if actionable_count >= 2:
                return TerminationSignal(
                    should_terminate=True,
                    confidence=0.85,
                    reason=f"High quality response (score: {quality_score:.2f}) with actionable information",
                    iteration_count=iteration,
                    response_quality=quality_score
                )
        
        return TerminationSignal(should_terminate=False, confidence=0.0, reason="", iteration_count=iteration)
    
    def _check_repetitive_responses(self, iteration: int) -> TerminationSignal:
        """Check for repetitive or circular reasoning"""
        if len(self.response_history) < 3:
            return TerminationSignal(should_terminate=False, confidence=0.0, reason="", iteration_count=iteration)
        
        # Check last 3 responses for similar content
        recent_responses = self.response_history[-3:]
        
        # Simple similarity check (could be enhanced with more sophisticated NLP)
        similarities = []
        for i in range(len(recent_responses)):
            for j in range(i + 1, len(recent_responses)):
                similarity = self._calculate_response_similarity(recent_responses[i], recent_responses[j])
                similarities.append(similarity)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        
        if avg_similarity > 0.7:  # High similarity threshold
            return TerminationSignal(
                should_terminate=True,
                confidence=0.75,
                reason=f"Repetitive responses detected (similarity: {avg_similarity:.2f})",
                iteration_count=iteration,
                response_quality=0.3
            )
        
        return TerminationSignal(should_terminate=False, confidence=0.0, reason="", iteration_count=iteration)
    
    def _calculate_response_quality(self, response: str) -> float:
        """Calculate the quality score of a response"""
        if not response or len(response.strip()) < 20:
            return 0.0
        
        quality_score = 0.0
        
        # Check for quality indicators
        for pattern in self.quality_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                quality_score += 0.15
        
        # Length and structure bonus
        if 50 <= len(response) <= 1000:  # Reasonable length
            quality_score += 0.1
        
        # Information density
        words = response.split()
        if len(words) > 10:
            unique_words = len(set(word.lower() for word in words))
            density = unique_words / len(words)
            quality_score += density * 0.2
        
        return min(quality_score, 1.0)
    
    def _calculate_response_similarity(self, response1: str, response2: str) -> float:
        """Calculate similarity between two responses (simple word overlap)"""
        if not response1 or not response2:
            return 0.0
        
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def reset(self):
        """Reset the detector state for a new query"""
        self.iteration_history = []
        self.response_history = []
        self.consecutive_no_progress = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about termination decisions"""
        return {
            "total_iterations": len(self.iteration_history),
            "responses_analyzed": len(self.response_history),
            "consecutive_no_progress": self.consecutive_no_progress,
            "avg_response_length": sum(len(r) for r in self.response_history) / len(self.response_history) if self.response_history else 0
        }

class EarlyTerminationAgent:
    """
    Wrapper around ReActAgent that implements early termination logic.
    """
    
    def __init__(self, base_agent, termination_detector: EarlyTerminationDetector = None):
        self.base_agent = base_agent
        self.detector = termination_detector or EarlyTerminationDetector()
        self.termination_enabled = True
        
    async def achat(self, message: str) -> str:
        """Enhanced chat with early termination detection"""
        if not self.termination_enabled:
            return await self.base_agent.achat(message)
        
        self.detector.reset()
        
        # We need to hook into the agent's iteration loop
        # This is a simplified version - in practice, you'd need to modify
        # the ReActAgent's internal loop or use a custom agent implementation
        
        max_iterations = getattr(self.base_agent, 'max_iterations', 10)
        
        for iteration in range(1, max_iterations + 1):
            try:
                # Get response from base agent (simplified - would need proper integration)
                response = await self.base_agent.achat(message)
                
                # Check for early termination
                signal = self.detector.should_terminate(str(response), iteration, max_iterations)
                
                if signal.should_terminate:
                    logger.info(f"Early termination at iteration {iteration}: {signal.reason} (confidence: {signal.confidence:.2f})")
                    return str(response)
                
            except Exception as e:
                logger.error(f"Error in early termination detection: {e}")
                # Fallback to normal behavior
                break
        
        # If no early termination, return the final response
        return await self.base_agent.achat(message)
    
    def enable_early_termination(self, enabled: bool = True):
        """Enable or disable early termination"""
        self.termination_enabled = enabled
        logger.info(f"Early termination {'enabled' if enabled else 'disabled'}")