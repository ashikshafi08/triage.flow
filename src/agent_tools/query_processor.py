"""
Query Preprocessing and Enhancement Module

This module handles:
1. Query complexity analysis
2. Context enhancement for vague queries
3. Dynamic iteration limit calculation
4. Fallback strategy determination

Now using native LlamaIndex features instead of hardcoded logic
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core import PromptTemplate
from llama_index.core.llms import LLM

from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class QueryInfo:
    """Information about a processed query"""
    complexity: str
    max_iterations: int
    query_type: str
    enhanced_query: str
    confidence: float = 0.0

class QueryProcessor:
    """Processes and enhances queries for optimal agentic performance using native LlamaIndex"""
    
    def __init__(self, session_context: Optional[Dict[str, Any]] = None):
        self.session_context = session_context or {}
        self._tool_selector = None
        self._llm = None
        
    def set_llm(self, llm: LLM):
        """Set LLM for dynamic analysis"""
        self._llm = llm
        try:
            self._tool_selector = PydanticSingleSelector.from_defaults(llm=llm)
        except ValueError as e:
            if "does not support function calling" in str(e):
                logger.warning(f"LLM does not support function calling for PydanticSingleSelector, disabling advanced tool selection: {e}")
                self._tool_selector = None
            else:
                raise
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main processing pipeline for queries
        Returns enhanced query and processing metadata
        """
        try:
            # Step 1: Always start with heuristic analysis for speed
            heuristic_analysis = self._analyze_query_heuristic(query)
            
            # Step 2: Use LLM only for low-confidence cases if available
            analysis = self._smart_query_analysis(query, heuristic_analysis)
            
            # Step 2: Enhance vague queries with context
            enhanced_query = self._enhance_query_with_context(query, analysis)
            
            # Step 3: Calculate optimal iteration limits (simplified)
            iteration_config = self._calculate_iteration_limits(analysis)
            
            # Step 4: Determine processing strategy
            strategy = self._determine_strategy(analysis)
            
            return {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "analysis": analysis,
                "iteration_config": iteration_config,
                "strategy": strategy,
                "should_preprocess": enhanced_query != query
            }
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return self._get_fallback_result(query)
            
    def _smart_query_analysis(self, query: str, heuristic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Smart query analysis that uses heuristics first and LLM only when needed.
        This significantly reduces LLM calls for simple queries.
        """
        confidence = heuristic_analysis.get("confidence", 0.0)
        complexity = heuristic_analysis.get("complexity", "moderate")
        
        # Use heuristic result if confidence is high enough
        if confidence >= 0.7 or not self._llm:
            logger.debug(f"Using heuristic analysis (confidence: {confidence:.2f})")
            return heuristic_analysis
        
        # Use LLM for borderline cases (moderate confidence) and complex queries
        if confidence >= 0.4 and complexity in ["simple", "moderate"]:
            logger.debug(f"Borderline case, using heuristic analysis (confidence: {confidence:.2f})")
            return heuristic_analysis
        
        # Use LLM for low confidence or complex queries
        logger.debug(f"Low confidence ({confidence:.2f}) or complex query, using LLM analysis")
        try:
            llm_analysis = self._analyze_query_with_llm(query)
            llm_analysis["analysis_method"] = "llm"
            llm_analysis["fallback_confidence"] = confidence
            return llm_analysis
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, falling back to heuristics")
            heuristic_analysis["analysis_method"] = "heuristic_fallback"
            return heuristic_analysis
    
    def _analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Analyze query using LLM for better understanding"""
        analysis_prompt = PromptTemplate(
            """Analyze this query and provide:
            1. Complexity level (simple/moderate/complex/research)
            2. Query type (search/analysis/debugging/exploration/implementation/architecture/general)
            3. Whether it's vague (yes/no)
            4. Whether it has specific references like file names or functions (yes/no)
            
            Query: {query}
            
            Respond in JSON format:
            {{"complexity": "...", "query_type": "...", "is_vague": true/false, "has_specific_refs": true/false}}
            """
        )
        
        try:
            response = self._llm.complete(analysis_prompt.format(query=query))
            import json
            result = json.loads(response.text)
            
            # Add computed fields
            result["word_count"] = len(query.split())
            result["original_query"] = query
            result["complexity_score"] = self._map_complexity_to_score(result["complexity"])
            result["estimated_difficulty"] = result["complexity"]
            
            return result
        except:
            # Fallback to heuristic analysis
            return self._analyze_query_heuristic(query)
    
    def _analyze_query_heuristic(self, query: str) -> Dict[str, Any]:
        """Enhanced heuristic analysis with confidence scoring"""
        words = query.split()
        word_count = len(words)
        query_lower = query.lower()
        
        # Enhanced vague detection
        vague_indicators = ["this", "that", "it", "here", "there", "issue", "problem", "error", "thing", "stuff"]
        vague_count = sum(1 for word in words if word.lower() in vague_indicators)
        
        # Enhanced specific reference detection
        has_specific_refs = bool(
            re.search(r'@[\w/.-]+', query) or  # File references
            re.search(r'\.(py|js|ts|java|cpp|c|go|rs|jsx|tsx|vue|php|rb|swift|kt)\b', query) or  # File extensions
            re.search(r'(def|class|function|import|package|const|let|var|interface|type)\s+\w+', query) or  # Code constructs
            re.search(r'#\d+', query) or  # Issue numbers
            re.search(r'(src/|lib/|components/|utils/|tests?/)', query) or  # Common paths
            re.search(r'\b[A-Z][a-zA-Z]*[A-Z][a-zA-Z]*\b', query) or  # CamelCase (likely class/component names)
            re.search(r'\b\w*[._]\w+\b', query)  # Snake_case or dot.notation
        )
        
        # Calculate confidence based on multiple factors
        confidence_factors = {
            "has_specific_refs": 0.4 if has_specific_refs else 0.0,
            "low_vague_count": 0.3 if vague_count <= 1 else 0.0,
            "moderate_length": 0.2 if 3 <= word_count <= 15 else 0.0,
            "clear_intent": 0.1 if any(keyword in query_lower for keyword in 
                                     ["find", "search", "show", "list", "explain", "how", "what", "where", "who", "when"]) else 0.0
        }
        confidence = min(sum(confidence_factors.values()), 1.0)
        
        # Enhanced complexity scoring
        complexity_base = word_count // 4
        complexity_adjustments = {
            "vague_penalty": 2 if vague_count > 2 else 0,
            "specific_bonus": -1 if has_specific_refs else 0,
            "question_bonus": -1 if query.strip().endswith('?') else 0,
            "multi_part_penalty": 1 if query.count('and') + query.count('or') > 1 else 0
        }
        complexity_score = max(0, complexity_base + sum(complexity_adjustments.values()))
        
        # Map to complexity level with better thresholds
        if complexity_score <= 1 and has_specific_refs:
            complexity = "simple"
        elif complexity_score <= 3:
            complexity = "moderate"
        elif complexity_score <= 6:
            complexity = "complex"
        else:
            complexity = "research"
        
        # Determine query type with enhanced patterns
        query_type = self._classify_query_type(query)
        
        return {
            "original_query": query,
            "word_count": word_count,
            "vague_count": vague_count,
            "has_specific_refs": has_specific_refs,
            "complexity_score": complexity_score,
            "complexity": complexity,
            "query_type": query_type,
            "is_vague": vague_count > 0 and not has_specific_refs,
            "estimated_difficulty": complexity,
            "confidence": confidence,
            "confidence_factors": confidence_factors,
            "analysis_method": "heuristic"
        }
    
    def _classify_query_type(self, query: str) -> str:
        """Enhanced query type classification for optimal processing"""
        query_lower = query.lower()
        
        # Enhanced patterns with more comprehensive keywords and scoring
        patterns = [
            ("git_history", ["when", "who changed", "git blame", "commit", "history", "author", "last edited", "evolution"]),
            ("git_blame", ["blame", "who wrote", "who added", "author of", "last modified by"]),
            ("issue_analysis", ["issue #", "bug #", "problem with", "error in", "issue", "github issue"]),
            ("pr_analysis", ["pull request", "pr #", "merge", "diff", "changes in pr"]),
            ("file_content", ["read", "show content", "what's in", "contents of", "file content"]),
            ("file_structure", ["structure", "organize", "hierarchy", "files in", "directory structure"]),
            ("search", ["find", "search", "locate", "where is", "which file", "spot", "grep"]),
            ("code_generation", ["generate", "create", "write", "implement", "example", "template"]),
            ("debugging", ["debug", "fix", "error", "bug", "troubleshoot", "not working", "failing", "broken"]),
            ("architecture", ["architecture", "design", "pattern", "relationship", "dependency", "overview"]),
            ("analysis", ["explain", "what does", "how does", "analyze", "understand", "describe", "why"]),
            ("exploration", ["explore", "list", "show me", "directory", "folder", "browse"]),
        ]
        
        # Score each pattern
        scores = {}
        for query_type, keywords in patterns:
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[query_type] = score
        
        # Return the highest scoring type, or "general" if no matches
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    def _enhance_query_with_context(self, query: str, analysis: Dict[str, Any]) -> str:
        """Enhance vague queries with available context"""
        if not settings.ENABLE_CONTEXT_ENHANCEMENT or not analysis.get("is_vague", False):
            return query
        
        # Use LLM to enhance if available
        if self._llm and self.session_context:
            return self._enhance_with_llm(query, analysis)
        
        # Fallback to simple enhancement
        return self._simple_enhance(query, analysis)
    
    def _enhance_with_llm(self, query: str, analysis: Dict[str, Any]) -> str:
        """Use LLM to naturally integrate context"""
        context_parts = []
        
        if self.session_context.get("current_file"):
            context_parts.append(f"Current file: {self.session_context['current_file']}")
        if self.session_context.get("recent_error"):
            context_parts.append(f"Recent error: {self.session_context['recent_error']}")
        if self.session_context.get("repo_info"):
            repo = self.session_context["repo_info"]
            context_parts.append(f"Repository: {repo.get('owner', 'unknown')}/{repo.get('repo', 'unknown')}")
        
        if not context_parts:
            return query
            
        enhance_prompt = PromptTemplate(
            """Rewrite this query to be more specific by naturally incorporating the context.
            Keep it concise and clear.
            
            Query: {query}
            Context: {context}
            
            Enhanced query:"""
        )
        
        try:
            response = self._llm.complete(
                enhance_prompt.format(query=query, context="; ".join(context_parts))
            )
            return response.text.strip()
        except:
            return self._simple_enhance(query, analysis)
    
    def _simple_enhance(self, query: str, analysis: Dict[str, Any]) -> str:
        """Simple context enhancement without LLM"""
        enhanced_query = query
        context_additions = []
        
        # Add available context
        if self.session_context.get("current_file"):
            context_additions.append(f"Current file: {self.session_context['current_file']}")
        
        if self.session_context.get("recent_error"):
            context_additions.append(f"Recent error: {self.session_context['recent_error']}")
        
        # Replace vague references
        if "this issue" in query.lower() and self.session_context.get("current_issue"):
            issue = self.session_context["current_issue"]
            enhanced_query = enhanced_query.replace(
                "this issue", 
                f"issue #{issue.get('number', 'unknown')}"
            )
        
        if "this file" in query.lower() and self.session_context.get("current_file"):
            enhanced_query = enhanced_query.replace(
                "this file",
                f"the file {self.session_context['current_file']}"
            )
        
        # Add context if we have any
        if context_additions and enhanced_query == query:
            enhanced_query = query + "\n\nContext: " + "; ".join(context_additions)
        
        return enhanced_query
    
    def _calculate_iteration_limits(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal iteration limits - simplified approach"""
        base_iterations = settings.AGENTIC_BASE_ITERATIONS
        max_iterations = settings.AGENTIC_MAX_ITERATIONS
        
        # Simple multipliers based on complexity
        multipliers = {
            "simple": 1.0,
            "moderate": 1.5,
            "complex": 2.0,
            "research": 2.5
        }
        
        complexity = analysis.get("complexity", "moderate")
        multiplier = multipliers.get(complexity, 1.5)
        
        # Calculate iterations
        calculated_iterations = int(base_iterations * multiplier)
        
        # Add small adjustment for vague queries
        if analysis.get("is_vague", False) and not analysis.get("has_specific_refs", False):
            calculated_iterations = int(calculated_iterations * 1.2)
        
        # Cap at maximum
        calculated_iterations = min(calculated_iterations, max_iterations)
        
        return {
            "max_iterations": calculated_iterations,
            "base_iterations": base_iterations,
            "complexity_key": complexity,
            "multiplier": multiplier,
            "adjustment_factor": calculated_iterations / base_iterations if base_iterations > 0 else 1.0,
            "early_stopping_config": {
                "enabled": settings.ENABLE_EARLY_STOPPING,
                "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
                "min_iterations": settings.MIN_ITERATIONS_BEFORE_STOP,
                "repetition_threshold": settings.REPETITION_THRESHOLD
            }
        }
    
    def _determine_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the best processing strategy - simplified"""
        strategy = {
            "approach": "standard",
            "use_enhanced_agent": False,
            "enable_early_stopping": settings.ENABLE_EARLY_STOPPING,
            "fallback_mode": settings.AGENTIC_FALLBACK_MODE,
            "selected_tools": self._get_suggested_tools(analysis),
            "system_prompt_type": self._get_prompt_type(analysis)
        }
        
        # Use enhanced agent for complex queries
        if analysis.get("complexity") in ["complex", "research"]:
            strategy["use_enhanced_agent"] = True
            strategy["approach"] = "enhanced"
        
        # Special handling for vague queries
        if analysis.get("is_vague", False) and not analysis.get("has_specific_refs", False):
            strategy["approach"] = "context_first"
            strategy["enable_early_stopping"] = False
        
        # Simple queries can use early stopping
        if analysis.get("complexity") == "simple" and analysis.get("has_specific_refs", False):
            strategy["approach"] = "direct"
            strategy["enable_early_stopping"] = True
        
        return strategy
    
    def _get_suggested_tools(self, analysis: Dict[str, Any]) -> List[str]:
        """Get suggested tools - let LlamaIndex handle actual selection"""
        # Return generic suggestions, actual selection happens in agent
        query_type = analysis.get("query_type", "general")
        
        # These are just hints - LlamaIndex will do actual selection
        tool_hints = {
            "search": ["search", "find"],
            "analysis": ["analyze", "explain"],
            "debugging": ["debug", "trace"],
            "exploration": ["explore", "list"],
            "implementation": ["generate", "create"],
            "architecture": ["structure", "dependency"],
            "issue": ["issue", "github"],
            "general": ["*"]  # Use all available
        }
        
        return tool_hints.get(query_type, ["*"])
    
    def _get_prompt_type(self, analysis: Dict[str, Any]) -> str:
        """Get the appropriate system prompt type based on analysis"""
        query_type = analysis.get("query_type", "general")
        complexity = analysis.get("complexity", "moderate")
        
        # Map to prompt types
        if query_type == "search" and complexity == "simple":
            return "focused_search"
        elif query_type == "debugging":
            return "systematic_debugging"
        elif query_type == "exploration":
            return "comprehensive_exploration"
        elif query_type == "analysis":
            return "detailed_analysis"
        elif query_type == "architecture":
            return "comprehensive_exploration"
        else:
            return "general_assistant"
    
    def _map_complexity_to_score(self, complexity: str) -> int:
        """Map complexity level to numeric score"""
        scores = {
            "simple": 2,
            "moderate": 5,
            "complex": 7,
            "research": 9
        }
        return scores.get(complexity, 5)
    
    def _get_fallback_result(self, query: str) -> Dict[str, Any]:
        """Get fallback result when processing fails"""
        return {
            "original_query": query,
            "enhanced_query": query,
            "analysis": {
                "complexity": "moderate",
                "is_vague": False,
                "query_type": "general",
                "has_specific_refs": False
            },
            "iteration_config": {
                "max_iterations": settings.AGENTIC_BASE_ITERATIONS,
                "early_stopping_config": {
                    "enabled": False
                }
            },
            "strategy": {
                "approach": "standard",
                "selected_tools": ["*"],
                "system_prompt_type": "general_assistant"
            },
            "should_preprocess": False
        }
    
    def analyze_query(self, query: str) -> QueryInfo:
        """Simple query analysis for routing decisions"""
        try:
            # Process query
            result = self.process_query(query)
            analysis = result["analysis"]
            iteration_config = result["iteration_config"]
            
            return QueryInfo(
                complexity=analysis.get("complexity", "moderate"),
                max_iterations=iteration_config.get("max_iterations", settings.AGENTIC_BASE_ITERATIONS),
                query_type=analysis.get("query_type", "general"),
                enhanced_query=result.get("enhanced_query", query),
                confidence=0.8 if analysis.get("has_specific_refs", False) else 0.5
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            # Return safe defaults
            return QueryInfo(
                complexity='moderate',
                max_iterations=settings.AGENTIC_BASE_ITERATIONS,
                query_type='general',
                enhanced_query=query,
                confidence=0.5
            )
    
    def create_fallback_response(self, query: str, error: str, partial_results: List[Dict] = None) -> Dict[str, Any]:
        """Create a helpful fallback response when agentic processing fails"""
        analysis = self._analyze_query_heuristic(query)
        
        suggestions = []
        
        # Generate suggestions based on analysis
        if analysis["is_vague"]:
            suggestions.extend([
                "Try being more specific about what you're referring to",
                "Include specific file names, functions, or error messages"
            ])
        
        if analysis["complexity"] in ["complex", "research"]:
            suggestions.extend([
                "Try breaking your question into smaller, more specific parts",
                "Focus on one aspect at a time"
            ])
        
        if analysis["query_type"] == "search" and not analysis["has_specific_refs"]:
            suggestions.append("Include specific terms or patterns you're looking for")
        
        # Default suggestions if none generated
        if not suggestions:
            suggestions = [
                "Try rephrasing with more specific details",
                "Include relevant context or examples",
                "Break down complex questions into simpler parts"
            ]
        
        return {
            "type": "fallback",
            "original_query": query,
            "error": error,
            "message": "I couldn't complete the analysis within the available resources.",
            "suggestions": suggestions,
            "partial_results": partial_results or [],
            "next_steps": [
                "Try a more focused version of your question",
                "Use specific file or function names if known",
                "Consider exploring the codebase first to gather context"
            ]
        }