"""
Agent Manager with Production-Ready Features

This module implements sophisticated agentic features using native LlamaIndex:
- Built-in iteration control and timeout management
- Native memory management with ChatMemoryBuffer
- LlamaIndex callback system for monitoring
- Automatic retry and error handling
- Native tool selection based on metadata
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer, ChatSummaryMemoryBuffer
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, TokenCountingHandler
from llama_index.core.tools import ToolMetadata
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core import Settings

from ..config import settings
from .query_processor import QueryProcessor

logger = logging.getLogger(__name__)

class AgentManager:
    """Production-ready agent manager using native LlamaIndex features"""
    
    def __init__(self, tools: List, llm, session_context: Optional[Dict] = None):
        self.tools = tools
        self.llm = llm
        self.session_context = session_context or {}
        
        # Initialize query processor with LLM
        self.query_processor = QueryProcessor(session_context)
        self.query_processor.set_llm(llm)
        
        # Configure global settings
        Settings.llm = llm
        Settings.callback_manager = self._create_callback_manager()
        
        # Token counting handler for monitoring
        self.token_counter = TokenCountingHandler()
        
    def _create_callback_manager(self) -> CallbackManager:
        """Create callback manager with native handlers"""
        handlers = []
        
        # Debug handler for development
        if settings.AGENTIC_DEBUG_MODE:
            debug_handler = LlamaDebugHandler(print_trace_on_end=True)
            handlers.append(debug_handler)
        
        # Token counting for monitoring
        self.token_counter = TokenCountingHandler()
        handlers.append(self.token_counter)
        
        return CallbackManager(handlers)
    
    async def execute_query(self, query: str) -> Dict[str, Any]:
        """Execute query with native LlamaIndex features"""
        
        try:
            # Step 1: Process and analyze query
            query_result = self.query_processor.process_query(query)
            enhanced_query = query_result["enhanced_query"]
            analysis = query_result["analysis"]
            iteration_config = query_result["iteration_config"]
            strategy = query_result["strategy"]
            
            logger.info(f"Agent Manager: {analysis['complexity']} query, "
                       f"{iteration_config['max_iterations']} max iterations, "
                       f"strategy: {strategy['approach']}")
            
            # Step 2: Create optimized agent with native features
            agent = self._create_optimized_agent(
                enhanced_query, analysis, iteration_config, strategy
            )
            
            # Step 3: Execute with native timeout and streaming
            result = await self._execute_with_native_features(
                agent, enhanced_query, iteration_config, strategy
            )
            
            if result["success"]:
                return result
            
            # Step 4: Apply native fallback strategies
            return await self._apply_native_fallbacks(
                query, enhanced_query, analysis, result
            )
            
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            return self._create_error_response(query, e)
    
    def _create_optimized_agent(
        self, 
        query: str, 
        analysis: Dict[str, Any], 
        iteration_config: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> ReActAgent:
        """Create agent with native LlamaIndex configuration"""
        
        # Select memory type based on complexity
        if analysis["complexity"] in ["complex", "research"]:
            # Use summary memory for complex queries
            memory = ChatSummaryMemoryBuffer.from_defaults(
                llm=self.llm,
                token_limit=4000,
                summarize_tmpl=(
                    "Summarize the key points from this conversation so far:\n"
                    "{context_str}\n\n"
                    "Summary:"
                )
            )
        else:
            # Standard memory for simple queries
            memory = ChatMemoryBuffer.from_defaults(
                token_limit=3000
            )
        
        # Get system prompt
        system_prompt = self._get_system_prompt(strategy["system_prompt_type"])
        
        # Add context to system prompt if available
        if self.session_context:
            context_info = self._format_context_info()
            if context_info:
                system_prompt = f"{system_prompt}\n\nCurrent context:\n{context_info}"
        
        # Create agent with native features
        agent = ReActAgent.from_tools(
            tools=self.tools,  # Let LlamaIndex handle tool selection
            llm=self.llm,
            memory=memory,
            max_iterations=iteration_config["max_iterations"],
            verbose=settings.AGENTIC_DEBUG_MODE,
            system_prompt=system_prompt,
            # Native features
            react_chat_formatter=ReActChatFormatter(),
            handle_parsing_errors=True,  # Automatic error recovery
            callback_manager=Settings.callback_manager
        )
        
        return agent
    
    def _format_context_info(self) -> str:
        """Format session context for system prompt"""
        context_parts = []
        
        if self.session_context.get("repo_info"):
            repo = self.session_context["repo_info"]
            context_parts.append(f"Repository: {repo.get('owner', 'unknown')}/{repo.get('repo', 'unknown')}")
        
        if self.session_context.get("current_file"):
            context_parts.append(f"Current file: {self.session_context['current_file']}")
        
        if self.session_context.get("recent_error"):
            context_parts.append(f"Recent error: {self.session_context['recent_error']}")
        
        return "\n".join(context_parts)
    
    def _get_system_prompt(self, prompt_type: str) -> str:
        """Get optimized system prompt based on query type"""
        
        from .prompts import DEFAULT_SYSTEM_PROMPT
        
        # Use the main triage.flow system prompt for all agent operations
        return DEFAULT_SYSTEM_PROMPT
    
    async def _execute_with_native_features(
        self,
        agent: ReActAgent,
        query: str,
        iteration_config: Dict[str, Any],
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute using native LlamaIndex features"""
        
        try:
            # Reset token counter
            self.token_counter.reset()
            
            # Set timeout based on complexity
            timeout = self._get_timeout_for_strategy(strategy)
            
            # Use native streaming for better UX
            if getattr(settings, "ENABLE_STREAMING", False):
                response = await self._execute_streaming(agent, query, timeout)
            else:
                # Standard execution with timeout
                if timeout > 0:
                    response = await asyncio.wait_for(
                        agent.achat(query),
                        timeout=timeout
                    )
                else:
                    response = await agent.achat(query)
            
            # Get token usage
            token_usage = {
                "total_llm_tokens": self.token_counter.total_llm_token_count,
                "total_embedding_tokens": self.token_counter.total_embedding_token_count,
                "llm_calls": self.token_counter.llm_token_counts
            }
            
            return {
                "success": True,
                "response": str(response),
                "method": "native_agent",
                "metadata": {
                    "strategy": strategy["approach"],
                    "max_iterations": iteration_config["max_iterations"],
                    "token_usage": token_usage,
                    "complexity": strategy.get("complexity", "unknown")
                }
            }
            
        except asyncio.TimeoutError:
            logger.warning(f"Agent timed out after {timeout} seconds")
            return {
                "success": False,
                "error": "timeout",
                "partial_results": self._extract_partial_from_memory(agent)
            }
        except Exception as e:
            error_msg = str(e)
            
            # Handle max iterations gracefully
            if "max_iterations" in error_msg.lower() or "reached max" in error_msg.lower():
                partial = self._extract_partial_from_memory(agent)
                if partial:
                    return {
                        "success": True,
                        "response": f"Analysis (iteration limit reached):\n{partial}",
                        "method": "partial_recovery",
                        "metadata": {"reached_limit": True}
                    }
            
            return {
                "success": False,
                "error": str(e),
                "partial_results": self._extract_partial_from_memory(agent)
            }
    
    async def _execute_streaming(self, agent: ReActAgent, query: str, timeout: int) -> str:
        """Execute with streaming response"""
        try:
            # Use native streaming
            response_stream = await asyncio.wait_for(
                agent.astream_chat(query),
                timeout=timeout
            )
            
            full_response = ""
            async for token in response_stream.async_response_gen():
                full_response += token
                # Could emit streaming updates here
            
            return full_response
        except:
            # Fallback to non-streaming
            return await agent.achat(query)
    
    def _get_timeout_for_strategy(self, strategy: Dict[str, Any]) -> int:
        """Get timeout based on strategy"""
        approach = strategy.get("approach", "standard")
        
        timeouts = {
            "direct": 30,
            "standard": 60,
            "enhanced": 120,
            "context_first": 90
        }
        
        # Override with config if set
        if settings.AGENTIC_TIMEOUT_SECONDS > 0:
            return settings.AGENTIC_TIMEOUT_SECONDS
        
        return timeouts.get(approach, 60)
    
    def _extract_partial_from_memory(self, agent: ReActAgent) -> Optional[str]:
        """Extract partial results from agent's memory"""
        try:
            if hasattr(agent, 'memory') and agent.memory:
                # Get all messages from memory
                all_messages = agent.memory.get_all()
                
                # Look for assistant messages with substantial content
                useful_content = []
                for msg in reversed(all_messages):
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        if msg.role == "assistant" and len(str(msg.content)) > 50:
                            content = str(msg.content)
                            # Skip internal reasoning
                            if not content.startswith("Thought:") and not content.startswith("Action:"):
                                useful_content.append(content[:500])
                                if len(useful_content) >= 2:  # Get last 2 meaningful responses
                                    break
                
                if useful_content:
                    return "\n\n".join(reversed(useful_content))
        except Exception as e:
            logger.debug(f"Failed to extract partial results: {e}")
        
        return None
    
    async def _apply_native_fallbacks(
        self,
        original_query: str,
        enhanced_query: str,
        analysis: Dict[str, Any],
        failed_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply fallback strategies using native features"""
        
        # Strategy 1: Return partial results if available
        partial = failed_result.get("partial_results")
        if partial and len(partial) > 100:
            return {
                "success": True,
                "response": f"Partial analysis:\n{partial}",
                "method": "partial_fallback",
                "partial": True
            }
        
        # Strategy 2: Try simplified agent
        if analysis["complexity"] in ["complex", "research"]:
            logger.info("Trying simplified agent approach")
            
            # Create simple agent with fewer iterations
            simple_memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
            simple_agent = ReActAgent.from_tools(
                tools=self.tools[:3] if len(self.tools) > 3 else self.tools,
                llm=self.llm,
                memory=simple_memory,
                max_iterations=5,
                verbose=False,
                handle_parsing_errors=True
            )
            
            try:
                response = await asyncio.wait_for(
                    simple_agent.achat(original_query),
                    timeout=30
                )
                return {
                    "success": True,
                    "response": str(response),
                    "method": "simplified_fallback"
                }
            except:
                pass
        
        # Strategy 3: Direct tool call for simple searches
        if analysis["query_type"] == "search" and analysis["complexity"] == "simple":
            return await self._try_direct_tool_call(original_query)
        
        # Final fallback: helpful suggestions
        return self.query_processor.create_fallback_response(
            original_query,
            failed_result.get("error", "Unknown error"),
            [partial] if partial else None
        )
    
    async def _try_direct_tool_call(self, query: str) -> Dict[str, Any]:
        """Try direct tool call for simple queries"""
        # Find a search tool
        search_tool = None
        for tool in self.tools:
            if hasattr(tool, 'metadata') and hasattr(tool.metadata, 'name'):
                if 'search' in tool.metadata.name.lower():
                    search_tool = tool
                    break
        
        if search_tool:
            try:
                result = await search_tool.acall(query)
                return {
                    "success": True,
                    "response": f"Direct search result:\n{result}",
                    "method": "direct_tool_fallback"
                }
            except:
                pass
        
        return {"success": False, "error": "No suitable tool found"}
    
    def _create_error_response(self, query: str, error: Exception) -> Dict[str, Any]:
        """Create helpful error response"""
        error_msg = str(error)
        
        suggestions = [
            "Try rephrasing your question more specifically",
            "Break down complex queries into simpler parts",
            "Include specific file names or function names if known"
        ]
        
        if "timeout" in error_msg.lower():
            suggestions.insert(0, "The query was too complex - try a simpler version")
        
        return {
            "success": False,
            "response": f"An error occurred: {error_msg}",
            "error": error_msg,
            "suggestions": suggestions,
            "method": "error_response"
        }