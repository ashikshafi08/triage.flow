"""
Production-grade conversation memory system inspired by ChatGPT/Claude
"""
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import re

@dataclass
class MessageImportance:
    """Scoring system for message importance"""
    base_score: float = 0.0
    code_bonus: float = 0.0
    question_bonus: float = 0.0
    error_bonus: float = 0.0
    length_bonus: float = 0.0
    recency_bonus: float = 0.0
    total_score: float = 0.0

class ProductionMemoryManager:
    """
    ChatGPT/Claude-inspired conversation memory management
    """
    
    def __init__(self, max_context_tokens: int = 8000):
        self.max_context_tokens = max_context_tokens
        self.min_recent_messages = 3  # Always keep last 3 messages
        self.compression_threshold = 15  # Compress when > 15 messages
        
        # Token estimation (rough approximation)
        self.avg_chars_per_token = 4
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        return len(text) // self.avg_chars_per_token
    
    def calculate_message_importance(self, message: Dict[str, Any], position_from_end: int) -> MessageImportance:
        """
        Calculate importance score for a message (ChatGPT-style)
        """
        content = message.get('content', '')
        role = message.get('role', '')
        
        importance = MessageImportance()
        
        # Base scores by role
        role_scores = {
            'system': 10.0,      # System messages are always important
            'assistant': 3.0,    # Assistant responses have context
            'user': 5.0          # User messages drive conversation
        }
        importance.base_score = role_scores.get(role, 1.0)
        
        # Code blocks are highly important
        if '```' in content:
            importance.code_bonus = 8.0
        elif '`' in content:
            importance.code_bonus = 3.0
        
        # Questions are important for context
        question_indicators = ['?', 'how', 'what', 'why', 'when', 'where', 'which']
        if any(indicator in content.lower() for indicator in question_indicators):
            importance.question_bonus = 4.0
        
        # Error messages and problems are crucial
        error_indicators = ['error', 'bug', 'issue', 'problem', 'fail', 'exception', 'traceback']
        if any(indicator in content.lower() for indicator in error_indicators):
            importance.error_bonus = 6.0
        
        # Length indicates detail (but with diminishing returns)
        length = len(content)
        if length > 500:
            importance.length_bonus = min(3.0, length / 500)
        
        # Recency bonus (more recent = more important)
        importance.recency_bonus = max(0, 5.0 - (position_from_end * 0.3))
        
        # Calculate total
        importance.total_score = (
            importance.base_score + 
            importance.code_bonus + 
            importance.question_bonus + 
            importance.error_bonus + 
            importance.length_bonus + 
            importance.recency_bonus
        )
        
        return importance
    
    def get_context_aware_history(self, conversation_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get conversation context using ChatGPT-inspired smart selection
        """
        if not conversation_history:
            return []
        
        # If conversation is short, include everything
        if len(conversation_history) <= 8:
            return conversation_history
        
        # Always include the most recent messages
        recent_messages = conversation_history[-self.min_recent_messages:]
        older_messages = conversation_history[:-self.min_recent_messages]
        
        # Calculate importance scores for older messages
        scored_messages = []
        for i, message in enumerate(older_messages):
            position_from_end = len(older_messages) - i + self.min_recent_messages
            importance = self.calculate_message_importance(message, position_from_end)
            scored_messages.append((importance.total_score, message, i))
        
        # Sort by importance (highest first)
        scored_messages.sort(reverse=True, key=lambda x: x[0])
        
        # Select messages that fit within token budget
        selected_older = []
        current_tokens = sum(self.estimate_tokens(msg['content']) for msg in recent_messages)
        
        for score, message, original_index in scored_messages:
            message_tokens = self.estimate_tokens(message['content'])
            
            if current_tokens + message_tokens <= self.max_context_tokens:
                selected_older.append((original_index, message))
                current_tokens += message_tokens
            else:
                break
        
        # Sort selected older messages by original order
        selected_older.sort(key=lambda x: x[0])
        older_context = [msg for _, msg in selected_older]
        
        # Combine older context with recent messages
        return older_context + recent_messages
    
    async def compress_conversation_chunks(self, conversation_history: List[Dict[str, Any]], llm_client) -> List[Dict[str, Any]]:
        """
        Compress old conversation chunks when history gets very long (Claude-style)
        """
        if len(conversation_history) <= self.compression_threshold:
            return conversation_history
        
        # Keep recent messages as-is
        recent_messages = conversation_history[-8:]
        old_messages = conversation_history[:-8]
        
        # Group old messages into coherent chunks
        chunks = self._create_semantic_chunks(old_messages)
        
        compressed_chunks = []
        for chunk in chunks:
            if len(chunk) >= 3:  # Only compress substantial chunks
                try:
                    compressed = await self._compress_chunk(chunk, llm_client)
                    compressed_chunks.append(compressed)
                except Exception as e:
                    print(f"Error compressing chunk: {e}")
                    # Fall back to including original messages
                    compressed_chunks.extend(chunk)
            else:
                compressed_chunks.extend(chunk)
        
        return compressed_chunks + recent_messages
    
    def _create_semantic_chunks(self, messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group messages into semantically coherent chunks
        """
        chunks = []
        current_chunk = []
        chunk_size_limit = 5
        
        for message in messages:
            current_chunk.append(message)
            
            # Split chunks at natural boundaries
            content = message.get('content', '').lower()
            
            # End chunk after code blocks (natural completion point)
            if '```' in message.get('content', ''):
                chunks.append(current_chunk)
                current_chunk = []
            # End chunk after questions (natural transition point)
            elif '?' in content and len(current_chunk) >= 2:
                chunks.append(current_chunk)
                current_chunk = []
            # End chunk at size limit
            elif len(current_chunk) >= chunk_size_limit:
                chunks.append(current_chunk)
                current_chunk = []
        
        # Add remaining messages
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _compress_chunk(self, chunk: List[Dict[str, Any]], llm_client) -> Dict[str, Any]:
        """
        Compress a conversation chunk while preserving key information
        """
        chunk_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in chunk
        ])
        
        compression_prompt = f"""
        Summarize this conversation segment in 2-3 sentences while preserving:
        1. Key technical details and code snippets
        2. Important questions and their answers
        3. Decisions made or conclusions reached
        4. Any unresolved issues or next steps
        
        Conversation segment:
        {chunk_text}
        
        Create a concise summary that maintains the essential context:
        """
        
        # Use the existing LLM client to compress
        from .models import PromptResponse
        response = await llm_client.process_prompt(
            compression_prompt, 
            "summarize", 
            context=None
        )
        
        return {
            "role": "system",
            "content": f"[SUMMARY]: {response.prompt}",
            "timestamp": chunk[-1].get('timestamp', datetime.now()),
            "original_message_count": len(chunk)
        }
    
    def format_context_for_llm(self, context_messages: List[Dict[str, Any]]) -> str:
        """
        Format conversation context for LLM (ChatGPT-style)
        """
        if not context_messages:
            return ""
        
        # Group by conversation flow
        formatted_parts = []
        
        for message in context_messages:
            role = message.get('role', '').upper()
            content = message.get('content', '')
            timestamp = message.get('timestamp')
            
            # Add timestamp for system summaries to show temporal context
            if role == 'SYSTEM' and '[SUMMARY]' in content:
                time_str = timestamp.strftime('%H:%M') if timestamp else 'earlier'
                formatted_parts.append(f"[{time_str}] {content}")
            else:
                formatted_parts.append(f"{role}: {content}")
        
        return "\n".join(formatted_parts)
    
    def get_memory_stats(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about conversation memory usage
        """
        if not conversation_history:
            return {"total_messages": 0, "total_tokens": 0, "memory_strategy": "empty"}
        
        total_messages = len(conversation_history)
        total_tokens = sum(self.estimate_tokens(msg.get('content', '')) for msg in conversation_history)
        
        # Determine current strategy
        if total_messages <= 8:
            strategy = "full_history"
        elif total_messages <= self.compression_threshold:
            strategy = "smart_selection"
        else:
            strategy = "compression_active"
        
        return {
            "total_messages": total_messages,
            "total_tokens": total_tokens,
            "context_tokens": min(total_tokens, self.max_context_tokens),
            "memory_strategy": strategy,
            "compression_ratio": total_tokens / self.max_context_tokens if total_tokens > 0 else 0
        }

# Integration helper for your existing system
class ConversationContextManager:
    """
    Drop-in replacement for your current conversation history logic
    """
    
    def __init__(self, max_context_tokens: int = 8000):
        self.memory_manager = ProductionMemoryManager(max_context_tokens)
    
    async def get_conversation_context(
        self, 
        conversation_history: List[Dict[str, Any]], 
        llm_client,
        use_compression: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Get formatted conversation context for LLM
        
        Returns:
            - Formatted context string for LLM
            - Memory statistics
        """
        
        # Apply compression if needed and enabled
        if use_compression and len(conversation_history) > self.memory_manager.compression_threshold:
            processed_history = await self.memory_manager.compress_conversation_chunks(
                conversation_history, llm_client
            )
        else:
            processed_history = conversation_history
        
        # Get smart context selection
        context_messages = self.memory_manager.get_context_aware_history(processed_history)
        
        # Format for LLM
        formatted_context = self.memory_manager.format_context_for_llm(context_messages)
        
        # Get statistics
        stats = self.memory_manager.get_memory_stats(conversation_history)
        stats['context_messages_used'] = len(context_messages)
        
        return formatted_context, stats 