# 🧠 Conversation Memory Strategies Guide

## Current Implementation Analysis

Your system uses a **sliding window approach** with the last 6 messages. Here's how different memory strategies work:

## 1. 📝 Simple Message History (Current)

```python
# Current implementation
history = session.get("conversation_history", [])
conversation_history_for_prompt = "\n".join(
    [f"{msg['role'].upper()}: {msg['content']}" for msg in history[-6:]]
)
```

**✅ Pros:**
- Simple to implement
- Maintains recent context
- Predictable token usage
- Good for short conversations

**❌ Cons:**
- Loses older context
- No semantic understanding of importance
- Fixed window size might cut important info

## 2. 🎯 Smart Context Selection

```python
def get_smart_context(history, max_tokens=4000):
    """Select most relevant messages based on content and recency"""
    
    # Always include last 2-3 messages for immediate context
    recent_messages = history[-3:]
    
    # Calculate importance scores for older messages
    important_messages = []
    for msg in history[:-3]:
        score = calculate_importance(msg)
        if score > threshold:
            important_messages.append((score, msg))
    
    # Sort by importance and fit within token limit
    important_messages.sort(reverse=True)
    
    context = []
    token_count = 0
    
    # Add recent messages first
    for msg in recent_messages:
        context.append(msg)
        token_count += estimate_tokens(msg['content'])
    
    # Add important older messages if space allows
    for score, msg in important_messages:
        estimated_tokens = estimate_tokens(msg['content'])
        if token_count + estimated_tokens <= max_tokens:
            context.insert(-len(recent_messages), msg)
            token_count += estimated_tokens
    
    return context

def calculate_importance(message):
    """Score message importance based on various factors"""
    score = 0
    
    # Code snippets are important
    if '```' in message['content']:
        score += 10
    
    # Questions are important
    if '?' in message['content']:
        score += 5
    
    # Error messages are important
    if any(word in message['content'].lower() for word in ['error', 'bug', 'issue', 'problem']):
        score += 8
    
    # Length indicates detail
    score += len(message['content']) / 100
    
    return score
```

## 3. 📚 Hierarchical Summarization

```python
async def get_summarized_context(history, llm_client):
    """Create hierarchical summaries of conversation chunks"""
    
    if len(history) <= 6:
        return history  # No need to summarize short conversations
    
    # Split into chunks
    chunks = [history[i:i+4] for i in range(0, len(history)-6, 4)]
    recent_messages = history[-6:]  # Always keep recent messages
    
    summaries = []
    for chunk in chunks:
        chunk_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chunk])
        
        summary_prompt = f"""
        Summarize this conversation chunk in 2-3 sentences, preserving key technical details:
        
        {chunk_text}
        """
        
        summary = await llm_client.process_prompt(summary_prompt, "summarize")
        summaries.append({
            "role": "system",
            "content": f"[SUMMARY]: {summary.prompt}",
            "timestamp": chunk[-1]['timestamp']
        })
    
    # Combine summaries with recent messages
    return summaries + recent_messages
```

## 4. 🔍 RAG-Enhanced Memory

```python
class ConversationRAG:
    def __init__(self):
        self.vector_store = {}  # Store conversation embeddings
        
    async def store_message(self, session_id, message):
        """Store message with semantic embeddings"""
        embedding = await self.get_embedding(message['content'])
        
        self.vector_store[f"{session_id}_{message['timestamp']}"] = {
            "message": message,
            "embedding": embedding,
            "session_id": session_id
        }
    
    async def get_relevant_context(self, session_id, current_query, max_messages=10):
        """Retrieve semantically similar messages from conversation history"""
        query_embedding = await self.get_embedding(current_query)
        
        # Find similar messages in this session
        session_messages = [
            (key, data) for key, data in self.vector_store.items() 
            if data['session_id'] == session_id
        ]
        
        # Calculate similarity scores
        similarities = []
        for key, data in session_messages:
            similarity = cosine_similarity(query_embedding, data['embedding'])
            similarities.append((similarity, data['message']))
        
        # Sort by relevance and return top messages
        similarities.sort(reverse=True)
        return [msg for _, msg in similarities[:max_messages]]
```

## 5. 💾 Persistent Memory with Database

```python
class DatabaseMemory:
    def __init__(self, db_connection):
        self.db = db_connection
    
    async def store_conversation(self, session_id, message):
        """Store in database with metadata"""
        await self.db.execute("""
            INSERT INTO conversation_history 
            (session_id, role, content, timestamp, tokens, importance_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            message['role'],
            message['content'],
            message['timestamp'],
            self.count_tokens(message['content']),
            self.calculate_importance(message)
        ))
    
    async def get_context(self, session_id, strategy="hybrid"):
        """Get conversation context using different strategies"""
        
        if strategy == "recent":
            return await self.get_recent_messages(session_id, limit=6)
        elif strategy == "important":
            return await self.get_important_messages(session_id, limit=8)
        elif strategy == "hybrid":
            recent = await self.get_recent_messages(session_id, limit=3)
            important = await self.get_important_messages(session_id, limit=5)
            return self.merge_contexts(recent, important)
```

## 🚀 Recommended Improvements for Your System

### Option 1: Enhanced Smart Window (Easy)
```python
def get_enhanced_context(history, max_messages=8):
    """Improved version of your current approach"""
    
    if len(history) <= max_messages:
        return history
    
    # Always include last 3 messages
    recent = history[-3:]
    older = history[:-3]
    
    # Score older messages
    scored_messages = []
    for msg in older:
        score = 0
        content = msg['content'].lower()
        
        # Code blocks are important
        if '```' in msg['content']:
            score += 10
        
        # Questions and errors are important
        if '?' in content or any(word in content for word in ['error', 'issue', 'bug']):
            score += 5
        
        # System messages are important
        if msg['role'] == 'system':
            score += 8
        
        scored_messages.append((score, msg))
    
    # Sort and take top messages
    scored_messages.sort(reverse=True)
    important_older = [msg for _, msg in scored_messages[:max_messages-3]]
    
    # Combine and sort by timestamp
    combined = important_older + recent
    combined.sort(key=lambda x: x['timestamp'])
    
    return combined
```

### Option 2: Add Message Compression (Medium)
```python
async def compress_old_messages(history, llm_client, compression_threshold=10):
    """Compress older messages when conversation gets long"""
    
    if len(history) <= compression_threshold:
        return history
    
    # Keep recent messages as-is
    recent_messages = history[-6:]
    old_messages = history[:-6]
    
    # Group old messages for compression
    chunks = [old_messages[i:i+4] for i in range(0, len(old_messages), 4)]
    
    compressed = []
    for chunk in chunks:
        if len(chunk) >= 3:  # Only compress substantial chunks
            chunk_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chunk])
            
            compression_prompt = f"""
            Compress this conversation chunk into key points while preserving technical details:
            
            {chunk_text}
            
            Format: [COMPRESSED] Key points: 1) ... 2) ... 3) ...
            """
            
            summary = await llm_client.process_prompt(compression_prompt, "summarize")
            compressed.append({
                "role": "system",
                "content": summary.prompt,
                "timestamp": chunk[-1]['timestamp']
            })
        else:
            compressed.extend(chunk)
    
    return compressed + recent_messages
```

## 🎯 Which Strategy Should You Use?

| **Strategy** | **Best For** | **Complexity** | **Performance** |
|--------------|--------------|----------------|-----------------|
| Simple Window (Current) | Short conversations | Low | Fast |
| Smart Context | Medium conversations | Medium | Good |
| Hierarchical Summary | Long conversations | High | Slower |
| RAG-Enhanced | Complex topics | High | Medium |
| Database + Compression | Production apps | Very High | Variable |

## 🛠️ Implementation Recommendation

For your current system, I recommend **Option 1 (Enhanced Smart Window)** because:

1. **Easy to implement** - minimal changes to existing code
2. **Maintains performance** - no additional API calls
3. **Better context selection** - preserves important messages
4. **Gradual improvement** - can be enhanced incrementally

Would you like me to implement this enhanced approach in your system? 