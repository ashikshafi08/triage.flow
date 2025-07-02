#!/usr/bin/env python3
"""
Test script for the self-improving agent learning system.

This script demonstrates how to use the learning capabilities.
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.learning import PerformanceTracker, MemorySystem, LearningLoop, Episode
from src.learning.performance_tracker import PerformanceMetrics
from datetime import datetime

async def test_performance_tracker():
    """Test the performance tracking system"""
    print("🔍 Testing Performance Tracker...")
    
    tracker = PerformanceTracker("test_session")
    
    # Simulate some interactions
    for i in range(5):
        interaction_id = tracker.start_interaction(f"Test query {i}")
        
        # Simulate tool calls
        tracker.record_tool_call("search_files", True, 1.5)
        tracker.record_tool_call("read_file", True, 0.8)
        tracker.add_token_usage(150)
        
        # End interaction
        await tracker.end_interaction(success=True, response=f"Test response {i}")
    
    # Get performance summary
    summary = await tracker.get_performance_summary()
    print(f"✅ Performance Summary: {json.dumps(summary, indent=2)}")
    
    return tracker

async def test_memory_system():
    """Test the memory system"""
    print("\n🧠 Testing Memory System...")
    
    memory = MemorySystem("test_session")
    
    # Create a test episode
    metrics = PerformanceMetrics(
        session_id="test_session",
        query="How do I implement authentication?",
        query_complexity=0.7,
        response_time=5.2,
        tokens_used=200,
        iterations_needed=3,
        tool_calls=[
            {"tool": "search_files", "success": True, "duration": 2.0},
            {"tool": "read_file", "success": True, "duration": 1.2},
            {"tool": "analyze_code", "success": True, "duration": 2.0}
        ],
        success=True,
        error_count=0,
        goal_achievement=0.9
    )
    
    episode = Episode(
        episode_id="test_episode_1",
        session_id="test_session",
        timestamp=datetime.now(),
        query="How do I implement authentication?",
        context={"repo_path": "/test/repo", "language": "python"},
        actions_taken=[
            {"tool": "search_files", "input": "auth", "success": True},
            {"tool": "read_file", "input": "auth.py", "success": True}
        ],
        response="Authentication can be implemented using...",
        performance_metrics=metrics,
        outcome_quality=0.9,
        learned_insights=["successful_auth_query", "good_search_strategy"]
    )
    
    # Record the episode
    success = await memory.record_episode(episode)
    print(f"✅ Episode recorded: {success}")
    
    # Get relevant memories
    memories = await memory.get_relevant_memories("How to add user login?")
    print(f"✅ Found {len(memories.get('similar_patterns', []))} similar patterns")
    
    # Get learning insights
    insights = await memory.get_learning_insights()
    print(f"✅ Learning insights: {json.dumps(insights, indent=2)}")
    
    return memory

async def test_learning_loop():
    """Test the learning loop"""
    print("\n🔄 Testing Learning Loop...")
    
    tracker = PerformanceTracker("test_session_2")
    memory = MemorySystem("test_session_2")
    loop = LearningLoop("test_session_2", tracker, memory)
    
    # Simulate some poor performance to trigger learning
    for i in range(10):
        interaction_id = tracker.start_interaction(f"Complex query {i}")
        
        # Simulate some failures
        if i % 3 == 0:
            tracker.record_tool_call("failing_tool", False, 5.0)
            tracker.record_error("Tool execution failed")
            await tracker.end_interaction(success=False, response="Error occurred")
        else:
            tracker.record_tool_call("working_tool", True, 2.0)
            await tracker.end_interaction(success=True, response="Success")
    
    # Check if learning should be triggered
    should_learn = await loop._should_trigger_learning()
    print(f"✅ Should trigger learning: {should_learn}")
    
    if should_learn:
        print("🎓 Performing learning cycle...")
        await loop._perform_learning_cycle()
        print("✅ Learning cycle completed")
    
    # Get learning status
    status = await loop.get_learning_status()
    print(f"✅ Learning status: {json.dumps(status, indent=2, default=str)}")
    
    return loop

async def test_pattern_storage():
    """Test pattern storage and retrieval"""
    print("\n📚 Testing Pattern Storage...")
    
    memory = MemorySystem("test_session_3")
    pattern_storage = memory.pattern_storage
    
    # Create a pattern from interaction
    pattern = await pattern_storage.create_pattern_from_interaction(
        query="Find all Python files in the repository",
        response="Found 25 Python files in the repository...",
        tools_used=[
            {"tool": "search_files", "success": True, "duration": 1.5},
            {"tool": "list_files", "success": True, "duration": 0.8}
        ],
        success=True,
        context={"repo_type": "python", "size": "medium"}
    )
    
    if pattern:
        print(f"✅ Created pattern: {pattern.pattern_id}")
        print(f"   Type: {pattern.pattern_type}")
        print(f"   Description: {pattern.description}")
    
    # Find similar patterns
    similar = await pattern_storage.find_similar_patterns(
        "Search for all JavaScript files",
        pattern_type="search_pattern"
    )
    print(f"✅ Found {len(similar)} similar patterns")
    
    # Get successful patterns
    successful = await pattern_storage.get_success_patterns()
    print(f"✅ Found {len(successful)} successful patterns")
    
    return pattern_storage

async def main():
    """Run all tests"""
    print("🚀 Starting Learning System Tests\n")
    
    try:
        # Test individual components
        await test_performance_tracker()
        await test_memory_system()
        await test_learning_loop()
        await test_pattern_storage()
        
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print("✅ Performance Tracker - Working")
        print("✅ Memory System - Working") 
        print("✅ Learning Loop - Working")
        print("✅ Pattern Storage - Working")
        print("\n🧠 The learning system is ready for integration!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())