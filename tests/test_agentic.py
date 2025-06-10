#!/usr/bin/env python3
"""
Test script for the agentic codebase explorer
"""

import asyncio
import os
import sys
import tempfile
import pytest
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.agent_tools import AgenticCodebaseExplorer

async def test_agentic_explorer():
    """Test the agentic explorer with the current repository"""
    
    # Use current directory as test repository
    repo_path = str(Path.cwd())
    session_id = "test_session"
    
    print(f"🤖 Testing Agentic Codebase Explorer")
    print(f"📁 Repository path: {repo_path}")
    print(f"🆔 Session ID: {session_id}")
    print("-" * 60)
    
    try:
        # Initialize the agentic explorer
        print("🔧 Initializing AgenticCodebaseExplorer...")
        explorer = AgenticCodebaseExplorer(session_id, repo_path)
        print("✅ Explorer initialized successfully!")
        
        # Test queries
        test_queries = [
            "What files are in the src directory?",
            "Search for 'RAG' in the codebase",
            "Analyze the structure of the main.py file",
            "Tell me about the overall architecture of this project"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Test Query {i}: {query}")
            print("-" * 40)
            
            try:
                response = await explorer.query(query)
                print(f"🤖 Response: {response[:500]}...")
                if len(response) > 500:
                    print("    (response truncated)")
                print("✅ Query completed successfully!")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
        
        print(f"\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Error initializing explorer: {e}")
        print(f"💡 Make sure you have set the required API keys:")
        print(f"   - OPENROUTER_API_KEY (if using OpenRouter)")
        print(f"   - OPENAI_API_KEY (if using OpenAI)")

if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: No API keys found!")
        print("🔑 Please set either OPENROUTER_API_KEY or OPENAI_API_KEY")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_agentic_explorer()) 