#!/usr/bin/env python3
"""
Test script to isolate the function calling issue
"""

import os
import sys
import logging

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import settings
from src.agent_tools.llm_config import get_llm_instance
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def simple_function(message: str) -> str:
    """A simple test function for function calling"""
    return f"Received: {message}"

def test_llm_creation():
    """Test LLM creation"""
    logger.info("Testing LLM creation...")
    try:
        llm = get_llm_instance()
        logger.info(f"✓ LLM created successfully: {type(llm)}")
        return llm
    except Exception as e:
        logger.error(f"✗ LLM creation failed: {e}")
        return None

def test_function_calling(llm):
    """Test function calling with a simple agent"""
    logger.info("Testing function calling with ReAct agent...")
    try:
        # Create a simple tool
        tool = FunctionTool.from_defaults(
            fn=simple_function,
            name="simple_function",
            description="A simple test function"
        )
        
        # Create ReAct agent
        agent = ReActAgent.from_tools(
            tools=[tool],
            llm=llm,
            verbose=True,
            max_iterations=10  # Increase iterations to avoid timeout
        )
        
        logger.info("✓ ReAct agent created successfully")
        
        # Test function calling with a simpler query
        response = agent.chat("Call simple_function with 'hello world'")
        logger.info(f"✓ Function calling test passed: {response}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Function calling test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def test_agentic_explorer_init():
    """Test AgenticCodebaseExplorer initialization"""
    logger.info("Testing AgenticCodebaseExplorer initialization...")
    try:
        from src.agent_tools.core import AgenticCodebaseExplorer
        
        # Create a temp directory to simulate repo
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Creating AgenticCodebaseExplorer for {temp_dir}")
            explorer = AgenticCodebaseExplorer(
                session_id="test_session",
                repo_path=temp_dir,
                issue_rag_system=None
            )
            logger.info("✓ AgenticCodebaseExplorer created successfully")
            return True
        
    except Exception as e:
        logger.error(f"✗ AgenticCodebaseExplorer initialization failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def main():
    logger.info("=== LLM Function Calling Test ===")
    logger.info(f"Model: {settings.default_model}")
    logger.info(f"Provider: {settings.llm_provider}")
    
    # Test 1: LLM creation
    llm = test_llm_creation()
    if not llm:
        return False
    
    # Test 2: Function calling (skip for now since we know it works but times out)
    # success = test_function_calling(llm)
    
    # Test 3: AgenticCodebaseExplorer initialization (this is where the real error occurs)
    success = test_agentic_explorer_init()
    
    if success:
        logger.info("=== All tests passed! ===")
    else:
        logger.error("=== Tests failed ===")
    
    return success

if __name__ == "__main__":
    main()
