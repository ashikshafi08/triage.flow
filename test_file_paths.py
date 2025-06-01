#!/usr/bin/env python3
"""
Test script to verify file path extraction is working correctly
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

from src.local_rag import LocalRepoContextExtractor

# Load environment variables
load_dotenv()

async def test_file_paths():
    """Test that file paths are extracted correctly from the repository"""
    
    # Initialize the repo context extractor
    repo_extractor = LocalRepoContextExtractor()
    
    # Test with huggingface/trl repository
    repo_url = "https://github.com/huggingface/trl.git"
    branch = "main"
    
    try:
        print(f"Loading repository from {repo_url}...")
        await repo_extractor.load_repository(repo_url, branch)
        print("✅ Repository loaded successfully")
        
        # Test query about trainers
        query = "PPOTrainer, DPOTrainer, SFTTrainer implementation files"
        
        print(f"\nQuerying: {query}")
        context = await repo_extractor.get_relevant_context(query)
        
        print(f"\n📊 Results:")
        print(f"- Found {len(context['sources'])} relevant sources")
        print(f"- Repository info: {context['repo_info']}")
        
        print(f"\n📁 File paths found:")
        for i, source in enumerate(context['sources'][:10], 1):
            file_path = source.get('file', 'UNKNOWN')
            language = source.get('language', 'unknown')
            print(f"{i:2d}. {file_path} ({language})")
        
        # Check if we're getting proper relative paths (not absolute or invented ones)
        print(f"\n🔍 Path Analysis:")
        for source in context['sources'][:5]:
            file_path = source.get('file', 'UNKNOWN')
            
            # Check for common path issues
            if file_path.startswith('/tmp/') or file_path.startswith('/var/'):
                print(f"❌ ABSOLUTE PATH DETECTED: {file_path}")
            elif file_path.startswith('src/trl/') and 'trl' not in file_path.replace('src/trl/', ''):
                print(f"❌ POTENTIALLY HALLUCINATED PATH: {file_path}")
            elif file_path == 'UNKNOWN':
                print(f"❌ UNKNOWN FILE PATH")
            else:
                print(f"✅ Good relative path: {file_path}")
        
        # Test the formatted context
        from src.llm_client import format_rag_context_for_llm
        formatted_context = format_rag_context_for_llm(context)
        
        print(f"\n📝 Formatted Context Preview (first 500 chars):")
        print("-" * 50)
        print(formatted_context[:500] + "..." if len(formatted_context) > 500 else formatted_context)
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_file_paths()) 