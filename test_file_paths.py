#!/usr/bin/env python3
"""
Test script to verify file path extraction and hybrid retrieval (BM25 + dense search)
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

async def test_hybrid_retrieval():
    """Test hybrid retrieval with both BM25 and dense search capabilities"""
    
    # Initialize the repo context extractor
    repo_extractor = LocalRepoContextExtractor()
    
    # Test with huggingface/trl repository
    repo_url = "https://github.com/huggingface/trl.git"
    branch = "main"
    
    try:
        print(f"Loading repository from {repo_url}...")
        await repo_extractor.load_repository(repo_url, branch)
        print("✅ Repository loaded successfully")
        
        # Test cases for different retrieval scenarios
        test_cases = [
            {
                "name": "Exact filename match (BM25)",
                "query": "ppo_trainer.py implementation",
                "expected_files": ["ppo_trainer.py"]
            },
            {
                "name": "Semantic search (Dense)",
                "query": "How does the PPO algorithm work in this codebase?",
                "expected_files": ["ppo_trainer.py", "ppo_config.py"]
            },
            {
                "name": "Hybrid search (both)",
                "query": "PPOTrainer, DPOTrainer, SFTTrainer implementation files",
                "expected_files": ["ppo_trainer.py", "dpo_trainer.py", "sft_trainer.py"]
            },
            {
                "name": "Code identifier search (BM25)",
                "query": "class PPOTrainer",
                "expected_files": ["ppo_trainer.py"]
            }
        ]
        
        for test_case in test_cases:
            print(f"\n🧪 Testing: {test_case['name']}")
            print(f"Query: {test_case['query']}")
            
            context = await repo_extractor.get_relevant_context(test_case['query'])
            
            print(f"\n📊 Results:")
            print(f"- Found {len(context['sources'])} relevant sources")
            
            print(f"\n📁 File paths found:")
            found_files = set()
            for i, source in enumerate(context['sources'][:10], 1):
                file_path = source.get('file', 'UNKNOWN')
                language = source.get('language', 'unknown')
                found_files.add(os.path.basename(file_path))
                print(f"{i:2d}. {file_path} ({language})")
            
            # Verify expected files are found
            print("\n✅ Expected files check:")
            for expected_file in test_case['expected_files']:
                if expected_file in found_files:
                    print(f"  ✓ Found {expected_file}")
                else:
                    print(f"  ✗ Missing {expected_file}")
            
            # Check path quality
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
            
            print("\n" + "="*80)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_hybrid_retrieval()) 