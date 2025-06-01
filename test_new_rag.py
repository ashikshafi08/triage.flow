# test_new_rag.py
import asyncio
import os
import sys
from pathlib import Path

# Add the parent directory to the sys.path to allow importing src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from new_rag import LocalRepoContextExtractor # Import from your new file
from config import settings # Assuming config.py is in src/
from local_repo_loader import clone_repo_to_temp_persistent # Assuming this is in src/

async def run_rag_test():
    # --- Configuration ---
    # Replace with a public GitHub repository URL for testing
    test_repo_url = "https://github.com/huggingface/smolagents.git" # A small, simple repo
    # test_repo_url = "https://github.com/langchain-ai/langchain.git" # A larger, more complex repo for performance testing
    test_branch = "main"
    test_query_code = "How does the main function work?"
    test_query_general = "What is the purpose of this repository? and how to use it to build an agent for news app?"

    print(f"--- Starting RAG Test for {test_repo_url} ---")

    # Ensure OpenAI API key is set for embeddings
    if not settings.openai_api_key:
        print("Error: OPENAI_API_KEY not set in .env file. Please set it to run this test.")
        return

    rag_extractor = LocalRepoContextExtractor()

    try:
        print(f"Loading repository: {test_repo_url} (branch: {test_branch})...")
        await rag_extractor.load_repository(test_repo_url, test_branch)
        print("Repository loaded successfully.")

        # --- Test 1: Code-specific query ---
        print(f"\n--- Querying for code: '{test_query_code}' ---")
        context_code = await rag_extractor.get_relevant_context(test_query_code)

        print("\nRetrieved Context for Code Query:")
        print(f"Response Summary: {context_code.get('response', 'N/A')}")
        print("\nSources (Code Chunks):")
        if context_code.get("sources"):
            for i, source in enumerate(context_code["sources"]):
                print(f"  {i+1}. File: {source.get('file', 'UNKNOWN')}")
                print(f"     Language: {source.get('language', 'UNKNOWN')}")
                print(f"     Content Preview:\n{source.get('content', 'N/A')}\n---")
        else:
            print("No sources found for this query.")

        # --- Test 2: General purpose query ---
        print(f"\n--- Querying for general info: '{test_query_general}' ---")
        context_general = await rag_extractor.get_relevant_context(test_query_general)

        print("\nRetrieved Context for General Query:")
        print(f"Response Summary: {context_general.get('response', 'N/A')}")
        print("\nSources (General Chunks):")
        if context_general.get("sources"):
            for i, source in enumerate(context_general["sources"]):
                print(f"  {i+1}. File: {source.get('file', 'UNKNOWN')}")
                print(f"     Language: {source.get('language', 'UNKNOWN')}")
                print(f"     Content Preview:\n{source.get('content', 'N/A')}\n---")
        else:
            print("No sources found for this query.")

        # --- Test 3: Test with a specific file restriction (if applicable to your test repo) ---
        # Find a known file in your test_repo_url, e.g., "README.md" or a specific code file
        # For Spoon-Knife, let's try "README.md"
        print(f"\n--- Querying with file restriction: 'README.md' ---")
        context_restricted = await rag_extractor.get_relevant_context(
            "What is this project about?",
            restrict_files=["README.md"]
        )
        print("\nRetrieved Context for Restricted Query:")
        if context_restricted.get("sources"):
            for i, source in enumerate(context_restricted["sources"]):
                print(f"  {i+1}. File: {source.get('file', 'UNKNOWN')}")
                print(f"     Content Preview:\n{source.get('content', 'N/A')}\n---")
        else:
            print("No sources found for restricted query.")


    except Exception as e:
        print(f"An error occurred during the RAG test: {e}")
    finally:
        # Clean up the cloned repository if it's persistent
        if hasattr(rag_extractor, 'current_repo_path') and os.path.exists(rag_extractor.current_repo_path):
            print(f"\nCleaning up temporary repository: {rag_extractor.current_repo_path}")
            import shutil
            shutil.rmtree(rag_extractor.current_repo_path, ignore_errors=True)
            print("Cleanup complete.")

if __name__ == "__main__":
    asyncio.run(run_rag_test())
