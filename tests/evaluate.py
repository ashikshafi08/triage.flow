#!/usr/bin/env python3
"""
Evaluation harness for Issue-Aware RAG system.
Reads queries from a CSV, runs them against the RAG system,
and calculates Recall@K and Mean Reciprocal Rank (MRR).
"""

import asyncio
import csv
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Configure logging to see debug information
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path and load environment variables
sys.path.append('src')
load_dotenv()

# Ensure necessary environment variables are set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
    sys.exit(1)
# GITHUB_TOKEN is checked within the components, but good to be aware.

# pylint: disable=wrong-import-position
from src.issue_rag import IssueAwareRAG
# pylint: enable=wrong-import-position

# Configuration
REPO_OWNER = "huggingface"
REPO_NAME = "smolagents"
CSV_FILE = "smolagents_eval_queries.csv"
OUTPUT_MD_FILE = "smolagents_evaluation_results_with_deepseek.md"
K_VALUES = [1, 3, 5]  # K values for Recall@K

async def load_eval_queries(file_path: str) -> list:
    """Loads evaluation queries from a CSV file."""
    queries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'query_text' in row and 'expected_issue_number' in row:
                    try:
                        queries.append({
                            "query_text": row["query_text"],
                            "expected_issue_number": int(row["expected_issue_number"])
                        })
                    except ValueError:
                        print(f"Warning: Skipping row with invalid expected_issue_number: {row}")
                else:
                    print(f"Warning: Skipping row with missing columns: {row}")
    except FileNotFoundError:
        print(f"Error: Evaluation CSV file not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        sys.exit(1)
    if not queries:
        print(f"Error: No valid queries loaded from {file_path}. Ensure CSV has 'query_text' and 'expected_issue_number' columns.")
        sys.exit(1)
    return queries

def calculate_metrics(results: list, k_values: list) -> dict:
    """Calculates Recall@K and MRR@K."""
    metrics = {}
    num_queries = len(results)
    if num_queries == 0:
        return {f"Recall@{k}": 0.0 for k in k_values}, {f"MRR@{k}": 0.0 for k in k_values}

    reciprocal_ranks = {k: [] for k in k_values}
    hits = {k: 0 for k in k_values}

    for result in results:
        expected_issue = result["expected_issue_number"]
        retrieved_issue_ids = [ret_issue.issue.id for ret_issue in result["retrieved_issues"]]

        for k in k_values:
            found_at_k = False
            rank_at_k = 0
            top_k_retrieved = retrieved_issue_ids[:k]
            if expected_issue in top_k_retrieved:
                hits[k] += 1
                found_at_k = True
                rank_at_k = top_k_retrieved.index(expected_issue) + 1
            
            if found_at_k:
                reciprocal_ranks[k].append(1.0 / rank_at_k)
            else:
                reciprocal_ranks[k].append(0.0)
    
    for k in k_values:
        metrics[f"Recall@{k}"] = hits[k] / num_queries if num_queries > 0 else 0.0
        metrics[f"MRR@{k}"] = sum(reciprocal_ranks[k]) / num_queries if num_queries > 0 else 0.0
        
    return metrics

async def run_evaluation():
    """Main function to run the evaluation."""
    print(f"🚀 Starting RAG Evaluation for {REPO_OWNER}/{REPO_NAME}")
    print("=" * 60)

    eval_queries = await load_eval_queries(CSV_FILE)
    print(f"Loaded {len(eval_queries)} evaluation queries from {CSV_FILE}")

    print("\n📡 Initializing Issue-Aware RAG system...")
    start_init_time = time.time()
    issue_rag = IssueAwareRAG(REPO_OWNER, REPO_NAME)
    # Use a smaller max_issues_for_patch_linkage for faster evaluation runs if index needs rebuilding
    # Set force_rebuild=False to use cached index if available, for faster repeated evaluations.
    # Set force_rebuild=True if you want to ensure the index is fresh for each eval.
    await issue_rag.initialize(force_rebuild=True, max_issues_for_patch_linkage=None)
    init_time = time.time() - start_init_time
    print(f"✅ RAG system initialized in {init_time:.2f} seconds.")

    all_results = []
    max_k = max(K_VALUES)

    print(f"\n⚙️  Running {len(eval_queries)} queries (retrieving top {max_k} for evaluation)...")
    total_query_time = 0

    for i, query_item in enumerate(eval_queries):
        query_text = query_item["query_text"]
        expected_issue = query_item["expected_issue_number"]
        
        print(f"\n[{i+1}/{len(eval_queries)}] Query: \"{query_text}\" (Expected: #{expected_issue})")
        
        start_query_time = time.time()
        try:
            # Get issue context, retrieving enough items for the largest K
            context = await issue_rag.get_issue_context(query_text, max_issues=max_k, include_patches=False)
            query_time = time.time() - start_query_time
            total_query_time += query_time

            retrieved_issues = context.related_issues
            retrieved_ids = [r.issue.id for r in retrieved_issues]
            print(f"   Retrieved {len(retrieved_issues)} issues in {query_time:.3f}s: {retrieved_ids}")

            all_results.append({
                "query_text": query_text,
                "expected_issue_number": expected_issue,
                "retrieved_issues": retrieved_issues,
                "retrieved_issue_ids_with_similarity": [
                    {"id": r.issue.id, "similarity": f"{r.similarity:.3f}"} for r in retrieved_issues
                ]
            })
        except Exception as e:
            print(f"   ⚠️ Error processing query: {e}")
            all_results.append({
                "query_text": query_text,
                "expected_issue_number": expected_issue,
                "retrieved_issues": [],
                "retrieved_issue_ids_with_similarity": [],
                "error": str(e)
            })
        await asyncio.sleep(0.1) # Small delay between API calls if any are made internally

    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(all_results, K_VALUES)

    print("\n📈 Evaluation Metrics:")
    print("-" * 40)
    for k_val in K_VALUES:
        print(f"Recall@{k_val}: {metrics[f'Recall@{k_val}']:.4f}")
        print(f"MRR@{k_val}:    {metrics[f'MRR@{k_val}']:.4f}")
    
    avg_query_time = total_query_time / len(eval_queries) if eval_queries else 0
    print(f"\n⏱️  Average query processing time: {avg_query_time:.3f}s")

    # Save detailed results to Markdown
    print(f"\n💾 Saving detailed results to {OUTPUT_MD_FILE}...")
    md_content = f"# RAG Evaluation Results for {REPO_OWNER}/{REPO_NAME}\n\n"
    md_content += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    md_content += f"Total Queries: {len(eval_queries)}\n"
    md_content += f"RAG Initialization Time: {init_time:.2f}s\n"
    md_content += f"Average Query Time: {avg_query_time:.3f}s\n\n"

    md_content += "## Summary Metrics\n"
    md_content += "| Metric         | Value   |\n"
    md_content += "|----------------|---------|\n"
    for k_val in K_VALUES:
        md_content += f"| Recall@{k_val}     | {metrics[f'Recall@{k_val}']:.4f} |\n"
        md_content += f"| MRR@{k_val}        | {metrics[f'MRR@{k_val}']:.4f} |\n"
    md_content += "\n"

    md_content += "## Detailed Results\n"
    for result in all_results:
        md_content += f"### Query: \"{result['query_text']}\"\n"
        md_content += f"- Expected Issue: `#{result['expected_issue_number']}`\n"
        if "error" in result:
            md_content += f"- **Error**: {result['error']}\n"
        else:
            md_content += "- Retrieved Issues (ID | Similarity):\n"
            if result["retrieved_issue_ids_with_similarity"]:
                for item in result["retrieved_issue_ids_with_similarity"]:
                    md_content += f"  - `#{item['id']}` | {item['similarity']}"
                    if item['id'] == result['expected_issue_number']:
                        md_content += " **(Correct)**"
                    md_content += "\n"
            else:
                md_content += "  - None found\n"
        md_content += "\n---\n"

    with open(OUTPUT_MD_FILE, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"✅ Detailed results saved.")

    print("\n✨ Evaluation complete!")

if __name__ == "__main__":
    if not os.getenv("GITHUB_TOKEN"):
        print("⚠️  GITHUB_TOKEN not found - API rate limits may apply during indexing if cache is cold.")
    asyncio.run(run_evaluation())
