#!/usr/bin/env python3
"""
Build patch linkage for Facebook React repository
This is the concrete deliverable for Task 1: patch_links.jsonl in index_dir/
Now includes Task 2: diff extraction and storage
"""

import asyncio
import logging
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables BEFORE any imports that might use them
load_dotenv()

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.patch_linkage import PatchLinkageBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Build patch linkage for Facebook React"""
    if not os.getenv("GITHUB_TOKEN"):
        logger.error("GITHUB_TOKEN environment variable is required")
        return 1
    
    logger.info("Starting Facebook React patch linkage build...")
    logger.info("This will:")
    logger.info("1. Fetch closed issues and their linked PRs")
    logger.info("2. Save patch_links.jsonl")
    logger.info("3. Download diff files for each PR")
    logger.info("4. Extract and clean diff hunks for embedding")
    logger.info("5. Save diff_docs.jsonl with metadata")
    
    start_time = time.time()
    
    try:
        # Build patch linkage with diff downloading
        builder = PatchLinkageBuilder("facebook", "react")
        await builder.build_patch_linkage(
            max_issues=50,  # Start small for testing
            download_diffs=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Build completed successfully in {elapsed_time:.1f} seconds")
        
        # Show what was created
        index_dir = builder.index_dir
        logger.info(f"📁 Files created in {index_dir}:")
        
        patch_links_file = index_dir / "patch_links.jsonl"
        if patch_links_file.exists():
            with open(patch_links_file, 'r') as f:
                lines = f.readlines()
            logger.info(f"  📋 patch_links.jsonl: {len(lines)} patch links")
        
        diff_docs_file = index_dir / "diff_docs.jsonl"
        if diff_docs_file.exists():
            with open(diff_docs_file, 'r') as f:
                lines = f.readlines()
            logger.info(f"  📋 diff_docs.jsonl: {len(lines)} diff documents")
        
        diffs_dir = index_dir / "diffs"
        if diffs_dir.exists():
            diff_files = list(diffs_dir.glob("*.diff"))
            logger.info(f"  📁 diffs/: {len(diff_files)} .diff files")
            
            # Show sample file sizes
            if diff_files:
                sample_file = diff_files[0]
                size_kb = sample_file.stat().st_size / 1024
                logger.info(f"  📄 Sample diff size: {size_kb:.1f} KB ({sample_file.name})")
        
        logger.info("\n🎯 Next steps:")
        logger.info("1. Integrate diff docs into IssueIndexer")
        logger.info("2. Add patch-aware retrieval modes")
        logger.info("3. Build evaluation harness")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Build failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 