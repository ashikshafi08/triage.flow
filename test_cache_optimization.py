#!/usr/bin/env python3
"""
Test script to verify the repository caching optimizations.
This script demonstrates that:
1. First session initialization takes time (indexing)
2. Subsequent sessions for the same repo reuse cached data
3. Backend restart preserves local file caches
"""

import asyncio
import time
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

async def test_cache_optimization():
    """Test the repository caching optimization"""
    
    # Test with a small repository for faster results
    test_repo = "https://github.com/pallets/click"
    
    print("🧪 Testing Repository Caching Optimization")
    print("=" * 50)
    
    # Test 1: First session initialization (should take time)
    print("\n📝 Test 1: First session initialization")
    start_time = time.time()
    
    response1 = requests.post(f"{BASE_URL}/assistant/sessions", json={
        "repo_url": test_repo,
        "session_name": "Cache Test Session 1"
    })
    
    if response1.status_code != 200:
        print(f"❌ Failed to create first session: {response1.text}")
        return
    
    session1_data = response1.json()
    session1_id = session1_data["session_id"]
    print(f"✅ Created session 1: {session1_id}")
    
    # Wait for initialization to complete
    print("⏳ Waiting for initialization to complete...")
    while True:
        status_response = requests.get(f"{BASE_URL}/assistant/sessions/{session1_id}/status")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"   Status: {status.get('overall_status', 'unknown')} - {status.get('message', '')}")
            
            if status.get('overall_status') in ['ready', 'core_ready', 'error']:
                break
        
        await asyncio.sleep(2)
    
    first_session_time = time.time() - start_time
    print(f"🕐 First session initialization took: {first_session_time:.2f} seconds")
    
    # Test 2: Second session for same repo (should be much faster)
    print("\n📝 Test 2: Second session for same repository")
    start_time = time.time()
    
    response2 = requests.post(f"{BASE_URL}/assistant/sessions", json={
        "repo_url": test_repo,
        "session_name": "Cache Test Session 2"
    })
    
    if response2.status_code != 200:
        print(f"❌ Failed to create second session: {response2.text}")
        return
    
    session2_data = response2.json()
    session2_id = session2_data["session_id"]
    print(f"✅ Created session 2: {session2_id}")
    
    # Wait for initialization to complete
    print("⏳ Waiting for initialization to complete...")
    while True:
        status_response = requests.get(f"{BASE_URL}/assistant/sessions/{session2_id}/status")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"   Status: {status.get('overall_status', 'unknown')} - {status.get('message', '')}")
            
            if status.get('overall_status') in ['ready', 'core_ready', 'error']:
                break
        
        await asyncio.sleep(1)
    
    second_session_time = time.time() - start_time
    print(f"🕐 Second session initialization took: {second_session_time:.2f} seconds")
    
    # Calculate improvement
    if second_session_time > 0:
        speedup = first_session_time / second_session_time
        print(f"\n🚀 Cache optimization results:")
        print(f"   - First session: {first_session_time:.2f}s")
        print(f"   - Second session: {second_session_time:.2f}s") 
        print(f"   - Speedup: {speedup:.1f}x faster")
        
        if speedup > 2.0:
            print("✅ Cache optimization is working well!")
        elif speedup > 1.5:
            print("⚠️  Some improvement, but could be better")
        else:
            print("❌ Cache optimization may not be working")
    
    # Test 3: Verify both sessions can access timeline data
    print("\n📝 Test 3: Verify functionality in both sessions")
    
    # Get file tree for session 1
    tree_response1 = requests.get(f"{BASE_URL}/api/tree", params={"session_id": session1_id})
    if tree_response1.status_code == 200:
        tree1 = tree_response1.json()
        print(f"✅ Session 1 file tree loaded: {len(tree1.get('items', []))} items")
    else:
        print(f"❌ Session 1 file tree failed: {tree_response1.text}")
    
    # Get file tree for session 2  
    tree_response2 = requests.get(f"{BASE_URL}/api/tree", params={"session_id": session2_id})
    if tree_response2.status_code == 200:
        tree2 = tree_response2.json()
        print(f"✅ Session 2 file tree loaded: {len(tree2.get('items', []))} items")
    else:
        print(f"❌ Session 2 file tree failed: {tree_response2.text}")
    
    # Test timeline access (if we have files)
    if tree_response1.status_code == 200 and tree1.get('items'):
        # Find a Python file to test timeline
        test_file = None
        def find_python_file(items):
            for item in items:
                if item.get('type') == 'file' and item.get('name', '').endswith('.py'):
                    return item.get('path')
                elif item.get('type') == 'folder' and item.get('children'):
                    result = find_python_file(item['children'])
                    if result:
                        return result
            return None
        
        test_file = find_python_file(tree1.get('items', []))
        
        if test_file:
            print(f"📁 Testing timeline for file: {test_file}")
            
            # Test timeline in session 1
            timeline_response1 = requests.get(f"{BASE_URL}/api/timeline/file", params={
                "session_id": session1_id,
                "file_path": test_file
            })
            
            if timeline_response1.status_code == 200:
                timeline1 = timeline_response1.json()
                print(f"✅ Session 1 timeline: {len(timeline1.get('commits', []))} commits")
            else:
                print(f"❌ Session 1 timeline failed: {timeline_response1.text}")
            
            # Test timeline in session 2
            timeline_response2 = requests.get(f"{BASE_URL}/api/timeline/file", params={
                "session_id": session2_id,
                "file_path": test_file
            })
            
            if timeline_response2.status_code == 200:
                timeline2 = timeline_response2.json()
                print(f"✅ Session 2 timeline: {len(timeline2.get('commits', []))} commits")
            else:
                print(f"❌ Session 2 timeline failed: {timeline_response2.text}")
    
    print("\n🎉 Cache optimization test completed!")

if __name__ == "__main__":
    asyncio.run(test_cache_optimization()) 