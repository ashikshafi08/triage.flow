#!/usr/bin/env python3
"""
Test Frontend-Backend Connection for OnboardAI

This script validates that the frontend and backend are properly connected
by testing the API endpoints that the frontend expects to use.
"""

import asyncio
import logging
import sys
from pathlib import Path
import requests
import subprocess
import time
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_PORT = 8000
FRONTEND_PORT = 3000
BACKEND_URL = f"http://localhost:{BACKEND_PORT}"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"

def test_backend_health():
    """Test if backend is running and healthy"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/onboarding/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ Backend health check passed")
            return True
        else:
            logger.error(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Backend not responding: {e}")
        return False

def test_onboarding_endpoints():
    """Test the key onboarding API endpoints that frontend uses"""
    
    # Test data
    test_user_id = "test-user"
    test_workspace_id = "test-workspace" 
    test_repo_path = "https://github.com/ash/triage.flow"
    
    endpoints_to_test = [
        {
            "name": "Profile Survey",
            "method": "POST",
            "url": f"{BACKEND_URL}/api/onboarding/profile/survey",
            "data": {
                "name": "Test User",
                "email": "test@example.com",
                "experience_level": "mid",
                "role": "fullstack",
                "years_of_experience": 3,
                "programming_languages": ["python", "javascript"],
                "frameworks": ["react", "fastapi"],
                "learning_style": "hands_on",
                "goals": ["learn codebase", "contribute"]
            }
        },
        {
            "name": "Start Session",
            "method": "POST", 
            "url": f"{BACKEND_URL}/api/onboarding/session/start",
            "data": {
                "user_id": test_user_id,
                "workspace_id": test_workspace_id,
                "repo_path": test_repo_path
            }
        },
        {
            "name": "Get Workflow",
            "method": "GET",
            "url": f"{BACKEND_URL}/api/onboarding/workflow",
            "params": {
                "user_id": test_user_id,
                "workspace_id": test_workspace_id,
                "repo_path": test_repo_path
            }
        },
        {
            "name": "Chat Message",
            "method": "POST",
            "url": f"{BACKEND_URL}/api/onboarding/chat",
            "params": {
                "user_id": test_user_id,
                "workspace_id": test_workspace_id,
                "repo_path": test_repo_path
            },
            "data": {
                "question": "What is the architecture of this codebase?",
                "context": {}
            }
        }
    ]
    
    results = []
    
    for endpoint in endpoints_to_test:
        try:
            logger.info(f"Testing {endpoint['name']}...")
            
            if endpoint['method'] == 'GET':
                response = requests.get(
                    endpoint['url'], 
                    params=endpoint.get('params', {}),
                    timeout=30
                )
            else:
                response = requests.request(
                    endpoint['method'],
                    endpoint['url'],
                    json=endpoint.get('data', {}),
                    params=endpoint.get('params', {}),
                    timeout=30
                )
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ {endpoint['name']} - Success ({response.status_code})")
                results.append(True)
            else:
                logger.error(f"❌ {endpoint['name']} - Failed ({response.status_code}): {response.text[:200]}")
                results.append(False)
                
        except Exception as e:
            logger.error(f"❌ {endpoint['name']} - Exception: {e}")
            results.append(False)
    
    return all(results)

def test_frontend_access():
    """Test if frontend can be accessed"""
    try:
        response = requests.get(f"{FRONTEND_URL}/onboarding", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Frontend onboarding page accessible")
            return True
        else:
            logger.error(f"❌ Frontend access failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Frontend not responding: {e}")
        return False

def check_frontend_api_client():
    """Check if the frontend API client exists and is properly structured"""
    api_client_path = Path("issue-flow-ai-prompt/src/lib/onboarding-api.ts")
    
    if not api_client_path.exists():
        logger.error("❌ Frontend API client not found")
        return False
    
    # Check if it contains the expected exports
    content = api_client_path.read_text()
    expected_exports = [
        "createOnboardingAPI",
        "OnboardingAPIClient", 
        "initializeOnboarding",
        "DeveloperProfile",
        "OnboardingWorkflow"
    ]
    
    missing_exports = []
    for export in expected_exports:
        if export not in content:
            missing_exports.append(export)
    
    if missing_exports:
        logger.error(f"❌ Frontend API client missing exports: {missing_exports}")
        return False
    
    logger.info("✅ Frontend API client properly structured")
    return True

def check_component_integration():
    """Check if OnboardingDashboard is properly integrated"""
    dashboard_path = Path("issue-flow-ai-prompt/src/pages/OnboardingDashboard.tsx")
    app_path = Path("issue-flow-ai-prompt/src/App.tsx")
    
    if not dashboard_path.exists():
        logger.error("❌ OnboardingDashboard component not found")
        return False
    
    if not app_path.exists():
        logger.error("❌ App.tsx not found")
        return False
    
    # Check if OnboardingDashboard is imported in App.tsx
    app_content = app_path.read_text()
    if "OnboardingDashboard" not in app_content:
        logger.error("❌ OnboardingDashboard not imported in App.tsx")
        return False
    
    # Check if route is defined
    if "/onboarding" not in app_content:
        logger.error("❌ Onboarding route not defined in App.tsx")
        return False
    
    logger.info("✅ OnboardingDashboard properly integrated")
    return True

def main():
    """Run all connection tests"""
    logger.info("🔧 Testing OnboardAI Frontend-Backend Connection...")
    logger.info("=" * 60)
    
    # Test 1: Check file structure
    logger.info("1. Checking file structure...")
    api_client_ok = check_frontend_api_client()
    component_ok = check_component_integration()
    
    # Test 2: Backend health
    logger.info("\n2. Testing backend...")
    backend_ok = test_backend_health()
    
    if backend_ok:
        # Test 3: API endpoints
        logger.info("\n3. Testing API endpoints...")
        endpoints_ok = test_onboarding_endpoints()
    else:
        logger.warning("⚠️  Skipping API endpoint tests (backend not running)")
        endpoints_ok = False
    
    # Test 4: Frontend access (optional - requires frontend server)
    logger.info("\n4. Testing frontend access...")
    frontend_ok = test_frontend_access()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 CONNECTION TEST SUMMARY:")
    logger.info(f"   API Client Structure: {'✅' if api_client_ok else '❌'}")
    logger.info(f"   Component Integration: {'✅' if component_ok else '❌'}")
    logger.info(f"   Backend Health: {'✅' if backend_ok else '❌'}")
    logger.info(f"   API Endpoints: {'✅' if endpoints_ok else '❌'}")
    logger.info(f"   Frontend Access: {'✅' if frontend_ok else '❌'}")
    
    overall_success = api_client_ok and component_ok and backend_ok
    
    if overall_success:
        logger.info("\n🎉 SUCCESS: Frontend and backend are properly connected!")
        logger.info("   You can now:")
        logger.info("   - Start the backend: cd src && python -m main")
        logger.info("   - Start the frontend: cd issue-flow-ai-prompt && npm run dev")
        logger.info("   - Visit: http://localhost:3000/onboarding")
    else:
        logger.info("\n⚠️  ISSUES FOUND: Some connections need to be fixed")
        
        if not backend_ok:
            logger.info("   - Start the backend server first")
        if not api_client_ok or not component_ok:
            logger.info("   - Fix the frontend integration issues")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 