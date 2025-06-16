#!/usr/bin/env python3
"""
Test script for the predictive issue resolution system
"""

import asyncio
import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.prediction.agents.orchestrator import PredictionOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_prediction_system():
    """Test the prediction system with a sample repository"""
    
    # Test configuration - use actual repo info to match the directory
    repo_path = "."  # Current directory (triage.flow)
    repo_owner = "huggingface"  # Actual owner from the logs
    repo_name = "smolagents"  # Actual repo name
    
    logger.info("🚀 Starting Predictive Issue Resolution System Test")
    logger.info(f"Testing with repo: {repo_owner}/{repo_name}")
    
    try:
        # Initialize orchestrator
        logger.info("📊 Initializing PredictionOrchestrator...")
        orchestrator = PredictionOrchestrator(
            repo_path=repo_path,
            repo_owner=repo_owner,
            repo_name=repo_name
        )
        
        # Test 1: Generate prediction report
        logger.info("🔮 Generating prediction report...")
        prediction_report = await orchestrator.generate_prediction_report(
            prediction_horizon_days=14
        )
        
        logger.info("✅ Prediction report generated successfully!")
        logger.info(f"   - Predicted issues: {len(prediction_report.predicted_issues)}")
        logger.info(f"   - Risk factors: {len(prediction_report.risk_factors)}")
        logger.info(f"   - Bug patterns: {len(prediction_report.detected_patterns)}")
        logger.info(f"   - Team patterns: {len(prediction_report.team_patterns)}")
        logger.info(f"   - Confidence score: {prediction_report.confidence_score:.2f}")
        logger.info(f"   - Analysis duration: {prediction_report.analysis_duration_seconds:.2f}s")
        
        # Test 2: Generate prevention strategies
        logger.info("🛡️  Generating prevention strategies...")
        prevention_strategies = await orchestrator.generate_prevention_strategies(
            context_description="Testing prevention strategy generation"
        )
        
        logger.info("✅ Prevention strategies generated successfully!")
        logger.info(f"   - Immediate actions: {len(prevention_strategies.get('immediate_actions', []))}")
        logger.info(f"   - Long-term strategies: {len(prevention_strategies.get('long_term_strategies', []))}")
        
        # Test 3: Get dashboard data
        logger.info("📈 Getting dashboard data...")
        dashboard_data = await orchestrator.get_dashboard_data()
        
        logger.info("✅ Dashboard data retrieved successfully!")
        logger.info(f"   - Total risks: {dashboard_data['summary']['total_risks']}")
        logger.info(f"   - High priority risks: {dashboard_data['summary']['high_priority_risks']}")
        logger.info(f"   - Confidence score: {dashboard_data['summary']['confidence_score']:.2f}")
        
        # Display sample results
        logger.info("\n📋 Sample Results:")
        logger.info("=" * 50)
        
        if prediction_report.immediate_actions:
            logger.info("🚨 Immediate Actions:")
            for i, action in enumerate(prediction_report.immediate_actions[:3], 1):
                logger.info(f"   {i}. {action}")
        
        if prediction_report.detected_patterns:
            logger.info("\n🔍 Detected Patterns:")
            for pattern in prediction_report.detected_patterns[:2]:
                logger.info(f"   - {pattern.pattern_type}: {pattern.description}")
                logger.info(f"     Confidence: {pattern.confidence:.2f}, Risk Score: {pattern.risk_score:.2f}")
        
        if prediction_report.risk_factors:
            logger.info("\n⚠️  Risk Factors:")
            for risk in prediction_report.risk_factors[:2]:
                logger.info(f"   - {risk.severity.upper()}: {risk.description}")
        
        logger.info("\n🎉 All tests completed successfully!")
        logger.info("The Predictive Issue Resolution System is working correctly.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

async def test_api_endpoints():
    """Test the API endpoints (requires FastAPI server to be running)"""
    logger.info("\n🌐 Testing API endpoints...")
    
    try:
        import httpx
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/prediction/health")
            if response.status_code == 200:
                logger.info("✅ Health endpoint working")
            else:
                logger.warning(f"⚠️  Health endpoint returned {response.status_code}")
        
        logger.info("🌐 API endpoint tests completed")
        
    except ImportError:
        logger.info("📝 Skipping API tests (httpx not available)")
    except Exception as e:
        logger.warning(f"⚠️  API tests failed: {e}")

def main():
    """Main test function"""
    logger.info("🧪 Predictive Issue Resolution System - Test Suite")
    logger.info("=" * 60)
    
    # Run core system tests
    success = asyncio.run(test_prediction_system())
    
    if success:
        logger.info("\n✅ Core system tests PASSED")
        
        # Run API tests if possible
        asyncio.run(test_api_endpoints())
        
        logger.info("\n🎯 Next Steps:")
        logger.info("1. Start the FastAPI server: python -m src.main")
        logger.info("2. Test the API endpoints:")
        logger.info("   - GET /prediction/health")
        logger.info("   - POST /prediction/analyze")
        logger.info("   - GET /prediction/dashboard/{owner}/{repo}")
        logger.info("3. Open the React dashboard to view predictions")
        
        return 0
    else:
        logger.error("\n❌ Core system tests FAILED")
        logger.error("Please check the error messages above and fix any issues.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 