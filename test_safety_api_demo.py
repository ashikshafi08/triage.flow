#!/usr/bin/env python3
"""
HTTP API Demo for Safety Crew - Test the REST endpoints
"""

import requests
import json
import time
from pathlib import Path

def test_safety_api():
    """Test the safety crew API endpoints"""
    
    base_url = "http://localhost:8000"
    session_id = "safety_api_test"
    
    print("🌐 Testing Safety Crew HTTP API\n")
    
    # Test file - let's analyze a simple Python file
    test_code = '''
import os
import subprocess
import sqlite3

def vulnerable_function(user_input):
    # SQL Injection vulnerability
    query = f"SELECT * FROM users WHERE name = '{user_input}'"
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    
    # Command injection vulnerability
    os.system(f"echo {user_input}")
    
    # Potential path traversal
    filename = f"/tmp/{user_input}.txt"
    with open(filename, 'w') as f:
        f.write("data")
    
    return result

def quality_issues():
    # High complexity function
    for i in range(10):
        for j in range(10):
            for k in range(10):
                if i == j == k:
                    print("complex logic")
    
    # Poor variable naming
    x = 1
    y = 2
    z = x + y
    return z

# Potential AI hallucination - fictional library
import nonexistent_lib
result = nonexistent_lib.magic_function()
'''
    
    # Prepare the request payload
    payload = {
        "session_id": session_id,
        "code": test_code,
        "file_path": "test_vulnerable.py",
        "language": "python",
        "analysis_depth": "standard",
        "include_auto_fix": True
    }
    
    try:
        # First, create a session
        print(f"🔧 Creating session for safety analysis...")
        session_payload = {
            "repo_url": "https://github.com/huggingface/smolagents",  # Use a known repo
            "session_name": "Safety Analysis Test"
        }
        
        session_response = requests.post(
            f"{base_url}/assistant/sessions",
            json=session_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if session_response.status_code != 200:
            print(f"❌ Failed to create session: {session_response.status_code}")
            print(f"Response: {session_response.text}")
            return
            
        session_data = session_response.json()
        actual_session_id = session_data.get("session_id")
        print(f"✅ Session created: {actual_session_id}")
        
        # Now send the safety analysis request
        print(f"🚀 Sending safety analysis request to {base_url}/api/safety/{actual_session_id}/analyze")
        print(f"📄 Code length: {len(test_code)} characters")
        print(f"🔍 Analysis depth: {payload['analysis_depth']}")
        print()
        
        # Update payload with actual session ID
        payload["session_id"] = actual_session_id
        
        # Send POST request
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/safety/{actual_session_id}/analyze",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300  # 5 minute timeout for safety analysis (increased for comprehensive analysis)
        )
        duration = time.time() - start_time
        
        print(f"⏱️ Request completed in {duration:.2f} seconds")
        print(f"📡 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Safety analysis successful!\n")
            
            # Display results
            print("📊 ANALYSIS RESULTS:")
            print("=" * 50)
            
            # Safety metrics
            metrics = result.get("safety_metrics", {})
            print(f"🔒 Security Score: {metrics.get('security_score', 'N/A')}/10")
            print(f"🎯 Grounding Score: {metrics.get('grounding_score', 'N/A')}/10")
            print(f"⭐ Quality Score: {metrics.get('quality_score', 'N/A')}/10")
            print(f"🚨 Overall Risk Score: {metrics.get('overall_risk_score', 'N/A')}/10")
            print(f"📈 Total Findings: {metrics.get('total_findings', 0)}")
            print()
            
            # Security findings
            security_findings = result.get("security_findings", [])
            if security_findings:
                print(f"🛡️ SECURITY FINDINGS ({len(security_findings)}):")
                print("-" * 30)
                for i, finding in enumerate(security_findings, 1):
                    severity = finding.get('severity', 'unknown').upper()
                    title = finding.get('title', 'Unknown issue')
                    description = finding.get('description', 'No description')
                    remediation = finding.get('remediation', 'No remediation')
                    
                    print(f"{i}. [{severity}] {title}")
                    print(f"   Description: {description}")
                    print(f"   Remediation: {remediation}")
                    if finding.get('line_number'):
                        print(f"   Line: {finding['line_number']}")
                    print()
            
            # Hallucination flags
            hallucination_flags = result.get("hallucination_flags", [])
            if hallucination_flags:
                print(f"🧠 HALLUCINATION FLAGS ({len(hallucination_flags)}):")
                print("-" * 30)
                for i, flag in enumerate(hallucination_flags, 1):
                    description = flag.get('description', 'Unknown hallucination')
                    hallucinated_code = flag.get('hallucinated_code', 'N/A')
                    suggested_correction = flag.get('suggested_correction', 'N/A')
                    
                    print(f"{i}. {description}")
                    print(f"   Hallucinated: {hallucinated_code}")
                    print(f"   Suggested: {suggested_correction}")
                    print()
            
            # Quality issues
            quality_issues = result.get("quality_issues", [])
            if quality_issues:
                print(f"📈 QUALITY ISSUES ({len(quality_issues)}):")
                print("-" * 30)
                for i, issue in enumerate(quality_issues, 1):
                    title = issue.get('title', 'Unknown quality issue')
                    description = issue.get('description', 'No description')
                    improvement = issue.get('improvement_suggestion', 'No suggestion')
                    
                    print(f"{i}. {title}")
                    print(f"   Description: {description}")
                    print(f"   Improvement: {improvement}")
                    print()
            
            # Agent recommendations
            recommendations = result.get("agent_recommendations", [])
            if recommendations:
                print(f"💡 AGENT RECOMMENDATIONS ({len(recommendations)}):")
                print("-" * 30)
                for i, rec in enumerate(recommendations, 1):
                    agent = rec.get('agent_name', 'Unknown')
                    recommendation = rec.get('recommendation', 'No recommendation')
                    priority = rec.get('priority', 'medium')
                    
                    print(f"{i}. [{agent}] Priority: {priority.upper()}")
                    print(f"   {recommendation}")
                    print()
            
            # Auto-fix suggestions
            auto_fixes = result.get("auto_fix_suggestions", [])
            if auto_fixes:
                print(f"🔧 AUTO-FIX SUGGESTIONS ({len(auto_fixes)}):")
                print("-" * 30)
                for i, fix in enumerate(auto_fixes, 1):
                    fix_type = fix.get('fix_type', 'unknown')
                    explanation = fix.get('explanation', 'No explanation')
                    confidence = fix.get('confidence', 0)
                    
                    print(f"{i}. {fix_type.upper()} fix (confidence: {confidence:.1%})")
                    print(f"   {explanation}")
                    print()
            
            print("✅ Analysis complete!")
            
        else:
            print(f"❌ API request failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Response text: {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_safety_api() 