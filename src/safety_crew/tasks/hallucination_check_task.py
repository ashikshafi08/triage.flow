"""Hallucination Check Task - AI-generated code verification"""

from typing import Dict, Any, Optional
from crewai import Task
from crewai.agent import Agent


class HallucinationCheckTask:
    """Task for detecting AI hallucinations in generated code"""
    
    @staticmethod
    def create(
        agent: Agent,
        code: str,
        session_id: str,
        file_path: Optional[str] = None,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a hallucination detection task"""
        
        # Check if we have pre-computed hallucination results
        has_precomputed = context and 'precomputed_hallucination' in context
        
        if has_precomputed:
            # Use optimized description that works with pre-computed results
            description = f"""
            Verify the factual accuracy of the provided code using pre-computed hallucination detection results.
            
            Code to analyze:
            ```{language}
            {code}
            ```
            
            Session ID: {session_id}
            File path: {file_path or "Not provided"}
            
            **IMPORTANT: Hallucination detection has been pre-computed to optimize performance.**
            
            Pre-computed hallucination results: {context.get('precomputed_hallucination', 'Not available')}
            
            Your analysis must:
            
            1. **Review Pre-computed Hallucination Detection**:
               - Examine the provided hallucination detection results
               - Identify any suspicious imports of non-existent libraries
               - Review fictional function calls that may not exist
               - Analyze suspicious API usage patterns
               - Check for impossible code constructs
            
            2. **Grounding Verification**:
               - Validate imports against known Python libraries and packages
               - Verify function calls exist in their respective modules
               - Check API signatures match documented interfaces
               - Confirm class and method names are factually correct
            
            3. **Context-Based Analysis**:
               - Cross-reference against codebase patterns if RAG system available
               - Verify consistency with project conventions
               - Check for anachronistic APIs or deprecated patterns
               - Validate against known good examples
            
            4. **Confidence Assessment**:
               - Rate confidence level for each potential hallucination
               - Provide evidence for hallucination claims
               - Suggest verification methods for uncertain cases
               - Identify high-risk hallucinations vs. style issues
            
            **Do NOT call hallucination detection tools directly - use the pre-computed results provided.**
            
            Additional context: {context.get('additional_info', 'None provided')}
            """
            
            tools_to_use = []  # No tools needed, using pre-computed results
            
        else:
            # Fallback to original description if no pre-computed results
            description = f"""
            Verify the factual accuracy and grounding of the provided code to detect potential AI hallucinations.
            
            Code to analyze:
            ```{language}
            {code}
            ```
            
            Session ID: {session_id}
            File path: {file_path or "Not provided"}
            
            Your verification must include:
            
            1. **Import Verification**:
               - Check all import statements against known libraries
               - Verify module and package names exist
               - Flag suspicious or non-existent imports
               - Cross-reference with PyPI, standard library, or known dependencies
            
            2. **API and Function Call Verification**:
               - Verify all function calls exist in their respective modules
               - Check method signatures match documented APIs
               - Validate parameter names and types
               - Flag fictional or non-existent methods
            
            3. **Pattern and Convention Analysis**:
               - Check for impossible code constructs
               - Verify syntax follows language conventions
               - Look for anachronistic or deprecated patterns
               - Identify suspicious naming patterns
            
            4. **Codebase Grounding** (if RAG available):
               - Cross-reference against existing codebase patterns
               - Verify consistency with project conventions
               - Check for similar patterns in the repository
               - Validate against known good examples
            
            5. **Confidence Scoring**:
               - Provide confidence scores for each finding
               - Explain reasoning behind hallucination detection
               - Suggest verification steps for ambiguous cases
            
            Additional context: {context or "None provided"}
            
            Focus on identifying clear hallucinations that would cause runtime errors or unexpected behavior.
            """
            
            tools_to_use = ["hallucination_detector"]
        
        expected_output = """
        A comprehensive hallucination detection report containing:
        
        1. Executive Summary:
           - Overall grounding score (0-100%)
           - Total potential hallucinations found
           - Confidence level of analysis
           - Risk assessment (High/Medium/Low)
        
        2. Detailed Hallucination Findings:
           For each potential hallucination:
           - Hallucination ID and type (import, function call, API, syntax)
           - Confidence score (0-100%)
           - Affected code location (line numbers)
           - Specific element that appears to be hallucinated
           - Evidence supporting hallucination claim
           - Potential impact (runtime error, incorrect behavior, etc.)
           - Suggested verification or correction method
        
        3. Import Analysis:
           - List of all imports with verification status
           - Flagged imports with reasons for suspicion
           - Recommendations for import verification
        
        4. API Verification Results:
           - Function/method calls with verification status
           - Signature mismatches or non-existent APIs
           - Recommendations for API validation
        
        5. Grounding Recommendations:
           - Steps to verify uncertain findings
           - Resources for fact-checking code elements
           - Prevention strategies for future AI-generated code
           - Integration with RAG systems for better grounding
        
        Prioritize findings by impact and confidence level. Focus on hallucinations that would cause immediate failures.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            tools_to_use=tools_to_use
        )