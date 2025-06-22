"""Security Analysis Task - Multi-layer vulnerability scanning"""

from typing import Dict, Any, Optional
from crewai import Task
from crewai.agent import Agent


class SecurityAnalysisTask:
    """Task for comprehensive security vulnerability analysis"""
    
    @staticmethod
    def create(
        agent: Agent,
        code: str,
        file_path: Optional[str] = None,
        language: str = "python",
        custom_rules: Optional[list] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a security analysis task"""
        
        # Check if we have pre-computed tool results to avoid redundant execution
        has_precomputed = context and (
            'precomputed_semgrep' in context or 
            'precomputed_security_patterns' in context
        )
        
        if has_precomputed:
            # Use optimized description that works with pre-computed results
            description = f"""
            Analyze the provided code for security vulnerabilities using pre-computed tool results.
            
            Code to analyze:
            ```{language}
            {code}
            ```
            
            File path: {file_path or "Not provided"}
            
            **IMPORTANT: Tool results have been pre-computed to optimize performance.**
            
            Pre-computed Semgrep results: {context.get('precomputed_semgrep', 'Not available')}
            Pre-computed security pattern results: {context.get('precomputed_security_patterns', 'Not available')}
            
            Your analysis must:
            
            1. **Review Pre-computed Semgrep Results**: 
               - Analyze the provided Semgrep scan results for OWASP Top 10 vulnerabilities
               - Interpret CWE Top 25 dangerous software error findings
               - Review security audit findings and secret detection results
               {f"- Consider custom rule results: {custom_rules}" if custom_rules else ""}
            
            2. **Review Pre-computed Pattern Analysis**:
               - Examine SecurityPatternAnalyzer results for:
                 • SQL injection vulnerabilities
                 • Command injection risks  
                 • Path traversal attacks
                 • Hardcoded secrets and credentials
                 • Other security anti-patterns
            
            3. **Contextual Risk Assessment**: For each finding:
               - Assign severity (Critical, High, Medium, Low, Info)
               - Explain potential impact and business risk
               - Provide exploitation scenario if applicable
               - Suggest specific remediation with code examples
               - Consider the code context and data flow
            
            4. **Supply Chain Considerations**:
               - Review any import statements for suspicious dependencies
               - Note outdated or vulnerable libraries if detected
            
            **Do NOT call security tools directly - use the pre-computed results provided.**
            
            Additional context: {context.get('additional_info', 'None provided')}
            """
            
            tools_to_use = []  # No tools needed, using pre-computed results
            
        else:
            # Fallback to original description if no pre-computed results
            description = f"""
            Perform comprehensive security analysis on the provided code.
            
            Code to analyze:
            ```{language}
            {code}
            ```
            
            File path: {file_path or "Not provided"}
            
            Your analysis must include:
            
            1. **Semgrep Scanning**: Run Semgrep with multiple rule sets:
               - OWASP Top 10 vulnerabilities
               - CWE Top 25 dangerous software errors
               - Security audit rules
               - Secret detection
               {f"- Custom rules: {custom_rules}" if custom_rules else ""}
            
            2. **Pattern Analysis**: Use SecurityPatternAnalyzer to detect:
               - SQL injection vulnerabilities
               - Command injection risks
               - Path traversal attacks
               - Weak cryptography usage
               - Hardcoded secrets and credentials
               - Insecure deserialization
               - XXE and SSRF vulnerabilities
            
            3. **Contextual Analysis**: Consider the code's context:
               - Is this user-facing code with untrusted input?
               - Does it handle sensitive data?
               - Are there authentication/authorization concerns?
               - What are the data flow paths?
            
            4. **Risk Assessment**: For each finding:
               - Assign severity (Critical, High, Medium, Low, Info)
               - Explain the potential impact
               - Provide exploitation scenario
               - Suggest remediation with code examples
            
            5. **Supply Chain Security**: Check for:
               - Vulnerable dependencies
               - Outdated libraries
               - Suspicious imports
            
            Additional context: {context or "None provided"}
            
            Provide a detailed security report with all findings, prioritized by risk.
            """
            
            tools_to_use = ["semgrep_scanner", "security_pattern_analyzer"]
        
        expected_output = """
        A comprehensive security analysis report containing:
        
        1. Executive Summary:
           - Total vulnerabilities by severity
           - Most critical findings
           - Overall security posture assessment (Critical/High/Medium/Low/Good)
        
        2. Detailed Findings:
           For each vulnerability:
           - Finding ID and type
           - Severity level with justification
           - Affected code location (line numbers)
           - Technical description
           - Potential impact and business risk
           - Exploitation difficulty and scenario
           - Remediation steps with specific code examples
           - References (CWE, OWASP, CVE if applicable)
        
        3. Risk Matrix:
           - Findings organized by severity and exploitability
           - Business impact assessment
           - Cumulative risk score
        
        4. Recommendations:
           - Immediate actions required (Critical/High severity)
           - Short-term security improvements
           - Long-term security enhancements
           - Security testing and validation strategies
           - Prevention measures for similar issues
        
        Focus on actionable, specific guidance that development teams can implement immediately.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            tools_to_use=tools_to_use
        )