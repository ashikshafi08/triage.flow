"""Quality Review Task - Code quality and architecture assessment"""

from typing import Dict, Any, Optional
from crewai import Task
from crewai.agent import Agent


class QualityReviewTask:
    """Task for comprehensive code quality assessment"""
    
    @staticmethod
    def create(
        agent: Agent,
        code: str,
        file_path: Optional[str] = None,
        language: str = "python",
        context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a quality review task"""
        
        # Check if we have pre-computed quality results
        has_precomputed = context and 'precomputed_quality' in context
        
        if has_precomputed:
            # Use optimized description that works with pre-computed results
            description = f"""
            Perform comprehensive code quality assessment using pre-computed analysis results.
            
            Code to review:
            ```{language}
            {code}
            ```
            
            File path: {file_path or "Not provided"}
            
            **IMPORTANT: Quality analysis has been pre-computed to optimize performance.**
            
            Pre-computed quality results: {context.get('precomputed_quality', 'Not available')}
            
            Your analysis must:
            
            1. **Review Pre-computed Quality Metrics**:
               - Examine the provided code quality analysis results
               - Review cyclomatic complexity measurements
               - Analyze function and class size metrics
               - Assess maintainability index scores
               - Review code smells and anti-patterns
            
            2. **Code Structure Assessment**:
               - Evaluate architectural patterns from the analysis
               - Review separation of concerns
               - Assess cohesion and coupling metrics
               - Analyze code organization and modularity
            
            3. **Maintainability Analysis**:
               - Review readability scores and recommendations
               - Assess documentation coverage and quality
               - Evaluate naming conventions and clarity
               - Check for potential technical debt indicators
            
            4. **Performance Considerations**:
               - Review algorithmic complexity assessments
               - Identify potential performance bottlenecks
               - Assess resource usage patterns
               - Consider scalability implications
            
            5. **Best Practices Compliance**:
               - Evaluate adherence to language-specific best practices
               - Review design pattern usage appropriateness
               - Assess error handling and robustness
               - Check testing considerations and testability
            
            **Do NOT call quality analysis tools directly - use the pre-computed results provided.**
            
            Additional context: {context.get('additional_info', 'None provided')}
            """
            
            tools_to_use = []  # No tools needed, using pre-computed results
            
        else:
            # Fallback to original description if no pre-computed results
            description = f"""
            Perform comprehensive code quality assessment and architectural review.
            
            Code to review:
            ```{language}
            {code}
            ```
            
            File path: {file_path or "Not provided"}
            
            Your quality review must include:
            
            1. **Code Quality Analysis**:
               - Run CodeQualityAnalyzer to get comprehensive metrics
               - Assess cyclomatic complexity
               - Evaluate function and class sizes
               - Check maintainability index
               - Identify code smells and anti-patterns
            
            2. **Architectural Assessment**:
               - Evaluate overall code structure and organization
               - Check separation of concerns
               - Assess coupling and cohesion
               - Review design patterns usage
               - Identify architectural issues
            
            3. **Maintainability Review**:
               - Assess code readability and clarity
               - Check naming conventions
               - Evaluate documentation quality
               - Identify technical debt
               - Review refactoring opportunities
            
            4. **Performance Analysis**:
               - Identify potential performance issues
               - Check algorithmic complexity
               - Assess resource usage
               - Review optimization opportunities
            
            5. **Best Practices Check**:
               - Verify adherence to language conventions
               - Check error handling patterns
               - Assess testing considerations
               - Review security implications from quality perspective
            
            Additional context: {context or "None provided"}
            
            Provide specific, actionable recommendations for improvement.
            """
            
            tools_to_use = ["code_quality_analyzer"]
        
        expected_output = """
        A comprehensive code quality assessment report containing:
        
        1. Executive Summary:
           - Overall quality score (0-100)
           - Key quality metrics summary
           - Most critical issues identified
           - Maintainability assessment (Excellent/Good/Fair/Poor)
        
        2. Quality Metrics Analysis:
           - Cyclomatic complexity breakdown by function/method
           - Function and class size analysis
           - Maintainability index calculations
           - Code duplication assessment
           - Technical debt indicators
        
        3. Architectural Assessment:
           - Code structure and organization evaluation
           - Design pattern usage analysis
           - Coupling and cohesion metrics
           - Separation of concerns assessment
           - Modularity and reusability evaluation
        
        4. Code Quality Issues:
           For each identified issue:
           - Issue type and category
           - Severity level (Critical/High/Medium/Low)
           - Affected code location (line numbers)
           - Detailed description and impact
           - Root cause analysis
           - Specific remediation steps with code examples
           - Priority ranking for fixes
        
        5. Maintainability Report:
           - Readability assessment
           - Documentation coverage and quality
           - Naming convention adherence
           - Code clarity and understanding metrics
           - Refactoring recommendations
        
        6. Performance Considerations:
           - Algorithmic complexity analysis
           - Potential performance bottlenecks
           - Resource usage patterns
           - Scalability implications
           - Optimization recommendations
        
        7. Best Practices Compliance:
           - Language-specific convention adherence
           - Error handling pattern assessment
           - Testing and testability considerations
           - Security implications from quality perspective
        
        8. Improvement Roadmap:
           - Immediate fixes required (Critical/High priority)
           - Short-term quality improvements
           - Long-term architectural enhancements
           - Preventive measures for quality maintenance
           - Code review and quality gate recommendations
        
        Focus on actionable, prioritized recommendations that development teams can implement systematically.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            tools_to_use=tools_to_use
        )