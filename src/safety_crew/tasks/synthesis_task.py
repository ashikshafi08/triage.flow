"""Synthesis Task - Strategic safety analysis orchestration"""

from typing import Dict, Any, List, Optional
from crewai import Task
from crewai.agent import Agent


class SynthesisTask:
    """Task for strategic synthesis of all safety analysis results"""
    
    @staticmethod
    def create(
        agent: Agent,
        context_tasks: List[Task],
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a synthesis task that orchestrates all safety analysis results"""
        
        # Extract pre-computed tool results if available
        tool_results = additional_context.get('all_tool_results', {}) if additional_context else {}
        has_tool_results = bool(tool_results)
        
        # Get request details
        request_data = additional_context.get('request', {}) if additional_context else {}
        analysis_depth = additional_context.get('analysis_depth', 'standard') if additional_context else 'standard'
        
        description = f"""
        Synthesize findings from Security, Grounding, and Quality analysis teams into a unified safety assessment.
        
        **File**: {request_data.get('file_path', 'test_vulnerable.py')}
        **Analysis Depth**: {analysis_depth}
        
        **Your Task**: Review the completed analysis from:
        1. Security Analysis Agent - vulnerability findings
        2. Grounding Verification Agent - hallucination detection  
        3. Quality Architecture Agent - code quality assessment
        
        **Required Output**:
        1. **Executive Summary** - Overall risk level and key findings count
        2. **Prioritized Action Plan** - Immediate, short-term, and long-term actions
        3. **Safety Metrics** - Quantified scores for Security, Grounding, Quality
        4. **Business Recommendations** - Resource allocation and process improvements
        
        **Focus**: {"Quick risk assessment for critical/high findings only" if analysis_depth == 'quick' else "Balanced assessment with practical recommendations"}
        
        Be concise and actionable. Avoid redundancy with the specialist reports.
        """
        
        expected_output = """
        Concise safety synthesis report with:
        
        1. **Executive Summary**:
           - Overall risk level (Critical/High/Medium/Low)
           - Finding counts by domain (Security: X, Grounding: Y, Quality: Z)
           - Top 3 immediate actions required
        
        2. **Safety Scores**:
           - Security Score: X/10
           - Grounding Score: Y/10  
           - Quality Score: Z/10
           - Overall Risk Score: W/10
        
        3. **Prioritized Actions**:
           - **Immediate (0-7 days)**: Critical fixes
           - **Short-term (1-4 weeks)**: High-priority improvements
           - **Long-term (1-3 months)**: Strategic enhancements
        
        4. **Resource Recommendations**:
           - Budget/effort estimates for immediate actions
           - Process improvements needed
           - Training requirements
        
        Keep it actionable and focused. Reference specific findings from specialist reports.
        """
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent,
            context=context_tasks,  # This ensures the task waits for other tasks to complete
            tools_to_use=[]  # Orchestrator doesn't use tools, just synthesizes results
        )