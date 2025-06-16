"""
Semantic code analyzer that uses RAG-indexed codebase for line-by-line risk analysis
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from ...llm_client import LLMClient
from ...agent_tools.core import AgenticCodebaseExplorer
from ..data_models.prediction_models import RiskFactor

logger = logging.getLogger(__name__)

class SemanticCodeAnalyzer:
    """
    Performs semantic analysis of code using RAG-indexed codebase
    Focuses on actual code quality risks rather than repository metrics
    """
    
    def __init__(self, repo_path: str, repo_owner: str, repo_name: str):
        self.repo_path = repo_path
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        
        # Use existing RAG system for semantic analysis
        self.llm_client = LLMClient()
        self.agentic_explorer = AgenticCodebaseExplorer(
            session_id=f"semantic_analysis_{repo_owner}_{repo_name}",
            repo_path=repo_path,
            issue_rag_system=None
        )
        
        logger.info(f"Initialized SemanticCodeAnalyzer for {repo_owner}/{repo_name}")
    
    async def analyze_code_quality_risks(self, file_paths: Optional[List[str]] = None, progress_callback=None) -> List[RiskFactor]:
        """
        Analyze code for quality-based risks using semantic understanding
        
        Args:
            file_paths: Specific files to analyze, or None for repository-wide analysis
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of actual code quality risk factors
        """
        logger.info("Starting semantic code quality analysis")
        
        try:
            if progress_callback:
                await progress_callback({
                    "type": "progress",
                    "phase": "semantic_discovery",
                    "message": "Discovering files for semantic analysis...",
                    "progress": 5
                })
            
            # Phase 1: Identify files to analyze
            target_files = await self._identify_analysis_targets(file_paths)
            
            if progress_callback:
                await progress_callback({
                    "type": "progress",
                    "phase": "semantic_analysis",
                    "message": f"Analyzing {len(target_files)} files with AI...",
                    "progress": 15
                })
            
            # Phase 2: Perform semantic analysis on each file
            risk_factors = []
            total_files = min(len(target_files), 10)  # Limit to 10 files for MVP
            
            for i, file_path in enumerate(target_files[:10]):
                if progress_callback:
                    progress = 15 + (i / total_files) * 70  # 15% to 85%
                    await progress_callback({
                        "type": "progress",
                        "phase": "file_analysis",
                        "message": f"Analyzing {file_path}...",
                        "progress": int(progress),
                        "current_file": file_path
                    })
                
                file_risks = await self._analyze_file_semantics(file_path)
                risk_factors.extend(file_risks)
            
            if progress_callback:
                await progress_callback({
                    "type": "progress",
                    "phase": "repository_patterns",
                    "message": "Analyzing repository-wide patterns...",
                    "progress": 90
                })
            
            # Phase 3: Repository-wide pattern analysis
            repo_risks = await self._analyze_repository_patterns()
            risk_factors.extend(repo_risks)
            
            logger.info(f"Identified {len(risk_factors)} semantic risk factors")
            return risk_factors
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return []
    
    async def _identify_analysis_targets(self, file_paths: Optional[List[str]]) -> List[str]:
        """Identify which files to analyze for risks"""
        
        if file_paths:
            return file_paths
        
        # Use RAG to find files that might have quality issues
        analysis_query = """
        Find files in this codebase that might have code quality issues. Look for:
        1. Complex functions with high cyclomatic complexity
        2. Files with many nested conditions or loops
        3. Large functions or classes
        4. Files with potential security vulnerabilities
        5. Code with poor error handling
        6. Files with outdated patterns or anti-patterns
        
        Return the file paths that should be prioritized for quality analysis.
        """
        
        try:
            response = await self.agentic_explorer.query(analysis_query)
            
            # Extract file paths from response (improved parsing)
            file_paths = []
            lines = response.split('\n')
            for line in lines:
                # Look for file paths in the response
                if any(ext in line for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']):
                    # Extract potential file path
                    words = line.split()
                    for word in words:
                        if any(ext in word for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']):
                            clean_path = word.strip('`"\'()[]{}.,;:')
                            # Only add if it looks like a real file path (not just an extension)
                            if len(clean_path) > 4 and '/' in clean_path:
                                file_paths.append(clean_path)
            
            # If we found actual file paths, use them
            if file_paths:
                return file_paths[:20]
            
            # Enhanced fallback: try to find actual files in common directories
            import os
            fallback_files = []
            common_dirs = ['src', 'lib', 'app', 'components', 'utils', 'services', 'models']
            
            for dir_name in common_dirs:
                dir_path = os.path.join(self.repo_path, dir_name)
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    for root, dirs, files in os.walk(dir_path):
                        for file in files:
                            if any(file.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']):
                                rel_path = os.path.relpath(os.path.join(root, file), self.repo_path)
                                fallback_files.append(rel_path)
                                if len(fallback_files) >= 10:  # Limit fallback files
                                    break
                        if len(fallback_files) >= 10:
                            break
                if len(fallback_files) >= 10:
                    break
            
            return fallback_files if fallback_files else ['README.md']  # Ultimate fallback
            
        except Exception as e:
            logger.error(f"Error identifying analysis targets: {e}")
            # Try to find any files in the repo as ultimate fallback
            try:
                import os
                fallback_files = []
                if os.path.exists(self.repo_path):
                    for root, dirs, files in os.walk(self.repo_path):
                        for file in files:
                            if any(file.endswith(ext) for ext in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs']):
                                rel_path = os.path.relpath(os.path.join(root, file), self.repo_path)
                                fallback_files.append(rel_path)
                                if len(fallback_files) >= 5:
                                    return fallback_files
                return fallback_files if fallback_files else ['README.md']
            except Exception:
                return ['README.md']  # Final fallback
    
    async def _analyze_file_semantics(self, file_path: str) -> List[RiskFactor]:
        """Perform semantic analysis on a specific file"""
        
        analysis_prompt = f"""
        Analyze the file '{file_path}' for code quality risks. Focus on:
        
        1. **Code Complexity**: Functions with high cyclomatic complexity, deep nesting
        2. **Error Handling**: Missing try-catch blocks, unhandled edge cases
        3. **Security Issues**: SQL injection, XSS vulnerabilities, insecure patterns
        4. **Performance Issues**: Inefficient algorithms, memory leaks, blocking operations
        5. **Maintainability**: Code duplication, unclear naming, lack of documentation
        6. **Bug Patterns**: Common anti-patterns that lead to bugs
        
        For each risk found, provide:
        - Specific line numbers or function names
        - Risk severity (low/medium/high/critical)
        - Clear description of the issue
        - Suggested mitigation
        
        Only report actual code quality issues, not repository-level metrics.
        """
        
        try:
            response = await self.agentic_explorer.query(analysis_prompt)
            return self._parse_file_risks(file_path, response)
            
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return []
    
    def _parse_file_risks(self, file_path: str, analysis_response: str) -> List[RiskFactor]:
        """Parse LLM response to extract risk factors"""
        
        risks = []
        lines = analysis_response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for severity indicators
            severity = 'medium'  # default
            if any(word in line.lower() for word in ['critical', 'severe', 'high risk']):
                severity = 'critical'
            elif any(word in line.lower() for word in ['high', 'important', 'significant']):
                severity = 'high'
            elif any(word in line.lower() for word in ['low', 'minor', 'small']):
                severity = 'low'
            
            # Look for risk descriptions
            if any(keyword in line.lower() for keyword in [
                'complexity', 'error handling', 'security', 'performance', 
                'maintainability', 'bug pattern', 'vulnerability', 'issue'
            ]):
                if len(line) > 20:  # Avoid very short lines
                    
                    # Extract mitigation if present
                    mitigation = []
                    if 'suggest' in line.lower() or 'recommend' in line.lower():
                        mitigation.append(line)
                    
                    risks.append(RiskFactor(
                        factor_type=self._classify_risk_type(line),
                        severity=severity,
                        confidence=0.7,  # Default confidence for semantic analysis
                        description=line,
                        affected_files=[file_path],
                        mitigation_actions=mitigation or [f"Review and refactor {file_path}"]
                    ))
        
        return risks[:5]  # Limit to top 5 risks per file
    
    def _classify_risk_type(self, description: str) -> str:
        """Classify the type of risk based on description"""
        
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ['security', 'vulnerability', 'injection', 'xss']):
            return 'security_vulnerability'
        elif any(word in desc_lower for word in ['performance', 'slow', 'inefficient', 'memory']):
            return 'performance_issue'
        elif any(word in desc_lower for word in ['complexity', 'nested', 'cyclomatic']):
            return 'code_complexity'
        elif any(word in desc_lower for word in ['error', 'exception', 'handling']):
            return 'error_handling'
        elif any(word in desc_lower for word in ['maintainability', 'duplication', 'naming']):
            return 'maintainability_issue'
        else:
            return 'code_quality'
    
    async def _analyze_repository_patterns(self) -> List[RiskFactor]:
        """Analyze repository-wide patterns for systemic risks"""
        
        pattern_query = """
        Analyze this codebase for systemic code quality patterns that indicate risk:
        
        1. **Architecture Issues**: Tight coupling, circular dependencies, violation of SOLID principles
        2. **Testing Gaps**: Critical code paths without tests, low test coverage areas
        3. **Configuration Risks**: Hardcoded secrets, missing environment validation
        4. **Dependency Issues**: Outdated packages with known vulnerabilities
        5. **Documentation Gaps**: Complex code without documentation
        
        Focus on patterns that could lead to production issues, not just style preferences.
        Provide specific examples and affected areas.
        """
        
        try:
            response = await self.agentic_explorer.query(pattern_query)
            return self._parse_repository_risks(response)
            
        except Exception as e:
            logger.error(f"Error analyzing repository patterns: {e}")
            return []
    
    def _parse_repository_risks(self, analysis_response: str) -> List[RiskFactor]:
        """Parse repository-wide risk analysis"""
        
        risks = []
        lines = analysis_response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if any(keyword in line.lower() for keyword in [
                'architecture', 'testing', 'configuration', 'dependency', 
                'documentation', 'coupling', 'vulnerability'
            ]) and len(line) > 30:
                
                # Determine severity based on keywords
                severity = 'medium'
                if any(word in line.lower() for word in ['critical', 'severe', 'security']):
                    severity = 'critical'
                elif any(word in line.lower() for word in ['high', 'important', 'production']):
                    severity = 'high'
                
                risks.append(RiskFactor(
                    factor_type=self._classify_risk_type(line),
                    severity=severity,
                    confidence=0.8,  # Higher confidence for repository-wide patterns
                    description=line,
                    affected_files=['repository-wide'],
                    mitigation_actions=[
                        "Conduct architecture review",
                        "Implement systematic improvements",
                        "Add monitoring and alerts"
                    ]
                ))
        
        return risks[:3]  # Limit to top 3 repository risks
    
    async def get_risk_summary(self) -> Dict[str, Any]:
        """Get a summary of semantic risk analysis"""
        
        risks = await self.analyze_code_quality_risks()
        
        # Categorize risks
        risk_categories = {}
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for risk in risks:
            # Count by category
            category = risk.factor_type
            if category not in risk_categories:
                risk_categories[category] = 0
            risk_categories[category] += 1
            
            # Count by severity
            severity_counts[risk.severity] += 1
        
        return {
            'total_risks': len(risks),
            'risk_categories': risk_categories,
            'severity_distribution': severity_counts,
            'top_risks': [
                {
                    'type': risk.factor_type,
                    'severity': risk.severity,
                    'description': risk.description[:100] + '...' if len(risk.description) > 100 else risk.description,
                    'files': risk.affected_files
                }
                for risk in sorted(risks, key=lambda r: {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[r.severity], reverse=True)[:5]
            ],
            'analysis_timestamp': datetime.now().isoformat()
        } 