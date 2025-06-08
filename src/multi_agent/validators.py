"""
Validation and Safety Mechanisms for Multi-Agent System

These components ensure code quality, safety checks, and human oversight
before any automated implementations are deployed.
"""

import ast
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import re
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class CodeValidator:
    """
    Validates generated code for syntax, imports, and basic quality checks.
    Prevents hallucinated or dangerous code from being executed.
    """
    
    def __init__(self):
        self.supported_languages = {
            "python": self._validate_python,
            "javascript": self._validate_javascript,
            "typescript": self._validate_typescript
        }
        
    def validate_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Validate code for syntax and basic quality"""
        
        if not code or not code.strip():
            return {
                "valid": False,
                "errors": ["Empty code provided"],
                "warnings": [],
                "suggestions": []
            }
        
        language = language.lower()
        validator = self.supported_languages.get(language, self._validate_generic)
        
        try:
            return validator(code)
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            return {
                "valid": False,
                "errors": [f"Validation error: {str(e)}"],
                "warnings": [],
                "suggestions": ["Manual review required"]
            }
    
    def _validate_python(self, code: str) -> Dict[str, Any]:
        """Validate Python code"""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Syntax validation
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
        except Exception as e:
            errors.append(f"Parse error: {e}")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            (r'exec\s*\(', "Use of exec() detected - potential security risk"),
            (r'eval\s*\(', "Use of eval() detected - potential security risk"),
            (r'__import__\s*\(', "Dynamic imports detected - review required"),
            (r'subprocess\.', "Subprocess usage detected - review required"),
            (r'os\.system', "os.system usage detected - potential security risk"),
            (r'open\s*\([\'"]\/.*[\'"]', "Absolute path file access detected"),
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                warnings.append(message)
        
        # Check for import validity (basic check)
        import_errors = self._check_python_imports(code)
        errors.extend(import_errors)
        
        # Code quality suggestions
        if len(code.split('\n')) > 100:
            suggestions.append("Consider breaking large functions into smaller ones")
        
        if not re.search(r'""".*?"""', code, re.DOTALL) and len(code) > 50:
            suggestions.append("Consider adding docstrings for better documentation")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def _check_python_imports(self, code: str) -> List[str]:
        """Check if Python imports are likely to be valid"""
        
        errors = []
        
        # Extract import statements
        import_pattern = r'^(?:from\s+(\S+)\s+)?import\s+(.+)$'
        
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith(('import ', 'from ')):
                match = re.match(import_pattern, line)
                if match:
                    module_name = match.group(1) or match.group(2).split('.')[0]
                    # Check for obviously invalid module names
                    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*$', module_name):
                        errors.append(f"Invalid module name: {module_name}")
        
        return errors
    
    def _validate_javascript(self, code: str) -> Dict[str, Any]:
        """Validate JavaScript code (basic validation)"""
        
        errors = []
        warnings = []
        suggestions = []
        
        # Basic syntax checks
        if code.count('{') != code.count('}'):
            errors.append("Mismatched curly braces")
        
        if code.count('(') != code.count(')'):
            errors.append("Mismatched parentheses")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            (r'eval\s*\(', "Use of eval() detected - potential security risk"),
            (r'innerHTML\s*=', "Direct innerHTML assignment - potential XSS risk"),
            (r'document\.write', "document.write usage - not recommended")
        ]
        
        for pattern, message in dangerous_patterns:
            if re.search(pattern, code):
                warnings.append(message)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions
        }
    
    def _validate_typescript(self, code: str) -> Dict[str, Any]:
        """Validate TypeScript code (extends JavaScript validation)"""
        
        # Start with JavaScript validation
        result = self._validate_javascript(code)
        
        # Add TypeScript-specific checks
        if ':any' in code:
            result["suggestions"].append("Avoid 'any' type - use specific types for better type safety")
        
        return result
    
    def _validate_generic(self, code: str) -> Dict[str, Any]:
        """Generic validation for unsupported languages"""
        
        return {
            "valid": True,
            "errors": [],
            "warnings": ["Language-specific validation not available"],
            "suggestions": ["Manual review recommended for this language"]
        }


class SafetyGate:
    """
    Safety gate that ensures human oversight for potentially risky operations.
    Implements approval workflows and risk assessment.
    """
    
    def __init__(self):
        self.risk_threshold = 7  # Risk scores above this require approval
        self.auto_approve_patterns = [
            "documentation",
            "comments",
            "readme",
            "test"
        ]
        
    def assess_implementation_risk(self, implementation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk level of an implementation plan"""
        
        risk_factors = {
            "code_complexity": self._assess_code_complexity_risk(implementation_plan),
            "file_operations": self._assess_file_operation_risk(implementation_plan),
            "external_dependencies": self._assess_dependency_risk(implementation_plan),
            "security_sensitive": self._assess_security_risk(implementation_plan),
            "data_operations": self._assess_data_risk(implementation_plan)
        }
        
        # Calculate overall risk score (1-10)
        risk_score = sum(risk_factors.values()) / len(risk_factors)
        
        # Determine approval requirement
        requires_approval = risk_score > self.risk_threshold
        
        # Check for auto-approve patterns
        query = implementation_plan.get("original_query", "").lower()
        if any(pattern in query for pattern in self.auto_approve_patterns):
            requires_approval = False
            risk_score = min(risk_score, 5)  # Cap risk for auto-approvable items
        
        return {
            "risk_score": risk_score,
            "risk_level": self._get_risk_level(risk_score),
            "risk_factors": risk_factors,
            "requires_approval": requires_approval,
            "approval_reason": self._get_approval_reason(risk_factors, risk_score),
            "mitigation_suggestions": self._get_mitigation_suggestions(risk_factors)
        }
    
    def _assess_code_complexity_risk(self, plan: Dict[str, Any]) -> float:
        """Assess risk based on code complexity"""
        
        # Check implementation steps count
        steps = plan.get("implementation_steps", [])
        step_risk = min(len(steps) * 0.5, 5.0)  # More steps = higher risk
        
        # Check main implementation length
        main_impl = plan.get("main_implementation", "")
        if isinstance(main_impl, str):
            line_count = len(main_impl.split('\n'))
            length_risk = min(line_count / 20, 5.0)  # Risk increases with length
        else:
            length_risk = 2.0
        
        return (step_risk + length_risk) / 2
    
    def _assess_file_operation_risk(self, plan: Dict[str, Any]) -> float:
        """Assess risk based on file operations"""
        
        risk = 1.0  # Base risk
        
        steps = plan.get("implementation_steps", [])
        
        for step in steps:
            action_type = step.get("action_type", "")
            
            if action_type == "create_file":
                risk += 1.0
            elif action_type == "modify_file":
                risk += 2.0  # Modifying is riskier than creating
            elif action_type == "delete_file":
                risk += 3.0  # Deletion is highest risk
            elif action_type == "run_command":
                risk += 4.0  # Running commands is very risky
        
        return min(risk, 10.0)
    
    def _assess_dependency_risk(self, plan: Dict[str, Any]) -> float:
        """Assess risk based on external dependencies"""
        
        # Look for import statements or dependency mentions
        main_impl = plan.get("main_implementation", "")
        
        if not isinstance(main_impl, str):
            return 2.0
        
        risk = 1.0
        
        # Count imports
        import_count = len(re.findall(r'^(?:import|from)\s+', main_impl, re.MULTILINE))
        risk += min(import_count * 0.3, 3.0)
        
        # Check for external package imports
        external_patterns = [
            r'requests',
            r'urllib',
            r'subprocess',
            r'os\.',
            r'sys\.',
            r'socket'
        ]
        
        for pattern in external_patterns:
            if re.search(pattern, main_impl):
                risk += 1.0
        
        return min(risk, 8.0)
    
    def _assess_security_risk(self, plan: Dict[str, Any]) -> float:
        """Assess security-related risks"""
        
        query = plan.get("original_query", "").lower()
        main_impl = plan.get("main_implementation", "")
        
        risk = 1.0
        
        # Check query for security-sensitive terms
        security_terms = [
            "auth", "password", "token", "secret", "api_key",
            "login", "session", "cookie", "encryption", "ssl"
        ]
        
        for term in security_terms:
            if term in query:
                risk += 1.5
        
        # Check implementation for risky patterns
        if isinstance(main_impl, str):
            risky_patterns = [
                r'password\s*=',
                r'secret\s*=',
                r'token\s*=',
                r'api_key\s*=',
                r'eval\(',
                r'exec\('
            ]
            
            for pattern in risky_patterns:
                if re.search(pattern, main_impl, re.IGNORECASE):
                    risk += 2.0
        
        return min(risk, 10.0)
    
    def _assess_data_risk(self, plan: Dict[str, Any]) -> float:
        """Assess data operation risks"""
        
        main_impl = plan.get("main_implementation", "")
        
        if not isinstance(main_impl, str):
            return 2.0
        
        risk = 1.0
        
        # Check for database operations
        db_patterns = [
            r'DELETE\s+FROM',
            r'DROP\s+TABLE',
            r'ALTER\s+TABLE',
            r'UPDATE\s+.*\s+SET',
            r'\.delete\(',
            r'\.drop\(',
            r'\.truncate\('
        ]
        
        for pattern in db_patterns:
            if re.search(pattern, main_impl, re.IGNORECASE):
                risk += 3.0
        
        # Check for file operations
        file_patterns = [
            r'open\s*\(',
            r'\.write\(',
            r'\.delete\(',
            r'os\.remove',
            r'shutil\.'
        ]
        
        for pattern in file_patterns:
            if re.search(pattern, main_impl):
                risk += 1.0
        
        return min(risk, 8.0)
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to human-readable level"""
        
        if risk_score >= 8:
            return "CRITICAL"
        elif risk_score >= 6:
            return "HIGH"
        elif risk_score >= 4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_approval_reason(self, risk_factors: Dict[str, float], risk_score: float) -> str:
        """Get human-readable reason for approval requirement"""
        
        if risk_score <= self.risk_threshold:
            return "Low risk - auto-approved"
        
        high_risk_factors = [k for k, v in risk_factors.items() if v >= 5.0]
        
        if high_risk_factors:
            return f"High risk factors detected: {', '.join(high_risk_factors)}"
        else:
            return f"Overall risk score ({risk_score:.1f}) exceeds threshold ({self.risk_threshold})"
    
    def _get_mitigation_suggestions(self, risk_factors: Dict[str, float]) -> List[str]:
        """Generate mitigation suggestions based on risk factors"""
        
        suggestions = []
        
        if risk_factors.get("code_complexity", 0) >= 5:
            suggestions.append("Consider breaking complex code into smaller, testable functions")
        
        if risk_factors.get("file_operations", 0) >= 5:
            suggestions.append("Add comprehensive error handling for file operations")
        
        if risk_factors.get("external_dependencies", 0) >= 5:
            suggestions.append("Review external dependencies for security and compatibility")
        
        if risk_factors.get("security_sensitive", 0) >= 5:
            suggestions.append("Conduct security review before implementation")
        
        if risk_factors.get("data_operations", 0) >= 5:
            suggestions.append("Implement data backup and validation checks")
        
        # Always suggest testing
        suggestions.append("Implement comprehensive unit and integration tests")
        suggestions.append("Test in development environment before production deployment")
        
        return suggestions
    
    def create_approval_request(self, implementation_plan: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Create a structured approval request for human review"""
        
        return {
            "request_id": f"approval_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "implementation_summary": {
                "query": implementation_plan.get("original_query"),
                "complexity": len(implementation_plan.get("implementation_steps", [])),
                "estimated_effort": implementation_plan.get("estimated_effort", {})
            },
            "risk_assessment": risk_assessment,
            "code_preview": self._create_code_preview(implementation_plan),
            "validation_results": implementation_plan.get("validation_results", {}),
            "recommended_action": self._get_recommended_action(risk_assessment),
            "review_checklist": self._create_review_checklist(risk_assessment)
        }
    
    def _create_code_preview(self, plan: Dict[str, Any]) -> str:
        """Create a preview of the generated code for review"""
        
        main_impl = plan.get("main_implementation", "")
        
        if not isinstance(main_impl, str):
            return "No code preview available"
        
        lines = main_impl.split('\n')
        
        # Show first 20 lines with line numbers
        preview_lines = []
        for i, line in enumerate(lines[:20], 1):
            preview_lines.append(f"{i:3d}: {line}")
        
        preview = '\n'.join(preview_lines)
        
        if len(lines) > 20:
            preview += f"\n... ({len(lines) - 20} more lines)"
        
        return preview
    
    def _get_recommended_action(self, risk_assessment: Dict[str, Any]) -> str:
        """Get recommended action based on risk assessment"""
        
        risk_level = risk_assessment.get("risk_level", "MEDIUM")
        
        if risk_level == "CRITICAL":
            return "REJECT - Too risky for automated implementation"
        elif risk_level == "HIGH":
            return "MANUAL_REVIEW - Requires careful human review"
        elif risk_level == "MEDIUM":
            return "APPROVE_WITH_CONDITIONS - Approve with additional safeguards"
        else:
            return "AUTO_APPROVE - Low risk implementation"
    
    def _create_review_checklist(self, risk_assessment: Dict[str, Any]) -> List[str]:
        """Create a checklist for human reviewers"""
        
        checklist = [
            "Code syntax and logic review",
            "Security implications assessment",
            "Integration impact analysis",
            "Test coverage verification"
        ]
        
        risk_factors = risk_assessment.get("risk_factors", {})
        
        if risk_factors.get("security_sensitive", 0) >= 4:
            checklist.append("Security expert review required")
        
        if risk_factors.get("data_operations", 0) >= 4:
            checklist.append("Data operations impact assessment")
        
        if risk_factors.get("external_dependencies", 0) >= 4:
            checklist.append("Dependency compatibility check")
        
        return checklist 