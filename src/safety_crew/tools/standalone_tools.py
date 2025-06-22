"""Standalone Safety Tools - Direct execution without CrewAI wrappers"""

import subprocess
import tempfile
import json
import re
import logging
import ast
import importlib.util
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class StandaloneSemgrepScanner:
    """Standalone Semgrep security scanner for batch execution"""
    
    def __init__(self):
        self.rule_sets = [
            "owasp-top-ten",
            "cwe-top-25", 
            "security-audit",
            "secrets"
        ]
    
    def scan(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Run Semgrep security scan on provided code
        
        Args:
            code: Source code to analyze
            language: Programming language (default: python)
            
        Returns:
            Dict containing scan results with findings
        """
        try:
            # Create temporary file for analysis
            with tempfile.NamedTemporaryFile(
                mode='w', 
                suffix=f'.{self._get_file_extension(language)}',
                delete=False
            ) as temp_file:
                temp_file.write(code)
                temp_file_path = temp_file.name
            
            all_findings = []
            
            # Run Semgrep with each rule set
            for rule_set in self.rule_sets:
                try:
                    findings = self._run_semgrep_rule_set(temp_file_path, rule_set)
                    if findings:
                        all_findings.extend(findings)
                except Exception as e:
                    logger.warning(f"Failed to run Semgrep rule set {rule_set}: {e}")
                    continue
            
            # Clean up temp file
            Path(temp_file_path).unlink(missing_ok=True)
            
            return {
                "tool": "semgrep_scanner",
                "total_findings": len(all_findings),
                "findings": all_findings,
                "rule_sets_used": self.rule_sets,
                "language": language,
                "scan_status": "completed"
            }
                        
        except Exception as e:
            logger.error(f"Semgrep scan failed: {e}")
            return {
                "tool": "semgrep_scanner", 
                "total_findings": 0,
                "findings": [],
                "error": str(e),
                "scan_status": "failed"
            }
    
    def _get_file_extension(self, language: str) -> str:
        """Get appropriate file extension for language"""
        extensions = {
            "python": "py",
            "javascript": "js", 
            "typescript": "ts",
            "java": "java",
            "go": "go",
            "rust": "rs",
            "c": "c",
            "cpp": "cpp",
            "php": "php"
        }
        return extensions.get(language.lower(), "txt")
    
    def _run_semgrep_rule_set(self, file_path: str, rule_set: str) -> List[Dict[str, Any]]:
        """Run Semgrep with specific rule set"""
        try:
            cmd = [
                "semgrep",
                "--config", f"p/{rule_set}",
                "--json",
                "--quiet",
                "--timeout", "30",
                file_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                findings = []
                
                for finding in data.get("results", []):
                    findings.append({
                        "rule_id": finding.get("check_id", "unknown"),
                        "message": finding.get("message", "No message"),
                        "severity": finding.get("extra", {}).get("severity", "INFO"),
                        "line": finding.get("start", {}).get("line", 1),
                        "column": finding.get("start", {}).get("col", 1),
                        "rule_set": rule_set,
                        "cwe": finding.get("extra", {}).get("metadata", {}).get("cwe", []),
                        "owasp": finding.get("extra", {}).get("metadata", {}).get("owasp", [])
                    })
                
                return findings
            
        except subprocess.TimeoutExpired:
            logger.warning(f"Semgrep timeout for rule set {rule_set}")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Semgrep output for {rule_set}: {e}")
        except Exception as e:
            logger.warning(f"Semgrep execution failed for {rule_set}: {e}")
        
        return []


class StandaloneSecurityPatternAnalyzer:
    """Standalone security pattern analyzer for batch execution"""
    
    def __init__(self):
        self.security_patterns = {
            # SQL Injection patterns
            "sql_injection": [
                r'(?i)(execute|exec)\s*\(\s*["\'].*?%.*?["\']',
                r'(?i)(?:select|insert|update|delete|drop).*?%.*?(?:from|into|set|table)',
                r'(?i)["\'][^"\']*\+.*?(?:select|insert|update|delete|union)',
                r'(?i)cursor\.execute\s*\([^)]*%[^)]*\)',
                r'(?i)query\s*=.*?["\'][^"\']*%.*?["\']'
            ],
            
            # Command injection patterns  
            "command_injection": [
                r'(?i)(?:os\.system|subprocess\.call|subprocess\.run|subprocess\.Popen)\s*\([^)]*\+',
                r'(?i)(?:eval|exec)\s*\([^)]*(?:request|input|raw_input)',
                r'(?i)(?:shell=True).*?(?:request|input|argv)',
                r'(?i)os\.popen\s*\([^)]*\+'
            ],
            
            # Path traversal patterns
            "path_traversal": [
                r'(?:\.\.\/|\.\.\\)+',
                r'(?i)open\s*\([^)]*(?:request|input|argv)',
                r'(?i)(?:file|path)\s*=.*?(?:request|input|argv)',
                r'(?i)os\.path\.join\s*\([^)]*(?:request|input|argv)'
            ],
            
            # Hardcoded secrets patterns
            "hardcoded_secrets": [
                r'(?i)(?:password|passwd|pwd)\s*=\s*["\'][^"\']{3,}["\']',
                r'(?i)(?:api_key|apikey|access_key)\s*=\s*["\'][^"\']{10,}["\']',
                r'(?i)(?:secret|token|auth)\s*=\s*["\'][^"\']{8,}["\']',
                r'(?i)["\'][A-Za-z0-9+/]{40,}={0,2}["\']',  # Base64-like strings
                r'(?i)(?:sk_|pk_|ak_)[a-zA-Z0-9]{20,}'      # Key prefixes
            ],
            
            # Weak crypto patterns
            "weak_crypto": [
                r'(?i)md5\s*\(',
                r'(?i)sha1\s*\(',
                r'(?i)des\s*\(',
                r'(?i)random\.random\s*\(',
                r'(?i)ssl\._create_unverified_context'
            ]
        }
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """
        Analyze code for security patterns
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dict containing pattern analysis results
        """
        try:
            findings = []
            lines = code.split('\n')
            
            for pattern_type, patterns in self.security_patterns.items():
                for pattern in patterns:
                    for line_num, line in enumerate(lines, 1):
                        matches = re.finditer(pattern, line)
                        for match in matches:
                            findings.append({
                                "pattern_type": pattern_type,
                                "line": line_num,
                                "column": match.start() + 1,
                                "matched_text": match.group(),
                                "pattern": pattern,
                                "severity": self._assess_severity(pattern_type),
                                "description": self._get_pattern_description(pattern_type)
                            })
                    
            return {
                "tool": "security_pattern_analyzer",
                "total_findings": len(findings),
                "findings": findings,
                "patterns_checked": list(self.security_patterns.keys()),
                "analysis_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Security pattern analysis failed: {e}")
            return {
                "tool": "security_pattern_analyzer",
                "total_findings": 0,
                "findings": [],
                "error": str(e),
                "analysis_status": "failed"
            }
    
    def _assess_severity(self, pattern_type: str) -> str:
        """Assess severity level for pattern type"""
        severity_map = {
            "sql_injection": "HIGH",
            "command_injection": "CRITICAL", 
            "path_traversal": "HIGH",
            "hardcoded_secrets": "CRITICAL",
            "weak_crypto": "MEDIUM"
        }
        return severity_map.get(pattern_type, "MEDIUM")
    
    def _get_pattern_description(self, pattern_type: str) -> str:
        """Get description for pattern type"""
        descriptions = {
            "sql_injection": "Potential SQL injection vulnerability detected",
            "command_injection": "Potential command injection vulnerability detected",
            "path_traversal": "Potential path traversal vulnerability detected", 
            "hardcoded_secrets": "Hardcoded secret or credential detected",
            "weak_crypto": "Weak cryptographic function usage detected"
        }
        return descriptions.get(pattern_type, "Security pattern detected")


class StandaloneHallucinationDetector:
    """Standalone hallucination detector for batch execution"""
    
    def __init__(self, rag_system=None, context_manager=None):
        self.rag_system = rag_system
        self.context_manager = context_manager
        # Initialize these first before calling them
        self.stdlib_modules = self._get_stdlib_modules()
        self.common_packages = self._get_common_packages()
    
    def _get_stdlib_modules(self) -> set:
        """Get set of standard library module names"""
        import sys
        return set(sys.stdlib_module_names) if hasattr(sys, 'stdlib_module_names') else set()
    
    def _get_common_packages(self) -> List[str]:
        """Get list of common package prefixes"""
        return [
            "numpy", "pandas", "matplotlib", "seaborn", "scikit", "scipy",
            "requests", "flask", "django", "fastapi", "click", "typer",
            "pytest", "unittest", "logging", "json", "csv", "xml", "yaml",
            "boto3", "google", "azure", "aws", "redis", "pymongo",
            "sqlalchemy", "psycopg2", "mysql", "sqlite3"
        ]
    
    def detect(self, code: str, session_id: str) -> Dict[str, Any]:
        """
        Detect potential AI hallucinations in code
        
        Args:
            code: Source code to analyze
            session_id: Session identifier for context
            
        Returns:
            Dict containing hallucination detection results
        """
        try:
            findings = []
            
            # Parse code into AST for analysis
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {
                    "tool": "hallucination_detector",
                    "total_findings": 1,
                    "findings": [{
                        "type": "syntax_error",
                        "line": e.lineno or 1,
                        "message": f"Syntax error: {e.msg}",
                        "confidence": 100,
                        "severity": "HIGH"
                    }],
                    "session_id": session_id,
                    "detection_status": "completed"
                }
            
            # Analyze imports
            import_findings = self._analyze_imports(tree)
            findings.extend(import_findings)
            
            # Analyze function calls
            call_findings = self._analyze_function_calls(tree, code)
            findings.extend(call_findings)
            
            # Analyze attribute access
            attr_findings = self._analyze_attributes(tree)
            findings.extend(attr_findings)
                    
            return {
                "tool": "hallucination_detector", 
                "total_findings": len(findings),
                "findings": findings,
                "session_id": session_id,
                "detection_status": "completed",
                "confidence_summary": self._calculate_confidence_summary(findings)
            }
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return {
                "tool": "hallucination_detector",
                "total_findings": 0,
                "findings": [],
                "error": str(e),
                "session_id": session_id,
                "detection_status": "failed"
            }
    
    def _analyze_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze import statements for potential hallucinations"""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not self._is_valid_module(alias.name):
                        findings.append({
                            "type": "suspicious_import",
                            "line": node.lineno,
                            "module": alias.name,
                            "message": f"Suspicious import: {alias.name} may not exist",
                            "confidence": 75,
                            "severity": "MEDIUM"
                        })
            
            elif isinstance(node, ast.ImportFrom):
                if node.module and not self._is_valid_module(node.module):
                    findings.append({
                        "type": "suspicious_import",
                        "line": node.lineno,
                        "module": node.module,
                        "message": f"Suspicious import from: {node.module} may not exist",
                        "confidence": 75,
                        "severity": "MEDIUM"
                    })
        
        return findings
    
    def _analyze_function_calls(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Analyze function calls for potential hallucinations"""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = self._get_function_name(node.func)
                if func_name and self._is_suspicious_function_call(func_name):
                    findings.append({
                        "type": "suspicious_function_call",
                        "line": node.lineno,
                        "function": func_name,
                        "message": f"Suspicious function call: {func_name} may not exist",
                        "confidence": 60,
                        "severity": "LOW"
                    })
        
        return findings
    
    def _analyze_attributes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Analyze attribute access for potential hallucinations"""
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                attr_chain = self._get_attribute_chain(node)
                if self._is_suspicious_attribute(attr_chain):
                    findings.append({
                        "type": "suspicious_attribute",
                        "line": node.lineno,
                        "attribute": attr_chain,
                        "message": f"Suspicious attribute access: {attr_chain}",
                        "confidence": 50,
                        "severity": "LOW"
                    })
        
        return findings
    
    def _is_valid_module(self, module_name: str) -> bool:
        """Check if module name is valid"""
        # Check stdlib modules
        if module_name in self.stdlib_modules:
            return True
        
        # Check common packages
        if any(module_name.startswith(pkg) for pkg in self.common_packages):
            return True
        
        # Try to find module spec
        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            return False
    
    def _is_suspicious_function_call(self, func_name: str) -> bool:
        """Check if function call is suspicious"""
        # Very basic heuristic - look for obviously fake functions
        suspicious_patterns = [
            "magic_",
            "auto_",
            "smart_", 
            "intelligent_",
            "ai_"
        ]
        return any(pattern in func_name.lower() for pattern in suspicious_patterns)
    
    def _is_suspicious_attribute(self, attr_chain: str) -> bool:
        """Check if attribute chain is suspicious"""
        # Basic heuristic for suspicious attributes
        return "magic_" in attr_chain.lower() or "auto_" in attr_chain.lower()
    
    def _get_function_name(self, node: ast.AST) -> Optional[str]:
        """Extract function name from call node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_chain(node)
        return None
    
    def _get_attribute_chain(self, node: ast.Attribute) -> str:
        """Get full attribute chain as string"""
        parts = []
        current = node
        
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        
        if isinstance(current, ast.Name):
            parts.append(current.id)
        
        return ".".join(reversed(parts))
    
    def _calculate_confidence_summary(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate confidence summary statistics"""
        if not findings:
            return {"average_confidence": 0, "high_confidence_count": 0}
        
        confidences = [f["confidence"] for f in findings]
        return {
            "average_confidence": sum(confidences) / len(confidences),
            "high_confidence_count": len([c for c in confidences if c >= 80]),
            "medium_confidence_count": len([c for c in confidences if 50 <= c < 80]),
            "low_confidence_count": len([c for c in confidences if c < 50])
        }


class StandaloneCodeQualityAnalyzer:
    """Standalone code quality analyzer for batch execution"""
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """
        Analyze code quality metrics
        
        Args:
            code: Source code to analyze
            
        Returns:
            Dict containing quality analysis results  
        """
        try:
            # Parse code into AST
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {
                    "tool": "code_quality_analyzer",
                    "error": f"Syntax error: {e.msg}",
                    "analysis_status": "failed"
                }
            
            metrics = {
                "lines_of_code": len(code.split('\n')),
                "cyclomatic_complexity": self._calculate_complexity(tree),
                "function_count": self._count_functions(tree),
                "class_count": self._count_classes(tree),
                "average_function_length": self._calculate_avg_function_length(tree, code),
                "maintainability_index": self._calculate_maintainability_index(tree, code),
                "code_smells": self._detect_code_smells(tree, code)
            }
                
            return {
                "tool": "code_quality_analyzer",
                "metrics": metrics,
                "overall_score": self._calculate_overall_score(metrics),
                "analysis_status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Code quality analysis failed: {e}")
            return {
                "tool": "code_quality_analyzer",
                "error": str(e),
                "analysis_status": "failed"
            }
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def _count_functions(self, tree: ast.AST) -> int:
        """Count function definitions"""
        return len([node for node in ast.walk(tree) 
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))])
    
    def _count_classes(self, tree: ast.AST) -> int:
        """Count class definitions"""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
    
    def _calculate_avg_function_length(self, tree: ast.AST, code: str) -> float:
        """Calculate average function length in lines"""
        functions = [node for node in ast.walk(tree) 
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        
        if not functions:
            return 0
        
        lines = code.split('\n')
        total_length = 0
        
        for func in functions:
            start_line = func.lineno - 1
            end_line = func.end_lineno if hasattr(func, 'end_lineno') else start_line + 10
            total_length += end_line - start_line
        
        return total_length / len(functions)
    
    def _calculate_maintainability_index(self, tree: ast.AST, code: str) -> float:
        """Calculate simplified maintainability index"""
        lines = len(code.split('\n'))
        complexity = self._calculate_complexity(tree)
        
        # Simplified formula
        if lines == 0:
            return 100
        
        mi = max(0, 171 - 5.2 * (complexity / lines * 100) - 0.23 * complexity - 16.2 * (lines / 100))
        return min(100, mi)
    
    def _detect_code_smells(self, tree: ast.AST, code: str) -> List[Dict[str, Any]]:
        """Detect basic code smells"""
        smells = []
        
        # Long functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 50
                
                if end_line - start_line > 50:
                    smells.append({
                        "type": "long_function",
                        "line": start_line,
                        "message": f"Function '{node.name}' is too long ({end_line - start_line} lines)",
                        "severity": "MEDIUM"
                    })
                
                # Too many parameters
                if len(node.args.args) > 5:
                    smells.append({
                        "type": "too_many_parameters",
                        "line": start_line,
                        "message": f"Function '{node.name}' has too many parameters ({len(node.args.args)})",
                        "severity": "LOW"
                    })
        
        # Deep nesting
        max_depth = self._calculate_max_nesting_depth(tree)
        if max_depth > 4:
            smells.append({
                "type": "deep_nesting",
                "line": 1,
                "message": f"Maximum nesting depth is too high ({max_depth})",
                "severity": "MEDIUM"
            })
        
        return smells
    
    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)):
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            
            return max_depth
        
        return get_depth(tree)
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> int:
        """Calculate overall quality score"""
        score = 100
        
        # Penalize based on metrics
        if metrics["cyclomatic_complexity"] > 10:
            score -= 20
        elif metrics["cyclomatic_complexity"] > 5:
            score -= 10
        
        if metrics["average_function_length"] > 30:
            score -= 15
        elif metrics["average_function_length"] > 20:
            score -= 5
        
        score -= len(metrics["code_smells"]) * 3
        
        # Factor in maintainability index
        mi_score = metrics["maintainability_index"]
        if mi_score < 50:
            score -= 25
        elif mi_score < 70:
            score -= 10
        
        return max(0, score)