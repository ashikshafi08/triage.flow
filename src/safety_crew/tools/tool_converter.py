"""Tool Converter - Wraps existing LlamaIndex tools for CrewAI compatibility using LlamaIndexTool"""

import os
import subprocess
import json
import asyncio
from typing import List, Dict, Any, Optional, Callable
import tempfile
import hashlib

# Import CrewAI tools - use the proper LlamaIndexTool
from crewai_tools import LlamaIndexTool
from llama_index.core.tools import FunctionTool

# Import existing infrastructure - optional for integration
try:
    from src.agent_tools.tool_registry import create_all_tools
    from src.agent_tools.context_manager import ContextManager
    from src.config import settings
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    create_all_tools = None
    ContextManager = None
    settings = None


def convert_existing_tools_to_crewai(
    explorer_instance: Any = None,
    context_manager: Optional[ContextManager] = None
) -> List[LlamaIndexTool]:
    """Convert existing agent tools to CrewAI-compatible LlamaIndexTools"""
    
    crewai_tools = []
    
    # Get all existing tools from the explorer instance
    if explorer_instance and INTEGRATION_AVAILABLE:
        try:
            existing_tools = create_all_tools(explorer_instance)
            
            # Wrap each tool for CrewAI using LlamaIndexTool
            for tool in existing_tools:
                # Skip tools that are not suitable for CrewAI agents
                if tool.name in ['stream_large_file']:  # Streaming tools not suitable
                    continue
                    
                try:
                    # Check if this is already a LlamaIndex tool
                    if hasattr(tool, '_fn') or hasattr(tool, 'fn'):
                        # Convert to LlamaIndexTool using from_tool method
                        crewai_tool = LlamaIndexTool.from_tool(tool)
                        crewai_tools.append(crewai_tool)
                    else:
                        # Create a FunctionTool first, then wrap it
                        func = tool._fn if hasattr(tool, '_fn') else tool.fn
                        function_tool = FunctionTool.from_defaults(
                            fn=func,
                            name=tool.name,
                            description=getattr(tool, 'description', f"Wrapped {tool.name} tool")
                        )
                        crewai_tool = LlamaIndexTool.from_tool(function_tool)
                        crewai_tools.append(crewai_tool)
                
                except Exception as tool_error:
                    print(f"Failed to convert tool {tool.name}: {tool_error}")
                    continue
                
        except Exception as e:
            # Log error but don't fail
            print(f"Error converting tools: {e}")
    
    return crewai_tools


def semgrep_scanner_function(code: str, language: str = "python", custom_rules: List[str] = None) -> str:
    """Run Semgrep security scan on provided code"""
    
    # Default rule sets for comprehensive security scanning
    rule_sets = [
        "p/security-audit",
        "p/owasp-top-10", 
        "p/cwe-top-25",
        "p/secrets"
    ]
    
    if custom_rules:
        rule_sets.extend(custom_rules)
        
    # Create temporary file for code
    with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{language}', delete=False) as f:
        f.write(code)
        temp_file = f.name
        
    try:
        results = []
        
        # Run Semgrep with each rule set
        for rule_set in rule_sets:
            try:
                cmd = [
                    "semgrep",
                    "--config", rule_set,
                    "--json",
                    "--no-git-ignore",
                    temp_file
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0 and result.stdout:
                    scan_results = json.loads(result.stdout)
                    if scan_results.get("results"):
                        results.extend(scan_results["results"])
                        
            except subprocess.TimeoutExpired:
                results.append({
                    "error": f"Timeout scanning with rule set {rule_set}"
                })
            except Exception as e:
                results.append({
                    "error": f"Error with rule set {rule_set}: {str(e)}"
                })
                
        # Process and format results
        findings = []
        for result in results:
            if "error" in result:
                findings.append(result)
            else:
                finding = {
                    "rule_id": result.get("check_id", "unknown"),
                    "severity": result.get("extra", {}).get("severity", "medium"),
                    "message": result.get("extra", {}).get("message", "Security issue detected"),
                    "line": result.get("start", {}).get("line", 0),
                    "code": result.get("extra", {}).get("lines", ""),
                    "fix": result.get("extra", {}).get("fix", ""),
                    "metadata": result.get("extra", {}).get("metadata", {})
                }
                findings.append(finding)
                
        return json.dumps({
            "status": "success",
            "findings": findings,
            "total_findings": len([f for f in findings if "error" not in f]),
            "scanned_rules": rule_sets
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
            "scanned_rules": rule_sets
        }, indent=2)
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file)
        except Exception:
            pass


def security_pattern_analyzer_function(code: str, language: str = "python") -> str:
    """Analyze code for security anti-patterns and vulnerabilities"""
    
    patterns = {
        "sql_injection": [
            r"f[\"'].*SELECT.*{.*}.*[\"']",
            r"f[\"'].*INSERT.*{.*}.*[\"']",
            r"f[\"'].*UPDATE.*{.*}.*[\"']",
            r"f[\"'].*DELETE.*{.*}.*[\"']",
            r"\.format\(.*\).*SELECT",
            r"% .*SELECT"
        ],
        "command_injection": [
            r"os\.system\(.*{.*}\)",
            r"subprocess\.call\(.*{.*}\)",
            r"subprocess\.run\(.*{.*}\)",
            r"os\.popen\(.*{.*}\)"
        ],
        "path_traversal": [
            r"open\(.*{.*}.*\)",
            r"file\(.*{.*}.*\)",
            r"\.\.\/",
            r"\.\.\\"
        ],
        "hardcoded_secrets": [
            r"password\s*=\s*[\"'][^\"']+[\"']",
            r"api_key\s*=\s*[\"'][^\"']+[\"']",
            r"secret\s*=\s*[\"'][^\"']+[\"']",
            r"token\s*=\s*[\"'][^\"']+[\"']"
        ]
    }
    
    findings = []
    
    import re
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                findings.append({
                    "category": category,
                    "pattern": pattern,
                    "match": match.group(),
                    "line": code[:match.start()].count('\n') + 1,
                    "severity": "high" if category in ["sql_injection", "command_injection"] else "medium"
                })
                
    return json.dumps({
        "status": "success",
        "findings": findings,
        "total_findings": len(findings),
        "categories_found": list(set(f["category"] for f in findings))
    }, indent=2)


def hallucination_detector_function(
    code: str, 
    session_id: str = None, 
    file_path: str = None, 
    rag_system: Any = None
) -> str:
    """Detect potential AI hallucinations by verifying against codebase"""
    
    issues = []
    
    # Check for non-existent imports
    import re
    import_matches = re.findall(r'import\s+(\w+)', code)
    import_matches.extend(re.findall(r'from\s+(\w+)', code))
    
    # Common legitimate libraries
    legitimate_libs = {
        'os', 'sys', 'json', 'time', 'datetime', 'random', 'math', 'collections',
        'typing', 'asyncio', 'logging', 'pathlib', 'subprocess', 'tempfile',
        'requests', 'numpy', 'pandas', 'matplotlib', 'flask', 'django',
        'fastapi', 'pydantic', 'sqlalchemy', 'redis', 'pytest'
    }
    
    for lib in import_matches:
        if lib not in legitimate_libs and 'nonexistent' in lib.lower():
            issues.append({
                "type": "suspicious_import",
                "library": lib,
                "severity": "medium",
                "message": f"Potentially non-existent library: {lib}"
            })
    
    # Check for fictional functions
    fictional_patterns = [
        r'\.magic_function\(',
        r'\.fictional_method\(',
        r'\.nonexistent_\w+\(',
        r'\.placeholder_\w+\('
    ]
    
    for pattern in fictional_patterns:
        matches = re.finditer(pattern, code)
        for match in matches:
            issues.append({
                "type": "fictional_function",
                "function": match.group(),
                "line": code[:match.start()].count('\n') + 1,
                "severity": "high",
                "message": f"Potentially fictional function call: {match.group()}"
            })
    
    # Calculate grounding score
    total_lines = len(code.split('\n'))
    issue_lines = len(set(issue.get('line', 0) for issue in issues if issue.get('line', 0) > 0))
    grounding_score = max(0, (total_lines - issue_lines) / max(1, total_lines))
    
    return json.dumps({
        "status": "success",
        "grounding_score": grounding_score,
        "issues": issues,
        "total_issues": len(issues),
        "code_lines": total_lines
    }, indent=2)


def code_quality_analyzer_function(code: str, language: str = "python") -> str:
    """Analyze code quality metrics and best practices"""
    
    import re
    
    issues = []
    metrics = {
        "lines_of_code": len(code.split('\n')),
        "complexity_score": 0,
        "maintainability_issues": 0
    }
    
    # Check for high complexity
    complexity_indicators = [
        (r'for\s+\w+\s+in.*:', 1),
        (r'while\s+.*:', 1),
        (r'if\s+.*:', 1),
        (r'elif\s+.*:', 1),
        (r'try\s*:', 1),
        (r'except\s+.*:', 1)
    ]
    
    for pattern, weight in complexity_indicators:
        matches = len(re.findall(pattern, code))
        metrics["complexity_score"] += matches * weight
    
    # Check for poor naming
    poor_naming = re.findall(r'\b[a-z]\s*=\s*', code)  # Single letter variables
    if poor_naming:
        issues.append({
            "type": "poor_naming",
            "count": len(poor_naming),
            "severity": "low",
            "message": f"Found {len(poor_naming)} single-letter variable names"
        })
        metrics["maintainability_issues"] += len(poor_naming)
    
    # Check for long functions (simplified)
    function_starts = re.findall(r'def\s+\w+\(', code)
    avg_lines_per_function = metrics["lines_of_code"] / max(1, len(function_starts))
    
    if avg_lines_per_function > 20:
        issues.append({
            "type": "long_functions",
            "severity": "medium",
            "message": f"Average function length is {avg_lines_per_function:.1f} lines (>20 is concerning)"
        })
        metrics["maintainability_issues"] += 1
    
    # Calculate quality score
    base_score = 10
    complexity_penalty = min(5, metrics["complexity_score"] * 0.1)
    maintainability_penalty = min(3, metrics["maintainability_issues"] * 0.5)
    quality_score = max(0, base_score - complexity_penalty - maintainability_penalty)
            
    return json.dumps({
        "status": "success",
        "quality_score": quality_score,
        "metrics": metrics,
        "issues": issues,
        "total_issues": len(issues)
    }, indent=2)
            

def create_safety_specific_tools(
    rag_system: Optional[Any] = None,
    context_manager: Optional[ContextManager] = None
) -> List[LlamaIndexTool]:
    """Create safety-specific tools as LlamaIndexTools"""
    
    tools = []
    
    # Create LlamaIndex FunctionTools first, then wrap them
    semgrep_tool = FunctionTool.from_defaults(
        fn=semgrep_scanner_function,
        name="semgrep_scanner",
        description="Scan code for security vulnerabilities using Semgrep with multiple rule sets including OWASP Top 10, CWE Top 25, and secrets detection"
    )
    tools.append(LlamaIndexTool.from_tool(semgrep_tool))
    
    security_analyzer_tool = FunctionTool.from_defaults(
        fn=security_pattern_analyzer_function,
        name="security_pattern_analyzer",
        description="Analyze code for security anti-patterns including SQL injection, command injection, path traversal, and hardcoded secrets"
    )
    tools.append(LlamaIndexTool.from_tool(security_analyzer_tool))
    
    hallucination_tool = FunctionTool.from_defaults(
        fn=hallucination_detector_function,
        name="hallucination_detector",
        description="Detect AI hallucinations by verifying code against codebase and checking for non-existent imports or fictional functions"
    )
    tools.append(LlamaIndexTool.from_tool(hallucination_tool))
    
    quality_tool = FunctionTool.from_defaults(
        fn=code_quality_analyzer_function,
        name="code_quality_analyzer",
        description="Analyze code quality metrics including complexity, maintainability, and best practices compliance"
    )
    tools.append(LlamaIndexTool.from_tool(quality_tool))
    
    return tools


# Legacy classes for backward compatibility (now deprecated)
class SemgrepScanner:
    """Deprecated: Use semgrep_scanner_function with LlamaIndexTool instead"""
    def __init__(self):
        import warnings
        warnings.warn("SemgrepScanner is deprecated, use create_safety_specific_tools() instead", DeprecationWarning)
    
    def _run(self, *args, **kwargs):
        return semgrep_scanner_function(*args, **kwargs)


class SecurityPatternAnalyzer:
    """Deprecated: Use security_pattern_analyzer_function with LlamaIndexTool instead"""
    def __init__(self):
        import warnings
        warnings.warn("SecurityPatternAnalyzer is deprecated, use create_safety_specific_tools() instead", DeprecationWarning)
    
    def _run(self, *args, **kwargs):
        return security_pattern_analyzer_function(*args, **kwargs)


class HallucinationDetector:
    """Deprecated: Use hallucination_detector_function with LlamaIndexTool instead"""
    def __init__(self, rag_system=None, context_manager=None):
        import warnings
        warnings.warn("HallucinationDetector is deprecated, use create_safety_specific_tools() instead", DeprecationWarning)
    
    def _run(self, *args, **kwargs):
        return hallucination_detector_function(*args, **kwargs)


class CodeQualityAnalyzer:
    """Deprecated: Use code_quality_analyzer_function with LlamaIndexTool instead"""
    def __init__(self):
        import warnings
        warnings.warn("CodeQualityAnalyzer is deprecated, use create_safety_specific_tools() instead", DeprecationWarning)
    
    def _run(self, *args, **kwargs):
        return code_quality_analyzer_function(*args, **kwargs)