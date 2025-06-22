# src/agent_tools/response_handling.py

import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)

# Compile regex patterns once for better performance
REACT_PATTERNS = {
    'main_sections': re.compile(r'^(Thought:|Action:|Action Input:|Observation:|Answer:|Final Answer:)', re.MULTILINE),
    'section_headers': re.compile(r'^(Thought:|Action:|Action Input:|Observation:|Answer:|Final Answer:)'),
    'log_patterns': re.compile(r'^\s*(DEBUG|INFO|WARNING|ERROR):'),
    'http_request': re.compile(r'HTTP Request:'),
    'running_step': re.compile(r'> Running step'),
    'ansi_escape': re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])'),
    'ansi_codes': re.compile(r'\033\[[0-9;]*m|\[0m|\[1;3;[0-9]+m')
}

# Maximum sizes for response optimization
MAX_RESPONSE_SIZE = 50000  # 50KB max response size
MAX_STEP_CONTENT_SIZE = 5000  # 5KB max per step
MAX_OBSERVATION_PREVIEW = 500  # 500 chars for observation preview

# Import the professional response formatter
try:
    from ..response_formatter import ResponseFormatter, ResponseType
except ImportError:
    # Fallback if formatter not available
    ResponseFormatter = None
    ResponseType = None

@lru_cache(maxsize=128)
def _get_section_boundaries(text_hash: str, text: str) -> List[Tuple[int, str, str]]:
    """Cache section boundary detection for repeated parsing."""
    boundaries = []
    for match in REACT_PATTERNS['main_sections'].finditer(text):
        boundaries.append((match.start(), match.group(1), match.group(0)))
    return boundaries

def parse_react_steps(raw_response: str):
    """Optimized ReAct steps parser with caching and size limits."""
    if not raw_response or len(raw_response.strip()) == 0:
        return [], None
    
    # Truncate overly large responses
    if len(raw_response) > MAX_RESPONSE_SIZE:
        logger.warning(f"Truncating large response from {len(raw_response)} to {MAX_RESPONSE_SIZE} chars")
        raw_response = raw_response[:MAX_RESPONSE_SIZE] + "\n[Response truncated for performance]"
    
    logger.debug(f"Parsing ReAct trace (length: {len(raw_response)})")
    
    # Use optimized section-based parsing instead of line-by-line
    text_hash = str(hash(raw_response))
    sections = _get_section_boundaries(text_hash, raw_response)
    
    if not sections:
        # No ReAct structure found, return raw response as answer
        return [{"type": "answer", "content": _truncate_content(raw_response.strip()), "step": 0}], raw_response.strip()
    
    steps = []
    step_counter = 0
    
    for i, (start_pos, section_type, full_match) in enumerate(sections):
        end_pos = sections[i + 1][0] if i + 1 < len(sections) else len(raw_response)
        section_content = raw_response[start_pos:end_pos]
        
        current_step_data = {"step": step_counter}
        content_lines = section_content.split('\n')
        
        # Extract content after the section header
        header_line = content_lines[0]
        content_text = '\n'.join(content_lines[1:]).strip() if len(content_lines) > 1 else ""
        
        # Remove section prefix from first line if present
        if ':' in header_line:
            remaining_text = header_line.split(':', 1)[1].strip()
            if remaining_text:
                content_text = remaining_text + ('\n' + content_text if content_text else '')
        
        # Truncate large content for performance
        content_text = _truncate_content(content_text)
        
        current_step_data = _parse_section_optimized(section_type, content_text, step_counter)
        if current_step_data:
            steps.append(current_step_data)
            step_counter += 1
    
    # Extract final answer from steps if present
    final_answer = None
    for step in reversed(steps):
        if step.get("type") == "answer":
            final_answer = step.get("content")
            break
    
    # If no final answer found but we have steps, use the last observation
    if not final_answer and steps:
        for step in reversed(steps):
            if step.get("type") == "observation":
                final_answer = step.get("content")
                break
    
    # If still no final answer, use raw response if it looks clean
    if not final_answer and raw_response.strip():
        if not any(pattern.search(raw_response) for pattern in [
            REACT_PATTERNS['log_patterns'],
            REACT_PATTERNS['http_request'], 
            REACT_PATTERNS['running_step']
        ]):
            final_answer = raw_response.strip()
    
    logger.debug(f"Parsed {len(steps)} steps. Final answer: {bool(final_answer)}")
    return steps, final_answer

def _parse_section_optimized(section_type: str, content: str, step_num: int) -> Optional[Dict[str, Any]]:
    """Optimized section parsing with reduced complexity."""
    if not content.strip():
        return None
    
    section_type_clean = section_type.rstrip(':')
    
    if section_type_clean == "Thought":
        return {
            "type": "thought",
            "content": content,
            "step": step_num
        }
    
    elif section_type_clean == "Action":
        return {
            "type": "action", 
            "tool_name": content.strip(),
            "content": f"Calling tool: {content.strip()}",
            "step": step_num
        }
    
    elif section_type_clean == "Action Input":
        # Try to parse as JSON, fallback to string
        tool_input = _safe_json_parse(content)
        return {
            "type": "action_input",
            "tool_input": tool_input,
            "content": str(tool_input),
            "step": step_num
        }
    
    elif section_type_clean == "Observation":
        parsed_content = _safe_json_parse(content) 
        preview = _create_observation_preview(parsed_content)
        return {
            "type": "observation",
            "content": parsed_content,
            "tool_output_preview": preview,
            "step": step_num
        }
    
    elif section_type_clean in ["Answer", "Final Answer"]:
        return {
            "type": "answer",
            "content": content,
            "step": step_num
        }
    
    return None

def _safe_json_parse(text: str) -> Any:
    """Safely parse JSON with fallback to original text."""
    text_stripped = text.strip()
    if not text_stripped or not (text_stripped.startswith(('{', '[')) and text_stripped.endswith(('}', ']'))):
        return text_stripped
    
    try:
        return json.loads(text_stripped)
    except (json.JSONDecodeError, ValueError):
        return text_stripped

def _create_observation_preview(content: Any) -> str:
    """Create a concise preview of observation content."""
    if isinstance(content, dict):
        keys = list(content.keys())[:3]
        return f"JSON data (Keys: {keys}{'...' if len(content) > 3 else ''})"
    elif isinstance(content, list):
        length = len(content)
        first_item = str(content[0])[:50] + "..." if content and len(str(content[0])) > 50 else str(content[0]) if content else ""
        return f"JSON array (Length: {length}, First: {first_item})"
    elif isinstance(content, str):
        return content[:MAX_OBSERVATION_PREVIEW] + "..." if len(content) > MAX_OBSERVATION_PREVIEW else content
    else:
        return str(content)[:MAX_OBSERVATION_PREVIEW] + "..." if len(str(content)) > MAX_OBSERVATION_PREVIEW else str(content)

def _truncate_content(content: str) -> str:
    """Truncate content to prevent memory issues."""
    if len(content) <= MAX_STEP_CONTENT_SIZE:
        return content
    return content[:MAX_STEP_CONTENT_SIZE] + "\n[Content truncated for performance]"

def format_agentic_response(steps, final_answer=None, partial=False, suggestions=None, repo_path=None, user_query=None):
    """Optimized response formatting with size limits and performance improvements."""
    
    # Optimize steps for frontend consumption
    optimized_steps = []
    for step in steps[:50]:  # Limit to 50 steps max for performance
        optimized_step = {
            "type": step.get("type", "unknown"),
            "step": step.get("step", 0)
        }
        
        # Truncate content for performance
        content = step.get("content", "")
        if isinstance(content, str):
            optimized_step["content"] = _truncate_content(content)
        elif isinstance(content, dict):
            # Summarize large dictionaries
            if len(str(content)) > MAX_STEP_CONTENT_SIZE:
                optimized_step["content"] = f"Large JSON object with {len(content)} keys"
                optimized_step["content_preview"] = str(content)[:200] + "..."
            else:
                optimized_step["content"] = content
        else:
            optimized_step["content"] = str(content)[:MAX_STEP_CONTENT_SIZE]
        
        # Include other important fields
        for field in ["tool_name", "tool_input", "tool_output_preview", "observed_tool_name"]:
            if field in step:
                optimized_step[field] = step[field]
        
        optimized_steps.append(optimized_step)
    
    # Basic response structure for backward compatibility
    basic_response = {
        "type": "final",
        "steps": optimized_steps,
        "final_answer": _truncate_content(final_answer) if isinstance(final_answer, str) else final_answer,
        "partial": partial,
        "suggestions": suggestions or [],
        "performance_info": {
            "total_steps": len(steps),
            "displayed_steps": len(optimized_steps),
            "truncated": len(steps) > 50
        }
    }
    
    # Try to apply professional formatting if formatter is available
    if ResponseFormatter and not partial and final_answer:
        try:
            # Initialize formatter with repo path if available
            if repo_path:
                formatter = ResponseFormatter(Path(repo_path))
                
                # Extract tool information from steps for better formatting
                tool_name = None
                tool_output = None
                
                # Find the last observation step which usually contains the main result
                for step in reversed(steps):
                    if step.get("type") == "observation" and step.get("observed_tool_name"):
                        tool_name = step.get("observed_tool_name")
                        tool_output = step.get("content")
                        break
                
                # Format the response professionally
                # Ensure we pass a string to the formatter
                content_to_format = tool_output if tool_output else final_answer
                if not isinstance(content_to_format, str):
                    if isinstance(content_to_format, dict):
                        content_to_format = json.dumps(content_to_format, indent=2)
                    else:
                        content_to_format = str(content_to_format)
                
                structured_response = formatter.format_response(
                    raw_content=content_to_format,
                    tool_name=tool_name,
                    query=user_query
                )
                
                # Add structured response to the basic response
                basic_response["structured_response"] = structured_response.to_dict()
                
                logger.info(f"Applied professional formatting for {structured_response.response_type.value} response")
                
        except Exception as e:
            logger.warning(f"Failed to apply professional formatting: {e}")
            # Continue with basic response if formatting fails
    
    return json.dumps(basic_response)

def clean_captured_output(captured_output: str) -> str:
    """Optimized output cleaning using precompiled patterns."""
    if not captured_output:
        return ""
    
    # Quick size check and truncation
    if len(captured_output) > MAX_RESPONSE_SIZE:
        captured_output = captured_output[:MAX_RESPONSE_SIZE]
    
    # Remove ANSI escape sequences using precompiled pattern
    cleaned_output = REACT_PATTERNS['ansi_escape'].sub('', captured_output)
    cleaned_output = REACT_PATTERNS['ansi_codes'].sub('', cleaned_output)

    # Split and filter lines efficiently
    lines = cleaned_output.split('\n')
    preserved_lines = []
    
    # Additional skip patterns for better filtering
    skip_patterns = [
        REACT_PATTERNS['log_patterns'],
        REACT_PATTERNS['http_request'],
        REACT_PATTERNS['running_step'],
        re.compile(r'^\s*⚡'),
        re.compile(r'^\s*Step \w+ produced event'),
        re.compile(r'^[\[\]0-9;m\s]*$')  # ANSI remnants
    ]

    for line in lines:
        # Skip empty lines and lines matching skip patterns
        if not line.strip() or any(pattern.search(line) for pattern in skip_patterns):
            continue
        preserved_lines.append(line)
    
    return '\n'.join(preserved_lines)

def extract_clean_answer(raw_response: str) -> str:
    """Optimized clean answer extraction."""
    if not raw_response:
        return ""
    
    try:
        # Quick answer extraction for performance
        if "Final Answer:" in raw_response:
            answer_parts = raw_response.split("Final Answer:", 1)
            if len(answer_parts) > 1:
                clean_answer = answer_parts[1].strip()
                return _truncate_content(clean_answer) if clean_answer else ""
        
        if "Answer:" in raw_response:
            answer_parts = raw_response.split("Answer:", 1)
            if len(answer_parts) > 1:
                clean_answer = answer_parts[1].strip()
                return _truncate_content(clean_answer) if clean_answer else ""
        
        # Fallback: filter out ReAct noise using compiled patterns
        lines = raw_response.split('\n')
        clean_lines = []
        
        # Optimized skip patterns
        skip_pattern = re.compile(r'(Thought:|Action:|Action Input:|Observation:|> Running step|INFO:|HTTP Request:|Step input:|Pandas [IO])')
        
        for line in lines:
            line = line.strip()
            if not line or skip_pattern.search(line) or line in ['{', '}']:
                continue
            clean_lines.append(line)
        
        if clean_lines:
            cleaned = '\n'.join(clean_lines)
            # Remove markdown code blocks efficiently
            cleaned = cleaned.replace('```', '')
            # Remove empty lines
            cleaned = '\n'.join(line for line in cleaned.split('\n') if line.strip())
            return _truncate_content(cleaned) if cleaned.strip() else ""
        
        return basic_cleanup(raw_response)
    except Exception as e:
        logger.error(f"Error extracting clean answer: {e}")
        return basic_cleanup(raw_response)

def basic_cleanup(text: str) -> str:
    """Basic cleanup of response text"""
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if (line and 
            not line.startswith('Thought:') and 
            not line.startswith('Action:') and 
            not line.startswith('Action Input:') and 
            not line.startswith('Observation:') and 
            not line.startswith('> Running step') and 
            not line.startswith('INFO:') and
            not line.startswith('HTTP Request:')):
            filtered_lines.append(line)
    result = '\n'.join(filtered_lines).strip()
    return result if result else "I analyzed your request but encountered some formatting issues. Please try asking in a different way."

def get_natural_exploration_suggestions(original_query: str) -> str:
    """Provide natural exploration suggestions instead of exposing iteration limits"""
    query_lower = original_query.lower()
    if "explore" in query_lower and "directory" in query_lower:
        return """## Codebase Exploration
I started exploring your codebase and can see it has an interesting structure! Let me help you discover it step by step.
**I can help you with:**
📂 **Directory Structure** - "What's in the src directory?"  
📄 **Key Files** - "Show me the main Python files"  
🏗️ **Architecture** - "How is this project organized?"  
🔧 **Specific Components** - "Explain the agents.py file"  
**What would you like to explore first?**"""
    elif "analyze" in query_lower:
        return """## Code Analysis Ready
I'm ready to analyze your codebase! I work best when you give me specific areas to focus on.
**Try asking me to:**
🔍 **Analyze specific files** - "What does main.py do?"  
📋 **Understand structure** - "How are the modules organized?"  
🔗 **Find relationships** - "What files are related to authentication?"  
💡 **Explain patterns** - "Show me the design patterns used"  
**What aspect of the code interests you most?**"""
    else:
        return """## Let's Explore Your Code Together
I'm here to help you understand your codebase! I work best with focused questions.
**Popular exploration patterns:**
🏠 **Project Overview** - "What is this project about?"  
📁 **Directory Exploration** - "What's in the [directory] folder?"  
📄 **File Analysis** - "Explain the [filename] file"  
🔍 **Find Functionality** - "Where is [feature] implemented?"  
**What would you like to discover?**"""

def get_natural_error_recovery(original_query: str, error_msg: str) -> str:
    """Provide natural error recovery without exposing technical details"""
    logger.error(f"Providing natural error recovery for: {error_msg}")
    return """## Let's Try a Different Approach
I had some trouble with that analysis. Let me help you explore your codebase with a more focused approach.
**Try these patterns:**
🎯 **Specific Questions** - "What files are in the src directory?"  
📖 **File Reading** - "Show me the contents of main.py"  
🔍 **Targeted Search** - "Find all Python files with 'agent' in the name"  
📂 **Step-by-step** - "First show me the project structure"  
**What specific part of your code would you like to explore?**"""
