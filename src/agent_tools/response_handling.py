# src/agent_tools/response_handling.py

import json
import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Import the professional response formatter
try:
    from ..response_formatter import ResponseFormatter, ResponseType
except ImportError:
    # Fallback if formatter not available
    ResponseFormatter = None
    ResponseType = None

def parse_react_steps(raw_response: str):
    """Parse ReAct steps from raw agent response into a structured format."""
    logger.info(f"[DEBUG] Parsing ReAct trace (length: {len(raw_response)}): {raw_response[:500]}")
    
    steps = []
    lines = raw_response.split('\n')
    i = 0
    step_counter = 0

    while i < len(lines):
        line_content_stripped = lines[i].strip() # For prefix checking
        original_line = lines[i] # Keep original for content extraction
        
        if not line_content_stripped: # Skip empty lines
            i += 1
            continue

        current_step_data = {"step": step_counter}

        if line_content_stripped.startswith("Thought:"):
            current_step_data["type"] = "thought"
            content_lines = [original_line.split("Thought:", 1)[1].strip()]
            i += 1
            while i < len(lines) and not re.match(r"^(Thought:|Action:|Action Input:|Observation:|Answer:|Final Answer:)", lines[i].strip()):
                content_lines.append(lines[i])
                i += 1
            current_step_data["content"] = "\n".join(content_lines).strip()
            steps.append(current_step_data)
            step_counter += 1
        
        elif line_content_stripped.startswith("Action:"):
            current_step_data["type"] = "action"
            current_step_data["tool_name"] = original_line.split("Action:", 1)[1].strip()
            current_step_data["content"] = f"Calling tool: {current_step_data['tool_name']}"
            i += 1
            if i < len(lines) and lines[i].strip().startswith("Action Input:"):
                input_json_lines = [lines[i].split("Action Input:", 1)[1].strip()]
                i += 1
                first_input_line_stripped = input_json_lines[0]
                is_likely_json = first_input_line_stripped.startswith('{') or first_input_line_stripped.startswith('[')
                if is_likely_json:
                    open_braces = first_input_line_stripped.count('{') + first_input_line_stripped.count('[')
                    close_braces = first_input_line_stripped.count('}') + first_input_line_stripped.count(']')
                    while i < len(lines) and not re.match(r"^(Thought:|Action:|Observation:|Answer:|Final Answer:)", lines[i].strip()):
                        current_input_line = lines[i]
                        input_json_lines.append(current_input_line)
                        open_braces += current_input_line.count('{') + current_input_line.count('[')
                        close_braces += current_input_line.count('}') + current_input_line.count(']')
                        i += 1
                        if open_braces > 0 and open_braces == close_braces:
                            break
                else:
                     while i < len(lines) and not re.match(r"^(Thought:|Action:|Observation:|Answer:|Final Answer:)", lines[i].strip()):
                        input_json_lines.append(lines[i])
                        i += 1
                action_input_str = "\n".join(input_json_lines).strip()
                try:
                    current_step_data["tool_input"] = json.loads(action_input_str)
                except json.JSONDecodeError:
                    logger.debug(f"Action Input for {current_step_data['tool_name']} not JSON: {action_input_str[:100]}")
                    current_step_data["tool_input"] = action_input_str 
            else:
                current_step_data["tool_input"] = None 
            steps.append(current_step_data)
            step_counter += 1

        elif line_content_stripped.startswith("Observation:"):
            current_step_data["type"] = "observation"
            if steps and steps[-1]["type"] == "action":
                current_step_data["observed_tool_name"] = steps[-1].get("tool_name", "unknown_tool")
            obs_content_lines = [original_line.split("Observation:", 1)[1].strip()]
            i += 1
            first_obs_line_stripped = obs_content_lines[0]
            is_likely_json_obs = first_obs_line_stripped.startswith('{') or first_obs_line_stripped.startswith('[')
            if is_likely_json_obs:
                open_braces_obs = first_obs_line_stripped.count('{') + first_obs_line_stripped.count('[')
                close_braces_obs = first_obs_line_stripped.count('}') + first_obs_line_stripped.count(']')
                while i < len(lines) and not re.match(r"^(Thought:|Action:|Answer:|Final Answer:|Observation:)", lines[i].strip()):
                    current_obs_line = lines[i]
                    obs_content_lines.append(current_obs_line)
                    open_braces_obs += current_obs_line.count('{') + current_obs_line.count('[')
                    close_braces_obs += current_obs_line.count('}') + current_obs_line.count(']')
                    i += 1
                    if open_braces_obs > 0 and open_braces_obs == close_braces_obs:
                        break
            else: 
                while i < len(lines) and not re.match(r"^(Thought:|Action:|Answer:|Final Answer:|Observation:)", lines[i].strip()):
                    obs_content_lines.append(lines[i])
                    i += 1
            observation_str = "\n".join(obs_content_lines).strip()
            try:
                parsed_observation = json.loads(observation_str)
                current_step_data["content"] = parsed_observation
                if isinstance(parsed_observation, dict):
                    current_step_data["tool_output_preview"] = f"JSON data (Keys: {list(parsed_observation.keys())[:3]}...)"
                elif isinstance(parsed_observation, list):
                    current_step_data["tool_output_preview"] = f"JSON array (Length: {len(parsed_observation)}, First item: {str(parsed_observation[0])[:50]}...)" if parsed_observation else "Empty JSON array"
                else:
                    current_step_data["tool_output_preview"] = observation_str[:200] + "..." if len(observation_str) > 200 else observation_str
            except json.JSONDecodeError:
                current_step_data["content"] = observation_str
                current_step_data["tool_output_preview"] = observation_str[:200] + "..." if len(observation_str) > 200 else observation_str
            steps.append(current_step_data)
            step_counter += 1

        elif line_content_stripped.startswith("Answer:") or line_content_stripped.startswith("Final Answer:"):
            current_step_data["type"] = "answer"
            if line_content_stripped.startswith("Final Answer:"):
                content_lines = [original_line.split("Final Answer:", 1)[1].strip()]
            else:
                content_lines = [original_line.split("Answer:", 1)[1].strip()]
            i += 1
            while i < len(lines) and not re.match(r"^(Thought:|Action:|Observation:)", lines[i].strip()):
                content_lines.append(lines[i])
                i += 1
            current_step_data["content"] = "\n".join(content_lines).strip()
            steps.append(current_step_data)
            step_counter += 1
        else:
            if steps and steps[-1]["type"] in ["thought", "answer"] and isinstance(steps[-1]["content"], str):
                logger.debug(f"Appending to previous {steps[-1]['type']}: {original_line.strip()}")
                steps[-1]["content"] += "\n" + original_line
                steps[-1]["content"] = steps[-1]["content"].strip()
            else:
                logger.debug(f"Skipping unexpected line: {original_line.strip()}")
            i += 1

    final_answer_obj = next((step for step in reversed(steps) if step["type"] == "answer"), None)
    final_answer_content = final_answer_obj["content"] if final_answer_obj else None

    if not final_answer_content and not steps and raw_response.strip():
        if not any(pattern.search(raw_response) for pattern in [
            re.compile(r'^\s*(DEBUG|INFO|WARNING|ERROR)'), 
            re.compile(r'HTTP Request:'),
            re.compile(r'> Running step')
        ]):
            logger.info(f"[DEBUG] No ReAct steps, using raw_response as final answer.")
            final_answer_content = raw_response.strip()
            steps.append({"type": "answer", "content": final_answer_content, "step": 0})

    for step in steps:
        if 'content' not in step:
            step['content'] = f"Step {step.get('step', 0)}: {step.get('type', 'unknown')} executed"
        if 'type' not in step:
            step['type'] = 'unknown'
        if 'step' not in step:
            step['step'] = 0
    
    logger.info(f"[DEBUG] Parsed {len(steps)} steps. Final answer derived: {bool(final_answer_content)}")
    return steps, final_answer_content

def format_agentic_response(steps, final_answer=None, partial=False, suggestions=None, repo_path=None, user_query=None):
    """Format agentic output as structured JSON for the frontend UI with professional formatting."""
    
    # Basic response structure for backward compatibility
    basic_response = {
        "type": "final",
        "steps": steps,
        "final_answer": final_answer,
        "partial": partial,
        "suggestions": suggestions or []
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
                structured_response = formatter.format_response(
                    raw_content=tool_output if tool_output else final_answer,
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
    """Clean captured output to remove logging noise but preserve ReAct trace."""
    if not captured_output:
        return ""
    
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned_output = ansi_escape.sub('', captured_output)
    
    cleaned_output = re.sub(r'\033\[[0-9;]*m', '', cleaned_output)
    cleaned_output = re.sub(r'\[0m', '', cleaned_output) 
    cleaned_output = re.sub(r'\[1;3;[0-9]+m', '', cleaned_output)

    lines = cleaned_output.split('\n')
    preserved_lines = []
    
    skip_log_patterns = [
        re.compile(r'^\s*DEBUG:'),
        re.compile(r'^\s*INFO:'),
        re.compile(r'^\s*WARNING:'),
        re.compile(r'^\s*ERROR:'),
        re.compile(r'^\s*INFO:httpx:HTTP Request:'),
        re.compile(r'^\s*⚡'), 
        re.compile(r'^\s*> Running step'), 
        re.compile(r'^\s*Step \w+ produced event'), 
    ]

    for line in lines:
        if any(pattern.search(line) for pattern in skip_log_patterns):
            continue
        if re.match(r'^[\[\]0-9;m\s]*$', line.strip()): 
            continue
        preserved_lines.append(line)
    
    return '\n'.join(preserved_lines)

def extract_clean_answer(raw_response: str) -> str:
    """Extract clean final answer from ReAct agent response"""
    try:
        if "Answer:" in raw_response:
            answer_parts = raw_response.split("Answer:")
            if len(answer_parts) > 1:
                clean_answer = answer_parts[-1].strip()
                if clean_answer:
                    return clean_answer
        
        lines = raw_response.split('\n')
        clean_lines = []
        skip_patterns = [
            'Thought:', 'Action:', 'Action Input:', 'Observation:', '> Running step',
            'INFO:', 'HTTP Request:', 'Step input:', '{', '}', 
            'Pandas Instructions:', 'Pandas Output:'
        ]
        for line in lines:
            line = line.strip()
            if not line: continue
            if not any(pattern in line for pattern in skip_patterns):
                clean_lines.append(line)
        
        if clean_lines:
            cleaned = '\n'.join(clean_lines)
            cleaned = cleaned.replace('```', '')
            cleaned = '\n'.join([line for line in cleaned.split('\n') if line.strip()])
            if cleaned.strip():
                return cleaned.strip()
        
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
