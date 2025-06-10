# src/agent_tools/code_generation.py

import json
import logging
from pathlib import Path
import re
from typing import List, Dict, Any, Optional, Annotated

# Assuming settings and LLMClient might be needed
try:
    from ..config import settings
    from ..llm_client import LLMClient # For write_complete_code
except ImportError:
    class MockSettings:
        pass # Add necessary mock attributes if testing in isolation
    settings = MockSettings()
    class LLMClient: # Mock LLMClient
        def _get_openrouter_llm(self, model_name: str):
            # Return a mock LLM that has a complete method
            class MockLLM:
                def complete(self, prompt: str):
                    class MockResponse:
                        text = f"// Mock LLM response for: {prompt[:50]}..."
                    return MockResponse()
            return MockLLM()

# Utilities that might be used by these methods, ensure they are available
# For now, assuming they are passed or self-contained if simple enough
# from .utilities import some_utility_if_needed 

logger = logging.getLogger(__name__)

class CodeGenerationOperations:
    def __init__(self, repo_path: Path, llm_instance: Optional[Any] = None): # llm_instance for write_complete_code
        self.repo_path = repo_path
        self.llm = llm_instance # This will be the agent's main LLM or a specific one for code gen

    def generate_code_example(
        self,
        description: Annotated[str, "Description of what kind of code example to generate"],
        context_files: Annotated[Optional[List[str]], "List of relevant files to base the example on"] = None
    ) -> str:
        """Generate practical, runnable code examples based on codebase analysis"""
        try:
            logger.info(f"[DEBUG] Generating code for: {description}")
            if not context_files:
                return json.dumps({
                    "message": "No context files provided.",
                    "suggestion": "Use @filename to specify files to analyze.",
                    "example": f"Try: 'Using @agents.py, show me how to build: {description}'"
                }, indent=2)
            
            primary_language = self._detect_primary_language_from_context(context_files)
            analysis = self._analyze_repository_context(context_files)
            analysis["detected_language"] = primary_language
            return self._create_language_appropriate_example(description, analysis, context_files, primary_language)
        except Exception as e:
            logger.error(f"Error generating code example: {e}")
            return json.dumps({
                "error": f"Failed to generate code: {str(e)}",
                "description": description,
                "suggestion": "Try providing specific files to analyze."
            }, indent=2)

    def _detect_primary_language_from_context(self, context_files: List[str]) -> str:
        language_map = {
            '.py': 'python', '.js': 'javascript', '.jsx': 'javascript', 
            '.ts': 'typescript', '.tsx': 'typescript', '.java': 'java',
            '.kt': 'kotlin', '.scala': 'scala', '.go': 'go', '.rs': 'rust',
            '.rb': 'ruby', '.php': 'php', '.cs': 'csharp', '.cpp': 'cpp',
            '.cc': 'cpp', '.cxx': 'cpp', '.c': 'c', '.swift': 'swift',
            '.m': 'objective-c', '.ex': 'elixir', '.exs': 'elixir',
            '.ml': 'ocaml', '.hs': 'haskell', '.clj': 'clojure',
            '.dart': 'dart', '.lua': 'lua', '.r': 'r', '.jl': 'julia'
        }
        language_counts = {}
        for file_path in context_files:
            ext = Path(file_path).suffix.lower()
            if ext in language_map:
                lang = language_map[ext]
                language_counts[lang] = language_counts.get(lang, 0) + 1
        
        if language_counts:
            return max(language_counts, key=language_counts.get)
        
        try: # Fallback to broader repository check
            repo_language_info = self._detect_repository_languages() # Assumes this method is also moved or accessible
            return repo_language_info.get("primary_language", "generic")
        except: return "generic"

    def _create_language_appropriate_example(self, description: str, analysis: Dict, context_files: List[str], primary_language: str) -> str:
        relevant_class = self._find_most_relevant_class(description, analysis)
        explanation = self._generate_language_explanation(description, analysis, relevant_class, context_files, primary_language)
        code_example = self._generate_language_specific_example(description, analysis, relevant_class, primary_language)
        return json.dumps({
            "detected_language": primary_language,
            "analysis": f"Analyzed {len(context_files)} files, found {len(analysis.get('classes',[]))} classes.",
            "available_patterns": {
                "classes": [cls["name"] for cls in analysis.get("classes", [])],
                "base_classes": analysis.get("base_classes", []),
                "capabilities": analysis.get("capabilities", [])[:10]
            },
            "recommendation": explanation,
            "code_example": code_example,
            "implementation_guide": self._get_language_implementation_steps(relevant_class, analysis, primary_language)
        }, indent=2)

    def _find_most_relevant_class(self, description: str, analysis: Dict) -> Optional[Dict]:
        user_intent = description.lower()
        for cls in analysis.get("classes", []):
            if any(keyword in cls["name"].lower() or keyword in cls.get("purpose", "").lower() 
                   for keyword in user_intent.split()):
                return cls
        return analysis.get("classes", [])[0] if analysis.get("classes") else None

    def _generate_language_explanation(self, description: str, analysis: Dict, relevant_class: Optional[Dict], context_files: List[str], language: str) -> str:
        explanation = f"## How to build: {description}\n\n"
        explanation += f"**Detected language:** {language.title()}\n"
        explanation += f"**Based on analysis of:** {', '.join(context_files)}\n\n"
        if relevant_class:
            explanation += f"**Recommended approach:** Use `{relevant_class['name']}` as a foundation.\n"
            explanation += f"- **Purpose:** {relevant_class.get('purpose', 'N/A')}\n"
            explanation += f"- **Base class:** {relevant_class.get('base', 'N/A')}\n"
            explanation += f"- **File:** {relevant_class.get('file', 'N/A')}\n\n"
        if analysis.get("base_classes"):
            explanation += f"**Available base classes:** {', '.join(analysis['base_classes'][:3])}\n\n"
        if analysis.get("capabilities"):
            explanation += f"**Repository capabilities include:** {', '.join(analysis['capabilities'][:8])}"
        return explanation

    def _generate_language_specific_example(self, description: str, analysis: Dict, relevant_class: Optional[Dict], language: str) -> str:
        imports = analysis.get("imports", [])[:3]
        base_class_name = "BaseClass"
        if relevant_class and relevant_class.get("base"):
            base_class_name = relevant_class["base"]
        elif analysis.get("base_classes"):
             base_class_name = analysis["base_classes"][0]


        words = description.split()
        class_name = "My" + "".join(word.capitalize() for word in words[:2] if word.isalpha())
        
        # Ensure relevant_class is not None before accessing its items
        relevant_class_name_for_template = relevant_class['name'] if relevant_class else 'repository analysis'

        lang_example_map = {
            "python": self._generate_python_example, "javascript": self._generate_javascript_example,
            "typescript": self._generate_typescript_example, "java": self._generate_java_example,
            "go": self._generate_go_example, "rust": self._generate_rust_example,
            "csharp": self._generate_csharp_example, "ruby": self._generate_ruby_example,
            "php": self._generate_php_example
        }
        
        generator_func = lang_example_map.get(language)
        if generator_func:
            if language == "python":
                 return generator_func(description, class_name, base_class_name, relevant_class_name_for_template, imports)
            elif language in ["go", "rust"]: # Go and Rust don't use base_class in the same way
                 return generator_func(description, class_name, relevant_class_name_for_template)
            return generator_func(description, class_name, base_class_name, relevant_class_name_for_template)
        return self._generate_generic_example(description, class_name, base_class_name, relevant_class_name_for_template, language)

    def _generate_python_example(self, description: str, class_name: str, base_class: str, relevant_class_name: str, imports: List[str]) -> str:
        return f"""# {description} - Python implementation
# Based on: {relevant_class_name}
{chr(10).join(imports)}

class {class_name}({base_class}):
    def __init__(self):
        super().__init__()
    async def process_request(self, input_data):
        return await self.process_data(input_data)
    async def process_data(self, data):
        return f"Processed: {{data}}"
if __name__ == "__main__":
    import asyncio
    async def main():
        processor = {class_name}()
        result = await processor.process_request("input")
        print(result)
    asyncio.run(main())"""

    def _generate_javascript_example(self, description: str, class_name: str, base_class: str, relevant_class_name: str) -> str:
        return f"""// {description} - JS
// Based on: {relevant_class_name}
class {class_name} extends {base_class} {{
    constructor() {{ super(); }}
    async processRequest(inputData) {{ return await this.processData(inputData); }}
    async processData(data) {{ return `Processed: ${{data}}`; }}
}}
module.exports = {class_name};"""
    
    def _generate_typescript_example(self, description: str, class_name: str, base_class: str, relevant_class_name: str) -> str:
        return f"""// {description} - TS
// Based on: {relevant_class_name}
interface ProcInput {{ data: string; }}
interface ProcResult {{ success: boolean; result: string; }}
class {class_name} extends {base_class} {{
    constructor() {{ super(); }}
    async processRequest(input: ProcInput): Promise<ProcResult> {{
        const result = await this.processData(input.data);
        return {{ success: true, result }};
    }}
    private async processData(data: string): Promise<string> {{ return `Processed: ${{data}}`; }}
}}
export default {class_name};"""

    def _generate_java_example(self, description: str, class_name: str, base_class: str, relevant_class_name: str) -> str:
        return f"""// {description} - Java
// Based on: {relevant_class_name}
import java.util.concurrent.CompletableFuture;
public class {class_name} extends {base_class} {{
    public {class_name}() {{ super(); }}
    public CompletableFuture<String> processRequest(String inputData) {{
        return CompletableFuture.supplyAsync(() -> processData(inputData));
    }}
    private String processData(String data) {{ return "Processed: " + data; }}
}}"""

    def _generate_go_example(self, description: str, class_name: str, relevant_class_name: str) -> str:
        return f"""// {description} - Go
// Based on: {relevant_class_name}
package main
import "fmt"
type {class_name} struct{{}}
func New{class_name}() *{class_name} {{ return &{class_name}{{}} }}
func (p *{class_name}) ProcessRequest(inputData string) (string, error) {{ return p.processData(inputData) }}
func (p *{class_name}) processData(data string) (string, error) {{ return fmt.Sprintf("Processed: %s", data), nil }}"""

    def _generate_rust_example(self, description: str, class_name: str, relevant_class_name: str) -> str:
        return f"""// {description} - Rust
// Based on: {relevant_class_name}
pub struct {class_name};
impl {class_name} {{
    pub fn new() -> Self {{ Self }}
    pub async fn process_request(&self, input_data: &str) -> Result<String, String> {{
        self.process_data(input_data).await
    }}
    async fn process_data(&self, data: &str) -> Result<String, String> {{ Ok(format!("Processed: {{}}", data)) }}
}}"""

    def _generate_csharp_example(self, description: str, class_name: str, base_class: str, relevant_class_name: str) -> str:
        return f"""// {description} - C#
// Based on: {relevant_class_name}
using System.Threading.Tasks;
public class {class_name} : {base_class} {{
    public {class_name}() : base() {{}}
    public async Task<string> ProcessRequestAsync(string inputData) {{ return await ProcessDataAsync(inputData); }}
    private async Task<string> ProcessDataAsync(string data) {{ await Task.Delay(10); return $"Processed: {{data}}"; }}
}}"""

    def _generate_ruby_example(self, description: str, class_name: str, base_class: str, relevant_class_name: str) -> str:
        return f"""# {description} - Ruby
# Based on: {relevant_class_name}
class {class_name} < {base_class}
  def initialize; super; end
  def process_request(input_data); process_data(input_data); end
  private
  def process_data(data); "Processed: #{{data}}"; end
end"""

    def _generate_php_example(self, description: str, class_name: str, base_class: str, relevant_class_name: str) -> str:
        return f"""<?php // {description} - PHP
// Based on: {relevant_class_name}
class {class_name} extends {base_class} {{
    public function __construct() {{ parent::__construct(); }}
    public function processRequest($inputData) {{ return $this->processData($inputData); }}
    private function processData($data) {{ return "Processed: " . $data; }}
}} ?>"""

    def _generate_generic_example(self, description: str, class_name: str, base_class: str, relevant_class_name: str, language: str) -> str:
        return f"""// {description} - {language} (Pseudocode)
// Based on: {relevant_class_name}
class {class_name} extends {base_class} {{
    constructor() {{ /* init */ }}
    processRequest(inputData) {{ return this.processData(inputData); }}
    processData(data) {{ return "Processed: " + data; }}
}}"""

    def _get_language_implementation_steps(self, relevant_class: Optional[Dict], analysis: Dict, language: str) -> List[str]:
        steps = [f"1. Study existing {language} patterns.", "2. Implement core logic.", "3. Add tests."]
        if relevant_class and analysis.get("capabilities"):
            steps.append(f"4. Leverage: {', '.join(analysis['capabilities'][:2])}")
        return steps

    def _detect_repository_languages(self) -> Dict[str, Any]: # Added self
        # This is a simplified version. A real one would scan more files.
        language_map = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript'} # etc.
        counts = {}
        for item in self.repo_path.rglob("*"): # Added self
            if item.is_file() and item.suffix in language_map:
                lang = language_map[item.suffix]
                counts[lang] = counts.get(lang,0) + 1
        if not counts: return {"primary_language": "generic", "languages": {}}
        primary = max(counts, key=counts.get)
        return {"primary_language": primary, "languages": {k: round(v/sum(counts.values())*100,1) for k,v in counts.items()}}

    def _analyze_repository_context(self, context_files: List[str]) -> Dict[str, Any]: # Added self
        analysis = {"classes": [], "base_classes": [], "imports": [], "capabilities": [], "functions": []}
        for file_path_str in context_files:
            full_path = self.repo_path / file_path_str # Added self
            if not full_path.exists(): continue
            try:
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                lang = self._get_language_from_extension(full_path.suffix) # Added self
                
                # Simplified extraction logic for brevity
                if lang == "python": self._extract_python_patterns(content, file_path_str, analysis)
                # Add other languages if needed
                
                # Generic capabilities
                if "agent" in content.lower(): analysis["capabilities"].append("agent-creation")
                if "tool" in content.lower(): analysis["capabilities"].append("tool-usage")

            except Exception as e: logger.error(f"Error analyzing {file_path_str}: {e}")
        analysis["base_classes"] = list(set(analysis["base_classes"]))
        analysis["capabilities"] = list(set(analysis["capabilities"]))
        return analysis

    def _get_language_from_extension(self, ext: str) -> str: # Added self
        lang_map = {'.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.java': 'java'} # etc.
        return lang_map.get(ext.lower(), 'generic')

    def _extract_python_patterns(self, content: str, file_path: str, analysis: Dict): # Added self
        for line in content.split('\n'):
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                analysis["imports"].append(line.strip())
            if " class " in line and line.strip().startswith("class "):
                match = re.search(r"class\s+(\w+)(?:\(([\w,\s\.]+)\))?:", line)
                if match:
                    name = match.group(1)
                    base = match.group(2) if match.group(2) else ""
                    analysis["classes"].append({"name": name, "base": base, "file": file_path, "purpose": "Python class"})
                    if base: analysis["base_classes"].append(base)
            if " def " in line and line.strip().startswith("def "):
                 match = re.search(r"def\s+(\w+)\s*\(", line)
                 if match: analysis["functions"].append(match.group(1))


    def _extract_class_purpose(self, content: str, class_name: str) -> str: # Added self
        # Simplified
        return f"Purpose of {class_name} (extracted)"
        
    def _extract_struct_purpose(self, content: str, struct_name: str) -> str: # Added self
        return f"Purpose of {struct_name} (extracted)"

    def _extract_file_capabilities(self, content: str) -> List[str]: # Added self
        # Simplified
        caps = []
        if "api" in content.lower(): caps.append("api-interaction")
        if "database" in content.lower(): caps.append("database-ops")
        return caps

    # --- write_complete_code and its helpers ---
    def write_complete_code(
        self,
        description: Annotated[str, "Detailed description of what code to write"],
        context_files: Annotated[Optional[List[str]], "List of reference files to base the code on"] = None,
        language: Annotated[Optional[str], "Programming language (auto-detected if not specified)"] = None,
        output_format: Annotated[str, "Output format: 'markdown' or 'raw'"] = "markdown"
    ) -> str:
        try:
            logger.info(f"[DEBUG] Writing complete code for: {description}")
            if not language and context_files:
                language = self._detect_primary_language_from_context(context_files)
            elif not language: language = "python" 

            context_content = ""
            patterns = {}
            if context_files:
                for file_path_str in context_files:
                    try:
                        full_path = self.repo_path / file_path_str
                        if full_path.exists():
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                context_content += f"\n\n=== {file_path_str} ===\n{content}"
                    except Exception as e: logger.warning(f"Could not read context file {file_path_str}: {e}")
                analysis = self._analyze_repository_context(context_files) # Added self
                patterns = {
                    "classes": [cls["name"] for cls in analysis.get("classes", [])],
                    "imports": analysis.get("imports", []),
                    "base_classes": analysis.get("base_classes", []),
                    "functions": analysis.get("functions", [])[:10]
                }
            
            prompt = self._build_complete_code_prompt(description, language, context_content, patterns, output_format)
            
            try:
                # Use the LLM instance passed to CodeGenerationOperations or a default one
                llm_to_use = self.llm
                if not llm_to_use: # Fallback if not provided via constructor
                    llm_client = LLMClient() # Assumes LLMClient can be instantiated
                    llm_to_use = llm_client._get_openrouter_llm(settings.default_model or "google/gemini-2.5-flash-preview-05-20")

                response = llm_to_use.complete(prompt)
                code_response = response.text.strip()
                return self._format_complete_code_response(code_response, description, language, output_format)
            except Exception as e:
                logger.error(f"Error generating code with LLM: {e}")
                return self._generate_error_response(description, language, str(e))
        except Exception as e:
            logger.error(f"Error in write_complete_code: {e}")
            return f"Error writing code: {str(e)}"

    def _build_complete_code_prompt(self, description: str, language: str, context_content: str, patterns: Dict, output_format: str) -> str:
        prompt = f"You are an expert {language} developer. Write COMPLETE, PRODUCTION-READY code...\nREQUIREMENTS:\n{description}\nLANGUAGE: {language}\n" # Truncated for brevity
        if context_content: prompt += f"REFERENCE CODE CONTEXT:\n{context_content[:8000]}\n"
        if patterns.get("classes"): prompt += f"AVAILABLE PATTERNS: Classes: {patterns['classes'][:3]}\n"
        prompt += self._get_language_specific_guidance(language) # Added self
        if output_format == "markdown": prompt += "Provide code in markdown blocks...\n"
        else: prompt += "Provide raw code...\n"
        prompt += "CRITICAL: Ensure code is COMPLETE and FUNCTIONAL."
        return prompt

    def _get_language_specific_guidance(self, language: str) -> str: # Added self
        # Simplified map
        guidance = {"python": "Follow PEP 8.", "javascript": "Use ES6+."}
        return guidance.get(language, "Follow general best practices.")

    def _format_complete_code_response(self, code_response: str, description: str, language: str, output_format: str) -> str: # Added self
        if output_format == "raw": return code_response
        
        def ensure_lang(match):
            return f"{match.group(1)}{language}\n{match.group(3)}{match.group(4)}" if not match.group(2).strip() else match.group(0)

        pattern = re.compile(r"(```)(.*?)\n(.*?)(```)", re.DOTALL)
        if pattern.search(code_response):
            processed = pattern.sub(ensure_lang, code_response)
        else:
            processed = f"# Code for: {description}\n\n```{language}\n{code_response}\n```"
        if not processed.lstrip().startswith("#"):
             processed = f"# Generated Code: {description}\n\n{processed}"
        return processed

    def _generate_error_response(self, description: str, language: str, error: str) -> str: # Added self
        return f"# Error: Could not generate code\n**Desc:** {description}\n**Lang:** {language}\n**Err:** {error}"
