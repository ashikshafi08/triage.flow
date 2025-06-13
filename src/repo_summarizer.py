"""
Repository Summarizer for Pre-computed Context
Generates and caches repository-level summaries to reduce LLM calls
"""
from typing import Dict, Any, List, Optional
import os
import asyncio
from pathlib import Path
import json
from .language_config import get_language_metadata
from .cache import folder_cache
from .config import settings

class RepositorySummarizer:
    """Generate and cache repository summaries for faster context retrieval"""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.summary_cache = {}
    
    async def generate_repo_overview(self) -> Dict[str, Any]:
        """Generate a comprehensive overview of the repository structure"""
        overview = {
            "total_files": 0,
            "total_directories": 0,
            "languages": {},
            "file_types": {},
            "key_directories": {},
            "architecture_hints": [],
            "dependencies": {},
            "entry_points": []
        }
        
        # Walk through repository
        for root, dirs, files in os.walk(self.repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            rel_root = os.path.relpath(root, self.repo_path)
            if rel_root == '.':
                rel_root = ''
            
            overview["total_directories"] += len(dirs)
            
            # Analyze files in current directory
            for file in files:
                if file.startswith('.'):
                    continue
                
                overview["total_files"] += 1
                file_path = os.path.join(root, file)
                metadata = get_language_metadata(file_path)
                
                # Track languages
                lang = metadata["language"]
                if lang != "unknown":
                    overview["languages"][lang] = overview["languages"].get(lang, 0) + 1
                
                # Track file types
                ext = os.path.splitext(file)[1].lower()
                if ext:
                    overview["file_types"][ext] = overview["file_types"].get(ext, 0) + 1
                
                # Identify key files
                await self._analyze_key_file(file, file_path, rel_root, overview)
        
        # Identify architecture patterns
        overview["architecture_hints"] = self._detect_architecture_patterns(overview)
        
        # Cache the overview
        cache_key = f"repo_overview_{self.repo_path}"
        await folder_cache.set(cache_key, overview, settings.CACHE_TTL_FOLDER)
        
        return overview
    
    async def _analyze_key_file(self, filename: str, filepath: str, rel_dir: str, overview: Dict[str, Any]):
        """Analyze key files for architectural insights"""
        # Configuration files
        if filename in ['package.json', 'requirements.txt', 'setup.py', 'Cargo.toml', 'go.mod']:
            overview["dependencies"][filename] = rel_dir
            
            # Extract dependencies if possible
            if filename == 'package.json':
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        deps = list(data.get('dependencies', {}).keys())[:10]  # Top 10
                        overview["dependencies"]['npm'] = deps
                except:
                    pass
            elif filename == 'requirements.txt':
                try:
                    with open(filepath, 'r') as f:
                        deps = [line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')][:10]
                        overview["dependencies"]['pip'] = deps
                except:
                    pass
        
        # Entry points
        if filename in ['main.py', 'app.py', 'server.py', 'index.js', 'index.ts', 'main.go', 'main.rs']:
            overview["entry_points"].append(os.path.join(rel_dir, filename))
        
        # Key directories
        if rel_dir and filename in ['__init__.py', 'index.js', 'index.ts']:
            parent_dir = rel_dir.split(os.sep)[0] if os.sep in rel_dir else rel_dir
            if parent_dir not in overview["key_directories"]:
                overview["key_directories"][parent_dir] = {
                    "type": self._infer_directory_type(parent_dir),
                    "files": 0
                }
            overview["key_directories"][parent_dir]["files"] += 1
    
    def _infer_directory_type(self, dirname: str) -> str:
        """Infer the purpose of a directory from its name"""
        dirname_lower = dirname.lower()
        
        if dirname_lower in ['src', 'lib', 'core']:
            return "source"
        elif dirname_lower in ['test', 'tests', 'spec', 'specs', '__tests__']:
            return "tests"
        elif dirname_lower in ['docs', 'documentation']:
            return "documentation"
        elif dirname_lower in ['api', 'routes', 'endpoints', 'controllers']:
            return "api"
        elif dirname_lower in ['models', 'schemas', 'entities']:
            return "data_models"
        elif dirname_lower in ['utils', 'helpers', 'common', 'shared']:
            return "utilities"
        elif dirname_lower in ['components', 'widgets', 'ui']:
            return "ui_components"
        elif dirname_lower in ['config', 'settings', 'conf']:
            return "configuration"
        elif dirname_lower in ['static', 'assets', 'public']:
            return "static_assets"
        else:
            return "other"
    
    def _detect_architecture_patterns(self, overview: Dict[str, Any]) -> List[str]:
        """Detect common architectural patterns from repository structure"""
        patterns = []
        
        # Web framework detection
        if 'package.json' in overview["dependencies"]:
            if 'npm' in overview["dependencies"]:
                deps = overview["dependencies"]['npm']
                if 'react' in deps or 'vue' in deps or 'angular' in deps:
                    patterns.append("Frontend SPA (Single Page Application)")
                if 'express' in deps or 'fastify' in deps:
                    patterns.append("Node.js Backend Server")
        
        if 'requirements.txt' in overview["dependencies"]:
            if 'pip' in overview["dependencies"]:
                deps = overview["dependencies"]['pip']
                if 'django' in deps:
                    patterns.append("Django Web Application")
                elif 'flask' in deps or 'fastapi' in deps:
                    patterns.append("Python Web API")
                if 'tensorflow' in deps or 'torch' in deps:
                    patterns.append("Machine Learning Project")
        
        # Directory structure patterns
        dirs = overview["key_directories"]
        if 'api' in dirs or 'routes' in dirs:
            patterns.append("RESTful API Structure")
        if 'components' in dirs and ('react' in overview.get("dependencies", {}).get('npm', [])):
            patterns.append("Component-Based Frontend")
        if 'models' in dirs and 'controllers' in dirs:
            patterns.append("MVC Architecture")
        
        return patterns
    
    async def generate_file_summary(self, file_path: str) -> Dict[str, Any]:
        """Generate a summary for a specific file"""
        cache_key = f"file_summary_{file_path}"
        
        # Try cache first
        cached = await folder_cache.get(cache_key)
        if cached:
            return cached
        
        summary = {
            "path": file_path,
            "language": "unknown",
            "purpose": "unknown",
            "key_elements": [],
            "imports": [],
            "exports": []
        }
        
        try:
            metadata = get_language_metadata(os.path.join(self.repo_path, file_path))
            summary["language"] = metadata["display_name"]
            
            # Read file content
            with open(os.path.join(self.repo_path, file_path), 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Language-specific analysis
            if metadata["language"] == "python":
                summary["key_elements"] = self._extract_python_elements(content)
                summary["imports"] = self._extract_python_imports(content)
            elif metadata["language"] in ["javascript", "typescript"]:
                summary["key_elements"] = self._extract_js_elements(content)
                summary["imports"] = self._extract_js_imports(content)
            
            # Infer purpose from filename and content
            summary["purpose"] = self._infer_file_purpose(file_path, content)
            
        except Exception as e:
            summary["error"] = str(e)
        
        # Cache the summary
        await folder_cache.set(cache_key, summary, settings.CACHE_TTL_FOLDER)
        
        return summary
    
    def _extract_python_elements(self, content: str) -> List[str]:
        """Extract key elements from Python code"""
        elements = []
        
        # Classes
        import re
        classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
        elements.extend([f"class:{cls}" for cls in classes[:5]])
        
        # Functions
        functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
        elements.extend([f"function:{func}" for func in functions[:5]])
        
        return elements
    
    def _extract_python_imports(self, content: str) -> List[str]:
        """Extract imports from Python code"""
        import re
        imports = []
        
        # Standard imports
        imports.extend(re.findall(r'^import\s+(\S+)', content, re.MULTILINE))
        
        # From imports
        from_imports = re.findall(r'^from\s+(\S+)\s+import', content, re.MULTILINE)
        imports.extend(from_imports)
        
        return list(set(imports))[:10]  # Unique, top 10
    
    def _extract_js_elements(self, content: str) -> List[str]:
        """Extract key elements from JavaScript/TypeScript code"""
        elements = []
        
        import re
        # Classes
        classes = re.findall(r'class\s+(\w+)', content)
        elements.extend([f"class:{cls}" for cls in classes[:5]])
        
        # Functions
        functions = re.findall(r'function\s+(\w+)', content)
        elements.extend([f"function:{func}" for func in functions[:5]])
        
        # Const functions
        const_funcs = re.findall(r'const\s+(\w+)\s*=\s*(?:async\s*)?\(', content)
        elements.extend([f"const:{func}" for func in const_funcs[:5]])
        
        return elements
    
    def _extract_js_imports(self, content: str) -> List[str]:
        """Extract imports from JavaScript/TypeScript code"""
        import re
        imports = []
        
        # ES6 imports
        imports.extend(re.findall(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", content))
        
        # Require statements
        imports.extend(re.findall(r"require\s*\(['\"]([^'\"]+)['\"]\)", content))
        
        return list(set(imports))[:10]
    
    def _infer_file_purpose(self, file_path: str, content: str) -> str:
        """Infer the purpose of a file from its path and content"""
        filename = os.path.basename(file_path).lower()
        
        # Common patterns
        if 'test' in filename or 'spec' in filename:
            return "test_file"
        elif filename.startswith('index.'):
            return "entry_point"
        elif filename in ['setup.py', 'package.json', 'requirements.txt']:
            return "configuration"
        elif 'model' in filename:
            return "data_model"
        elif 'route' in filename or 'controller' in filename:
            return "api_endpoint"
        elif 'util' in filename or 'helper' in filename:
            return "utility"
        elif 'component' in filename and ('.jsx' in filename or '.tsx' in filename):
            return "ui_component"
        else:
            # Check content patterns
            if 'def test_' in content or 'describe(' in content:
                return "test_file"
            elif 'class' in content[:1000]:
                return "class_definition"
            else:
                return "general" 