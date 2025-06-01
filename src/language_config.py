"""Language-specific configurations for the RAG system."""

from typing import Dict, Any
import os

# Language-specific file extensions and metadata
LANGUAGE_CONFIG: Dict[str, Dict[str, Any]] = {
    "python": {
        "extensions": [".py", ".pyw", ".pyi"],
        "doc_pattern": r'""".*?"""|\'\'\'.*?\'\'\'|#.*?$',
        "import_pattern": r'^(?:from|import)\s+([\w\.]+)',
        "display_name": "Python",
        "description": "A high-level, interpreted programming language known for its readability and versatility."
    },
    "javascript": {
        "extensions": [".js", ".jsx"],
        "doc_pattern": r'/\*\*.*?\*/|//.*?$',
        "import_pattern": r'^(?:import|require)\s*\(?[\'"]([^\'"]+)[\'"]\)?',
        "display_name": "JavaScript",
        "description": "A scripting language primarily used for web development."
    },
    "typescript": {
        "extensions": [".ts", ".tsx"],
        "doc_pattern": r'/\*\*.*?\*/|//.*?$',
        "import_pattern": r'^(?:import|require)\s*\(?[\'"]([^\'"]+)[\'"]\)?',
        "display_name": "TypeScript",
        "description": "A typed superset of JavaScript that compiles to plain JavaScript."
    },
    "java": {
        "extensions": [".java"],
        "doc_pattern": r'/\*\*.*?\*/|//.*?$',
        "import_pattern": r'^import\s+([\w\.]+)',
        "display_name": "Java",
        "description": "A class-based, object-oriented programming language."
    },
    "c": {
        "extensions": [".c", ".h"],
        "doc_pattern": r'/\*\*.*?\*/|//.*?$',
        "import_pattern": r'^#include\s*[<"]([^>"]+)[>"]',
        "display_name": "C",
        "description": "A general-purpose, procedural programming language."
    },
    "cpp": {
        "extensions": [".cpp", ".hpp", ".cc", ".hh"],
        "doc_pattern": r'/\*\*.*?\*/|//.*?$',
        "import_pattern": r'^#include\s*[<"]([^>"]+)[>"]',
        "display_name": "C++",
        "description": "An extension of C with object-oriented features."
    },
    "go": {
        "extensions": [".go"],
        "doc_pattern": r'//.*?$',
        "import_pattern": r'^import\s*\(?[\'"]([^\'"]+)[\'"]\)?',
        "display_name": "Go",
        "description": "A statically typed, compiled programming language designed for simplicity and efficiency."
    },
    "rust": {
        "extensions": [".rs"],
        "doc_pattern": r'//!.*?$|//.*?$',
        "import_pattern": r'^use\s+([\w:]+)',
        "display_name": "Rust",
        "description": "A systems programming language focused on safety and performance."
    },
    "ruby": {
        "extensions": [".rb"],
        "doc_pattern": r'=begin.*?=end|#.*?$',
        "import_pattern": r'^(?:require|require_relative)\s*[\'"]([^\'"]+)[\'"]',
        "display_name": "Ruby",
        "description": "A dynamic, object-oriented programming language with a focus on simplicity and productivity."
    },
    "php": {
        "extensions": [".php"],
        "doc_pattern": r'/\*\*.*?\*/|//.*?$',
        "import_pattern": r'^(?:require|include)\s*[\'"]([^\'"]+)[\'"]',
        "display_name": "PHP",
        "description": "A server-side scripting language designed for web development."
    },
    "swift": {
        "extensions": [".swift"],
        "doc_pattern": r'/\*\*.*?\*/|//.*?$',
        "import_pattern": r'^import\s+([\w\.]+)',
        "display_name": "Swift",
        "description": "A powerful and intuitive programming language for iOS, macOS, and other Apple platforms."
    },
    "kotlin": {
        "extensions": [".kt", ".kts"],
        "doc_pattern": r'/\*\*.*?\*/|//.*?$',
        "import_pattern": r'^import\s+([\w\.]+)',
        "display_name": "Kotlin",
        "description": "A modern programming language that makes developers happier."
    },
    "scala": {
        "extensions": [".scala"],
        "doc_pattern": r'/\*\*.*?\*/|//.*?$',
        "import_pattern": r'^import\s+([\w\.]+)',
        "display_name": "Scala",
        "description": "A general-purpose programming language providing support for functional programming."
    },
    "dart": {
        "extensions": [".dart"],
        "doc_pattern": r'/\*\*.*?\*/|//.*?$',
        "import_pattern": r'^import\s+[\'"]([^\'"]+)[\'"]',
        "display_name": "Dart",
        "description": "A client-optimized language for fast apps on any platform."
    },
    "haskell": {
        "extensions": [".hs"],
        "doc_pattern": r'--.*?$',
        "import_pattern": r'^import\s+([\w\.]+)',
        "display_name": "Haskell",
        "description": "A purely functional programming language with strong static typing."
    },
    "elixir": {
        "extensions": [".ex", ".exs"],
        "doc_pattern": r'#.*?$',
        "import_pattern": r'^import\s+([\w\.]+)',
        "display_name": "Elixir",
        "description": "A dynamic, functional language designed for building scalable applications."
    },
    "clojure": {
        "extensions": [".clj", ".cljs", ".cljc"],
        "doc_pattern": r';.*?$',
        "import_pattern": r'^\(require\s+\'\[([\w\.]+)\]\)',
        "display_name": "Clojure",
        "description": "A dynamic, general-purpose programming language combining the approachability of a scripting language with an efficient and robust infrastructure."
    },
    "erlang": {
        "extensions": [".erl", ".hrl"],
        "doc_pattern": r'%.*?$',
        "import_pattern": r'^-import\(([\w\.]+)\)',
        "display_name": "Erlang",
        "description": "A general-purpose, concurrent, functional programming language."
    },
    "lua": {
        "extensions": [".lua"],
        "doc_pattern": r'--.*?$',
        "import_pattern": r'^require\s*[\'"]([^\'"]+)[\'"]',
        "display_name": "Lua",
        "description": "A lightweight, high-level, multi-paradigm programming language designed primarily for embedded use in applications."
    },
    "perl": {
        "extensions": [".pl", ".pm"],
        "doc_pattern": r'#.*?$',
        "import_pattern": r'^use\s+([\w:]+)',
        "display_name": "Perl",
        "description": "A general-purpose programming language originally developed for text manipulation."
    },
    "markdown": {
        "extensions": [".md", ".markdown"],
        "doc_pattern": None, # Markdown doesn't have specific "doc patterns" like code
        "import_pattern": None, # No imports in markdown
        "display_name": "Markdown",
        "description": "A lightweight markup language for creating formatted text using a plain-text editor."
    },
    "html": {
        "extensions": [".html", ".htm"],
        "doc_pattern": r'<!--.*?-->', # HTML comments
        "import_pattern": r'<script\s+src=["\']([^"\']+)["\']|<link\s+href=["\']([^"\']+)["\']', # Basic script/link imports
        "display_name": "HTML",
        "description": "The standard markup language for documents designed to be displayed in a web browser."
    },
    "css": {
        "extensions": [".css"],
        "doc_pattern": r'/\*.*?\*/', # CSS comments
        "import_pattern": r'@import\s+url\([\'"]?([^\'"\)]+)[\'"]?\)', # CSS imports
        "display_name": "CSS",
        "description": "A style sheet language used for describing the presentation of a document written in HTML."
    }
}

def get_all_extensions() -> list[str]:
    """Get all file extensions from the language configurations."""
    extensions = []
    for lang_config in LANGUAGE_CONFIG.values():
        extensions.extend(lang_config["extensions"])
    return extensions

def get_language_metadata(file_path: str) -> Dict[str, Any]:
    """Get language metadata for a given file path."""
    _, ext = os.path.splitext(file_path)
    for lang, config in LANGUAGE_CONFIG.items():
        if ext in config["extensions"]:
            return {
                "language": lang,
                "display_name": config["display_name"],
                "description": config["description"],
                "doc_pattern": config["doc_pattern"],
                "import_pattern": config["import_pattern"]
            }
    return {
        "language": "unknown",
        "display_name": "Unknown",
        "description": "Unknown programming language",
        "doc_pattern": None,
        "import_pattern": None
    }
