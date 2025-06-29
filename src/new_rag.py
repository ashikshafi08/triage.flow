from typing import Optional, List, Dict, Any, Tuple
import os
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.text_splitter import CodeSplitter # Import CodeSplitter
from llama_index.packs.code_hierarchy import CodeHierarchyNodeParser # Import CodeHierarchyNodeParser
from llama_index.packs.code_hierarchy.code_hierarchy import _SignatureCaptureOptions, _SignatureCaptureType, _CommentOptions, _ScopeMethod # Import signature capture classes
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.retrievers import QueryFusionRetriever
from .config import settings
from .local_repo_loader import clone_repo_to_temp, clone_repo_to_temp_persistent
from .language_config import LANGUAGE_CONFIG, get_all_extensions, get_language_metadata
from .llm_client import LLMClient
from .agent_tools.llm_config import get_llm_instance
from .cache import rag_cache, folder_cache
from .enhanced_persistence import persistence_manager, IndexMetadata, FileChecksum
import re
import Stemmer
import logging
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import RetrieverTool
from llama_index.core.schema import IndexNode
from llama_index.core import SummaryIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import RelatedNodeInfo, NodeRelationship
import fnmatch

import asyncio
from pathlib import Path
import aiofiles # You would need to install aiofiles
import aiofiles.os as aios
from datetime import datetime

logger = logging.getLogger(__name__)

# Add a mapping for tree-sitter languages if necessary, or assume direct match
TREE_SITTER_LANGUAGE_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "go": "go",
    "rust": "rust",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "scala": "scala",
    "dart": "dart",
    "haskell": "haskell",
    "elixir": "elixir",
    "clojure": "clojure",
    "erlang": "erlang",
    "lua": "lua",
    "perl": "perl",
    "css": "css",  
    "markdown": "markdown",     
    "html": "unknown",  
    "shell": "unknown",  
    "dockerfile": "unknown",  
    "jinja": "unknown",  
    "yaml": "unknown",  
    "json": "unknown",  
    "xml": "unknown",  
    "ini": "unknown",  
    "toml": "unknown",  
    # Add more if your language_config has names different from tree-sitter's
}

# Path to local query files
LOCAL_QUERY_FILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pytree-sitter-queries")

# Custom signature identifiers for additional languages
CUSTOM_SIGNATURE_IDENTIFIERS = {
    "css": {
        "rule_set": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="selectors",
        ),
        "at_rule": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="at_keyword",
        ),
    },
    "markdown": {
        "atx_heading": _SignatureCaptureOptions(
            name_identifier="heading_content",
        ),
        "fenced_code_block": _SignatureCaptureOptions(
            name_identifier="info_string",
        ),
    },
    "javascript": {
        "method_definition": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="name.definition.method",
        ),
        "function_declaration": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="name.definition.function",
        ),
        "class_declaration": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="name.definition.class",
        ),
    },
    "c": {
        "function_definition": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="declarator",
        ),
        "struct_specifier": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="name",
        ),
        "enum_specifier": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="name",
        ),
    },
    "cpp": {
        "function_definition": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="declarator",
        ),
        "class_specifier": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="name",
        ),
        "struct_specifier": _SignatureCaptureOptions(
            end_signature_types=[_SignatureCaptureType(type="{", inclusive=False)],
            name_identifier="name",
        ),
    },
}

# Custom comment options for additional languages
CUSTOM_COMMENT_OPTIONS = {
    "css": _CommentOptions(
        comment_template="/* {} */", scope_method=_ScopeMethod.BRACKETS
    ),
    "markdown": _CommentOptions(
        comment_template="<!-- {} -->", scope_method=_ScopeMethod.INDENTATION
    ),
    "javascript": _CommentOptions(
        comment_template="// {}", scope_method=_ScopeMethod.BRACKETS
    ),
    "c": _CommentOptions(
        comment_template="// {}", scope_method=_ScopeMethod.BRACKETS
    ),
    "cpp": _CommentOptions(
        comment_template="// {}", scope_method=_ScopeMethod.BRACKETS
    ),
}

class AsyncDirectoryReader:
    """Custom asynchronous directory reader for LlamaIndex."""
    def __init__(self, input_dir: str, exclude_hidden: bool = True, recursive: bool = True, required_exts: List[str] = None, exclude: List[str] = None):
        self.input_dir = input_dir
        self.exclude_hidden = exclude_hidden
        self.recursive = recursive
        self.required_exts = required_exts if required_exts else []
        self.exclude_files = exclude if exclude else []

    async def load_data(self) -> List[Document]:
        documents = []
        file_paths = []
        
        for dirpath, dirnames, filenames in os.walk(self.input_dir):
            if not self.recursive:
                # If not recursive, only process the top directory
                if dirpath != self.input_dir:
                    continue

            if self.exclude_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith('.')]
                filenames = [f for f in filenames if not f.startswith('.')]

            for filename in filenames:
                file_path = Path(dirpath) / filename
                
                # Check required extensions
                if self.required_exts and file_path.suffix not in self.required_exts:
                    continue
                
                # Check excluded files
                if any(file_path.match(pattern) for pattern in self.exclude_files):
                    continue
                
                file_paths.append(file_path)

        semaphore = asyncio.Semaphore(100)  # Limit to 100 concurrent open files
        tasks = [self._read_file(file_path, semaphore) for file_path in file_paths]
        results = await asyncio.gather(*tasks)
        # Filter out None results (files that couldn't be read or were skipped)
        return [doc for doc in results if doc is not None]

    async def _read_file(self, file_path: Path, semaphore: asyncio.Semaphore) -> Optional[Document]:
        """Read a single file and return a Document object or None if reading fails."""
        async with semaphore: # Acquire semaphore before opening file
            try:
                # Check if it's a valid file and not a special file type like a socket or FIFO
                if not await aios.path.isfile(file_path) or await aios.path.islink(file_path) or await aios.path.ismount(file_path):
                    # print(f"Skipping non-regular file: {file_path}") # Optional: for debugging
                    return None
                async with aiofiles.open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
                    content = await f.read()
                return Document(text=content, metadata={"file_path": str(file_path)})
            except Exception as e:
                # print(f"Error reading file {file_path}: {e}") # Optional: for debugging, can be noisy
                return None


def fix_node_relationships(node):
    """Fix any None relationships in a node to prevent Pydantic validation errors."""
    # Handle None relationships
    for rel_type in list(node.relationships.keys()):
        if node.relationships[rel_type] is None:
            # Initialize with empty list for list relationships
            if rel_type in [NodeRelationship.CHILD, NodeRelationship.NEXT, NodeRelationship.PREVIOUS]:
                node.relationships[rel_type] = []
            # Remove any other None relationships that should be RelatedNodeInfo
            else:
                del node.relationships[rel_type]
    return node


def patch_code_hierarchy_parser():
    """Patch the CodeHierarchyNodeParser to support additional languages."""
    from llama_index.packs.code_hierarchy.code_hierarchy import _COMMENT_OPTIONS, _DEFAULT_SIGNATURE_IDENTIFIERS
    
    # Add our custom comment options to the global _COMMENT_OPTIONS
    for lang, options in CUSTOM_COMMENT_OPTIONS.items():
        if lang not in _COMMENT_OPTIONS:
            _COMMENT_OPTIONS[lang] = options
    
    # Add our custom signature identifiers to the global _DEFAULT_SIGNATURE_IDENTIFIERS
    for lang, identifiers in CUSTOM_SIGNATURE_IDENTIFIERS.items():
        if lang not in _DEFAULT_SIGNATURE_IDENTIFIERS:
            _DEFAULT_SIGNATURE_IDENTIFIERS[lang] = identifiers


# Patch the CodeHierarchyNodeParser to support additional languages
patch_code_hierarchy_parser()


class LocalRepoContextExtractor:
    """Extract context from a locally cloned repository with enhanced multi-language support"""
    
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your .env file.")
        
        # Initialize with empty index
        self.index = None
        self.query_engine = None
        self.repo_info = None
        self.reranker = None  # Initialize reranker
        
        # Get all required extensions
        self.all_extensions = get_all_extensions()
        
        # Initialize LLM client
        self.llm_client = LLMClient()
        
        # Cache for file pattern searches to avoid repeated disk walks
        self._file_pattern_cache = {}
    
    def _invalidate_file_cache(self):
        """Invalidate file pattern cache (call when repo is reloaded)."""
        self._file_pattern_cache = {}
    
    def _is_file_oriented_query(self, query: str) -> bool:
        """Determine if the query is asking about files specifically."""
        file_patterns = [
            r'\bwhich files?\b',
            r'\bwhat files?\b', 
            r'\blist.*files?\b',
            r'\bfiles? .*contain\b',
            r'\bfiles? .*define\b',
            r'\bfiles? .*implement\b',
            r'\bfiles? that\b',
            r'\bwhere.*file\b',
            r'\bfile.*path\b',
            r'\bshow.*files?\b',
            r'\bfind.*files?\b',
            r'\bshow me.*\.py\b',  # Added: "show me *.py"
            r'\bcontains class\b',  # Added: "contains class"
            r'\*\.py\b',  # glob patterns
            r'\*\.js\b',
            r'\*\.ts\b',
            r'\.py$',     # file extensions
            r'\.js$',
            r'\.ts$'
        ]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in file_patterns)
    
    def _search_files_by_pattern(self, query: str, restrict_files: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Search for files using glob patterns and keywords."""
        if not self.current_repo_path or not os.path.exists(self.current_repo_path):
            return []
        
        # Create cache key based on query and restrict_files
        cache_key = (query.lower(), tuple(sorted(restrict_files)) if restrict_files else None)
        
        # Check cache first
        if cache_key in self._file_pattern_cache:
            return self._file_pattern_cache[cache_key]
        
        results = []
        query_lower = query.lower()
        
        # Extract potential glob patterns from query
        glob_patterns = re.findall(r'\*\.[a-zA-Z]+', query)
        
        # Extract keywords that might be in file names
        keywords = re.findall(r'\b\w+\b', query_lower)
        keywords = [k for k in keywords if len(k) > 2 and k not in ['the', 'and', 'are', 'files', 'that', 'which', 'what']]
        
        for root, dirs, files in os.walk(self.current_repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.current_repo_path)
                
                # Skip if restrict_files is set and this file isn't in it
                if restrict_files and rel_path not in restrict_files:
                    continue
                
                match_score = 0
                match_reasons = []
                
                # Check glob patterns
                for pattern in glob_patterns:
                    if fnmatch.fnmatch(file.lower(), pattern.lower()):
                        match_score += 10
                        match_reasons.append(f"matches pattern {pattern}")
                
                # Check keywords in filename and path
                file_text = (file + " " + rel_path).lower()
                for keyword in keywords:
                    if keyword in file_text:
                        match_score += 5
                        match_reasons.append(f"contains '{keyword}'")
                
                if match_score > 0:
                    # Get language metadata without reading file content yet
                    metadata = get_language_metadata(file_path)
                    
                    file_result = {
                        "file": rel_path,
                        "language": metadata["display_name"],
                        "description": metadata["description"],
                        "content": "",  # Will be filled if score is high enough
                        "match_score": match_score,
                        "match_reasons": match_reasons
                    }
                    
                    # Only read file content if match score is high enough (threshold = 15)
                    if match_score >= 15:  # Raised from 10 to 15 for better performance
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                # Read first 3KB for preview instead of 1KB to show more actual code
                                content = f.read(3000)
                                file_result["content"] = content + "..." if len(content) == 3000 else content
                        except Exception:
                            file_result["content"] = "Could not read file content"
                    else:
                        file_result["content"] = "Preview not loaded (low match score)"
                    
                    results.append(file_result)
        
        # Sort by match score descending
        results.sort(key=lambda x: x["match_score"], reverse=True)
        final_results = results[:20]  # Limit to top 20 matches
        
        # Cache the results
        self._file_pattern_cache[cache_key] = final_results
        
        return final_results
    
    def _process_file_content(self, content: str, metadata: Dict[str, Any], file_path: str) -> str:
        """Process file content based on language-specific patterns."""
        # Define safe limits for embedding (account for ~1.3x token-to-character ratio)
        MAX_CONTENT_CHARS = 6000  # Safe limit well below 8192 tokens
        MAX_SECTION_CHARS = 1000  # Limit for docs/imports sections
        
        if metadata["language"] == "unknown":
            # Truncate unknown content to safe limits
            truncated_content = content[:MAX_CONTENT_CHARS]
            if len(content) > MAX_CONTENT_CHARS:
                truncated_content += "\n... [Content truncated for embedding]"
            return f"FILE_PATH: {file_path}\n{truncated_content}"
            
        # Extract documentation with size limits
        docs = "No documentation pattern available"
        if metadata["doc_pattern"]:
            doc_matches = re.findall(metadata["doc_pattern"], content, re.DOTALL | re.MULTILINE)
            if doc_matches:
                docs_text = "\n".join(doc_matches)
                if len(docs_text) > MAX_SECTION_CHARS:
                    docs = docs_text[:MAX_SECTION_CHARS] + "... [Docs truncated]"
                else:
                    docs = docs_text
        
        # Extract imports with size limits
        imports = "No import pattern available"
        if metadata["import_pattern"]:
            import_matches = re.findall(metadata["import_pattern"], content, re.MULTILINE)
            if import_matches:
                # Handle tuples from regex patterns with multiple capture groups
                if isinstance(import_matches[0], tuple):
                    # Flatten tuples and filter out empty strings
                    import_list = []
                    for match in import_matches:
                        for item in match:
                            if item.strip():  # Only add non-empty strings
                                import_list.append(item.strip())
                    imports_text = "\n".join(import_list)
                else:
                    imports_text = "\n".join(import_matches)
                
                if len(imports_text) > MAX_SECTION_CHARS:
                    imports = imports_text[:MAX_SECTION_CHARS] + "... [Imports truncated]"
                else:
                    imports = imports_text
        
        # Calculate remaining space for actual code content
        header_size = len(f"""FILE_PATH: {file_path}

Language: {metadata["display_name"]}
Description: {metadata["description"]}

Imports:
{imports}

Documentation:
{docs}

Code:
""")
        
        remaining_chars = MAX_CONTENT_CHARS - header_size
        if remaining_chars < 500:  # Ensure minimum space for code
            remaining_chars = 500
            
        # Truncate code content if necessary
        truncated_content = content
        if len(content) > remaining_chars:
            truncated_content = content[:remaining_chars] + "\n... [Code truncated for embedding]"

        return f"""FILE_PATH: {file_path}

Language: {metadata["display_name"]}
Description: {metadata["description"]}

Imports:
{imports}

Documentation:
{docs}

Code:
{truncated_content}
"""
    
    async def load_repository(self, repo_url: str, branch: str = "main") -> None:
        """Load repository by cloning it locally and creating a vector index with enhanced language support"""
        try:
            # Set OpenAI embedding model
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=settings.openai_api_key
            )
            Settings.embed_model = embed_model
            d = 1536
            
            # Clone repo to persistent temporary directory (not auto-deleted)
            repo_path = clone_repo_to_temp_persistent(repo_url, branch)
            print(f"Cloned repository to: {repo_path}")
            self.current_repo_path = repo_path
            
            # Extract repo info from URL
            url_parts = repo_url.replace(".git", "").split('/')
            owner = url_parts[-2]
            repo = url_parts[-1]
            
            # Use enhanced persistence manager
            index_dir = persistence_manager.get_index_dir(repo_url, branch)
            
            # Check if we can load existing index
            existing_metadata = persistence_manager.load_metadata(index_dir)
            should_rebuild = True
            
            if existing_metadata and persistence_manager.validate_index_integrity(index_dir):
                logger.info(f"Found existing index for {owner}/{repo}")
                should_rebuild, change_info = persistence_manager.should_rebuild_index(
                    existing_metadata, 
                    Path(repo_path), 
                    self.all_extensions
                )
                
                if not should_rebuild:
                    logger.info(f"Loading existing index: {change_info.get('reason', 'no changes detected')}")
                    try:
                        # Load existing index
                        storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
                        self.vector_store = VectorStoreIndex.load_from_storage(storage_context)
                        self.index = self.vector_store  # For backwards compatibility
                        
                        # We still need to create BM25 and other components since they're not persisted
                        await self._setup_retrievers_from_existing_index(existing_metadata, repo_path)
                        
                        # Update repo info
                        self.repo_info = {
                            "owner": owner,
                            "repo": repo,
                            "branch": branch,
                            "url": repo_url,
                            "languages": existing_metadata.to_dict().get("languages", {}),
                            "repo_path": repo_path
                        }
                        
                        # Invalidate file pattern cache when repo is reloaded
                        self._invalidate_file_cache()
                        
                        logger.info(f"Successfully loaded existing index with {existing_metadata.total_nodes} nodes")
                        return
                        
                    except Exception as e:
                        logger.warning(f"Failed to load existing index: {e}, rebuilding...")
                        persistence_manager.cleanup_corrupted_index(index_dir)
                        should_rebuild = True
                else:
                    logger.info(f"Rebuilding index: {change_info.get('reason', 'unknown')}")
                    if change_info.get('total_changes', 0) > 0:
                        logger.info(f"Changes: {change_info['total_changes']} files affected")
            
            # Build new index if needed
            if should_rebuild:
                await self._build_new_index(repo_url, branch, repo_path, owner, repo, index_dir)
                
        except Exception as e:
            raise Exception(f"Failed to load repository: {str(e)}")
    
    async def _setup_retrievers_from_existing_index(self, metadata: IndexMetadata, repo_path: str) -> None:
        """Setup retrievers and query engine from existing index metadata"""
        # We need to recreate the nodes for BM25 since it's not persisted
        # But we can do this efficiently by only reading the files we know about
        logger.info("Setting up retrievers from existing index...")
        
        # Create minimal documents for BM25 (we don't need full processing)
        documents = []
        for file_path, checksum in metadata.file_checksums.items():
            full_path = Path(repo_path) / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Create a simplified document
                    doc = Document(
                        text=content[:4000],  # Truncate for BM25
                        metadata={
                            "file_path": file_path,
                            "file_name": Path(file_path).name,
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to read {file_path} for BM25: {e}")
        
        if documents:
            # Create nodes for BM25
            parser = SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200)
            nodes = parser.get_nodes_from_documents(documents)
            
            # Setup BM25 retriever
            num_nodes = len(nodes)
            bm25_top_k = min(200, max(1, num_nodes // 3))
            dense_top_k = min(200, max(1, num_nodes // 3))
            
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=bm25_top_k,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
                tokenizer=lambda t: re.split(r'[^A-Za-z0-9]', t.lower())
            )
            
            # Create hybrid retriever
            vector_retriever = self.vector_store.as_retriever(similarity_top_k=dense_top_k)
            self.hybrid_retriever = QueryFusionRetriever(
                [vector_retriever, self.bm25_retriever],
                similarity_top_k=200,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=True,
                verbose=True,
            )
            
            # Create reranker and query engine
            self.reranker = LLMRerank(
                top_n=10,
                llm=get_llm_instance()
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=self.hybrid_retriever,
                node_postprocessors=[self.reranker]
            )
            
            logger.info(f"Successfully setup retrievers with {len(nodes)} nodes")
        else:
            logger.warning("No documents found for BM25 setup")
    
    async def _build_new_index(self, repo_url: str, branch: str, repo_path: str, owner: str, repo: str, index_dir: Path) -> None:
        """Build a completely new index"""
        logger.info(f"Building new index for {owner}/{repo}")
        
        # Load documents from the local repo using AsyncDirectoryReader
        reader = AsyncDirectoryReader(
            repo_path,
            exclude_hidden=True,
            recursive=True,
            required_exts=self.all_extensions,
            exclude=["*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.ico", "*.json", "*.ipynb"]
        )
        documents = await reader.load_data()
        
        print(f"Loaded {len(documents)} documents from repository")
        
        # Process each document with language-specific metadata and ensure proper file paths
        processed_documents = []
        for doc in documents:
            # Get the original FULL file path (not just filename)
            original_file_path = doc.metadata.get("file_path", "") or doc.metadata.get("source", "")
            
            # Make the file path relative to the repository root
            if original_file_path.startswith(repo_path):
                relative_file_path = os.path.relpath(original_file_path, repo_path)
            else:
                relative_file_path = original_file_path
            
            # Get language metadata
            metadata = get_language_metadata(original_file_path)
            
            # Process content based on language
            processed_content = self._process_file_content(doc.text, metadata, relative_file_path)
            
            # Create a new document with processed content and corrected metadata
            new_doc = Document(
                text=processed_content,
                metadata={
                    **doc.metadata,
                    "file_path": relative_file_path,  # Store the correct relative path
                    "file_name": os.path.basename(original_file_path),  # Keep just the filename
                    "original_file_path": original_file_path,  # Keep original for debugging
                    "owner": owner,
                    "repo": repo,
                    "branch": branch,
                    "language": metadata["language"],
                    "display_name": metadata["display_name"],
                    "description": metadata["description"]
                }
            )
            processed_documents.append(new_doc)
        
        # Create nodes using CodeHierarchyNodeParser
        all_nodes = []
        for doc in processed_documents:
            lang_metadata = get_language_metadata(doc.metadata.get("original_file_path", ""))
            language_key = lang_metadata["language"]
            tree_sitter_lang = TREE_SITTER_LANGUAGE_MAP.get(language_key, language_key)

            if tree_sitter_lang != "unknown":
                try:
                    signature_identifiers = CUSTOM_SIGNATURE_IDENTIFIERS.get(tree_sitter_lang)
                    
                    node_parser = CodeHierarchyNodeParser(
                        language=tree_sitter_lang,
                        signature_identifiers=signature_identifiers or {},
                        code_splitter=CodeSplitter(
                            language=tree_sitter_lang,
                            chunk_lines=80,
                            chunk_lines_overlap=20,
                            max_chars=4000
                        )
                    )
                    nodes_for_doc = node_parser.get_nodes_from_documents([doc])
                    all_nodes.extend(nodes_for_doc)
                except Exception as e:
                    print(f"Warning: CodeHierarchyNodeParser failed for {tree_sitter_lang}: {e}. Using SimpleNodeParser.")
                    simple_parser = SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200)
                    all_nodes.extend(simple_parser.get_nodes_from_documents([doc]))
            else:
                simple_parser = SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200)
                all_nodes.extend(simple_parser.get_nodes_from_documents([doc]))

        # Fix any None relationships in nodes
        nodes = [fix_node_relationships(node) for node in all_nodes]
        
        # Create FAISS vector store
        d = 1536  # text-embedding-3-small dimensions
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        logger.info(f"Creating new FAISS index with {len(nodes)} nodes (this will generate embeddings)")
        self.vector_store = VectorStoreIndex(nodes, storage_context=storage_context)
        
        # Persist to the enhanced persistence directory
        self.vector_store.storage_context.persist(persist_dir=str(index_dir))
        logger.info(f"Persisted new FAISS index to {index_dir}")

        # For backwards compatibility
        self.index = self.vector_store

        # Create BM25 retriever
        num_nodes = len(nodes)
        bm25_top_k = min(200, max(1, num_nodes // 3))
        dense_top_k = min(200, max(1, num_nodes // 3))
        
        logger.info(f"Number of nodes: {num_nodes}, BM25 top_k: {bm25_top_k}, Dense top_k: {dense_top_k}")

        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=bm25_top_k,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
            tokenizer=lambda t: re.split(r'[^A-Za-z0-9]', t.lower())
        )

        # Create hybrid retriever
        vector_retriever = self.vector_store.as_retriever(similarity_top_k=dense_top_k)
        self.hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, self.bm25_retriever],
            similarity_top_k=200,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
        )
        
        # Create reranker and query engine
        self.reranker = LLMRerank(
            top_n=10,
            llm=get_llm_instance()
        )
        
        self.query_engine = RetrieverQueryEngine(
            retriever=self.hybrid_retriever,
            node_postprocessors=[self.reranker]
        )
        
        # Get unique languages with their display names
        languages = {}
        for doc in processed_documents:
            lang = doc.metadata.get("language")
            if lang != "unknown":
                languages[lang] = doc.metadata.get("display_name", lang)
        
        # Update repo info
        self.repo_info = {
            "owner": owner,
            "repo": repo,
            "branch": branch,
            "url": repo_url,
            "languages": languages,
            "repo_path": repo_path
        }
        
        # Save enhanced metadata
        file_checksums = persistence_manager.scan_repository_files(Path(repo_path), self.all_extensions)
        
        metadata = IndexMetadata(
            repo_url=repo_url,
            branch=branch,
            owner=owner,
            repo=repo,
            total_files=len(processed_documents),
            total_nodes=len(nodes),
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat(),
            index_version="2.0",
            embedding_model="text-embedding-3-small",
            file_checksums=file_checksums
        )
        
        persistence_manager.save_metadata(index_dir, metadata)
        
        # Invalidate file pattern cache
        self._invalidate_file_cache()
        
        logger.info(f"Successfully built and saved new index for {owner}/{repo}")

    def _calculate_query_complexity(self, query: str, restrict_files: Optional[List[str]] = None) -> int:
        """Calculate query complexity score to determine optimal retrieval depth."""
        complexity = 0
        
        # Word count factor
        word_count = len(query.split())
        if word_count <= settings.SIMPLE_QUERY_WORD_LIMIT:
            complexity += 1
        elif word_count <= settings.COMPLEX_QUERY_WORD_THRESHOLD:
            complexity += 3
        else:
            complexity += 5
        
        # File mentions (@file) add complexity
        file_mentions = len(re.findall(r'@[\w\-/\\.]+', query))
        complexity += file_mentions * settings.FILE_MENTION_WEIGHT
        
        # Folder restrictions add complexity
        if restrict_files:
            complexity += min(5, len(restrict_files) // 10)  # Cap at 5
        
        # Repository overview questions need more sources
        overview_patterns = [
            r'\b(what is this|what does this do|tell me about|overview|summary|introduce|explain this)\b',
            r'\b(repo|repository|project|codebase)\b',
            r'\b(never worked|new to|unfamiliar|first time)\b'
        ]
        
        for pattern in overview_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                complexity += 5  # Boost overview questions significantly
        
        # Technical terms and patterns
        technical_patterns = [
            r'\b(implement|algorithm|optimize|refactor|debug|architecture)\b',
            r'\b(class|function|method|interface|abstract)\b',
            r'\b(async|await|promise|callback|thread|concurrent)\b',
            r'\b(api|endpoint|route|middleware|handler)\b',
            r'\b(bug|error|exception|issue|problem)\b'
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                complexity += 2
        
        # Question complexity indicators
        if any(word in query.lower() for word in ['how', 'why', 'explain', 'analyze']):
            complexity += 3
        
        return complexity
    
    def _get_optimal_source_count(self, complexity: int) -> int:
        """Determine optimal number of sources based on query complexity."""
        if complexity <= 5:
            return settings.MIN_RAG_SOURCES
        elif complexity <= 10:
            return settings.DEFAULT_RAG_SOURCES
        elif complexity <= 20:
            return 15
        else:
            return settings.MAX_RAG_SOURCES
    
    async def _generate_folder_summary(self, folder_path: str) -> Dict[str, Any]:
        """Generate a summary for a folder (cached for performance)."""
        cache_key = f"folder_summary_{folder_path}"
        
        # Try to get from cache
        cached = await folder_cache.get(cache_key)
        if cached:
            return cached
        
        summary = {
            "path": folder_path,
            "file_count": 0,
            "languages": {},
            "key_files": [],
            "structure": {}
        }
        
        # Analyze folder structure
        for root, dirs, files in os.walk(os.path.join(self.current_repo_path, folder_path)):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.startswith('.'):
                    continue
                    
                summary["file_count"] += 1
                file_path = os.path.join(root, file)
                metadata = get_language_metadata(file_path)
                
                lang = metadata["language"]
                if lang != "unknown":
                    summary["languages"][lang] = summary["languages"].get(lang, 0) + 1
                
                # Identify key files (configs, main files, etc.)
                if file in ['README.md', 'package.json', 'requirements.txt', 'setup.py', 
                           'main.py', 'index.js', 'index.ts', 'app.py', 'server.py']:
                    rel_path = os.path.relpath(file_path, self.current_repo_path)
                    summary["key_files"].append(rel_path)
        
        # Cache the summary
        await folder_cache.set(cache_key, summary, settings.CACHE_TTL_FOLDER)
        
        return summary
    
    async def get_relevant_context(self, query: str, restrict_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get relevant context from repository for a given query. Optionally restrict to specific files."""
        if not self.query_engine:
            raise Exception("Repository not loaded. Call load_repository first.")
        
        # Check if caching is enabled
        if settings.ENABLE_RAG_CACHING:
            # Generate cache key
            cache_key = rag_cache._generate_cache_key(query, restrict_files, self.repo_info)
            
            # Try to get from cache
            cached_result = await rag_cache.get(cache_key)
            if cached_result:
                print(f"RAG cache hit for query: {query[:50]}...")
                return cached_result
        
        try:
            # Check if this is a file-oriented query
            if self._is_file_oriented_query(query):
                print(f"Detected file-oriented query: {query}")
                
                # Use file pattern search
                file_results = self._search_files_by_pattern(query, restrict_files)
                
                if file_results:
                  
                    file_paths = [f["file"] for f in file_results[:10]]  # Top 10 files
                    wanted_files = set(file_paths)
                    
                    try:
                  
                        retrieved_nodes_with_score = self.query_engine.retriever.retrieve(query)
                        rag_sources = []
                        
                        # Normalize file paths for Windows compatibility
                        wanted_files_normalized = {fp.replace("\\", "/") for fp in wanted_files}
                        
                        for node_with_score in retrieved_nodes_with_score: # Iterate through NodeWithScore
                            node = node_with_score.node # Get the actual node
                            file_path = node.metadata.get("file_path", "unknown")
                            # Normalize the node's file path for comparison
                            normalized_file_path = file_path.replace("\\", "/")
                            # Only include nodes from the files we found
                            if normalized_file_path in wanted_files_normalized:
                                rag_sources.append({
                                    "file": file_path,  # Keep original path format
                                    "language": node.metadata.get("display_name", "unknown"),
                                    "description": node.metadata.get("description", "No description available"),
                                    "content": node.text[:5000] + "..." if len(node.text) > 5000 else node.text
                                })
                    except Exception as e:
                        print(f"Error getting RAG context for file results: {e}")
                        rag_sources = []
                    
                    # Combine file search results with RAG context
                    combined_sources = []
                    seen_files = set()
                    
                    # Add file search results first
                    for file_result in file_results:
                        file_path = file_result["file"]
                        if file_path not in seen_files:
                            combined_sources.append({
                                "file": file_path,
                                "language": file_result["language"],
                                "description": file_result["description"],
                                "content": file_result["content"],
                                "match_reasons": file_result.get("match_reasons", [])
                            })
                            seen_files.add(file_path)
                    
                    # Add any additional RAG sources
                    for rag_source in rag_sources:
                        file_path = rag_source["file"]
                        if file_path not in seen_files:
                            combined_sources.append(rag_source)
                            seen_files.add(file_path)
                    
                    result = {
                        "response": f"Found {len(file_results)} files matching your query. Here are the most relevant files:",
                        "sources": combined_sources,
                        "repo_info": self.repo_info,
                        "search_type": "file_oriented"
                    }
                    
                    # Cache the result
                    if settings.ENABLE_RAG_CACHING:
                        await rag_cache.set(cache_key, result, settings.CACHE_TTL_RAG)
                    
                    return result
                else:
                    # No file pattern matches found - return clear message
                    result = {
                        "response": "No files matched your query pattern. The search looked for file names, paths, and extensions but found no matches.",
                        "sources": [],
                        "repo_info": self.repo_info,
                        "search_type": "file_oriented_no_match"
                    }
                    
                    # Cache even empty results to avoid repeated searches
                    if settings.ENABLE_RAG_CACHING:
                        await rag_cache.set(cache_key, result, settings.CACHE_TTL_RAG // 2)  # Shorter TTL for empty results
                    
                    return result
            
            # Regular RAG search with smart sizing
            complexity = self._calculate_query_complexity(query, restrict_files)
            optimal_source_count = self._get_optimal_source_count(complexity)
            
            print(f"Query complexity: {complexity}, using {optimal_source_count} sources")
            
            # Temporarily adjust retriever settings if smart sizing is enabled
            if settings.ENABLE_SMART_SIZING:
                # Store original settings
                original_dense_top_k = self.index.as_retriever().similarity_top_k
                original_reranker_top_n = self.reranker.top_n if hasattr(self, 'reranker') else 10
                
                # Apply smart sizing
                self.index.as_retriever().similarity_top_k = optimal_source_count * 2  # Get more initially
                if hasattr(self, 'reranker'):
                    self.reranker.top_n = optimal_source_count
            
            # Conditionally disable expensive LLM reranker when only a few sources are needed
            if settings.ENABLE_SMART_SIZING and optimal_source_count <= settings.MIN_RAG_SOURCES:
                # Temporarily remove LLMRerank to avoid extra calls (if supported)
                if hasattr(self.query_engine, "postprocessors"):
                    original_postprocessors = self.query_engine.postprocessors
                    self.query_engine.postprocessors = [p for p in original_postprocessors if not isinstance(p, LLMRerank)]
                    try:
                        response = self.query_engine.query(query)
                    finally:
                        # Restore postprocessors safely
                        self.query_engine.postprocessors = original_postprocessors
                else:
                    # Fallback: attribute not present in current LlamaIndex version
                    response = self.query_engine.query(query)
            else:
                response = self.query_engine.query(query)
            
            # Restore original settings if we changed them
            if settings.ENABLE_SMART_SIZING:
                self.index.as_retriever().similarity_top_k = original_dense_top_k
                if hasattr(self, 'reranker'):
                    self.reranker.top_n = original_reranker_top_n

            # Extract relevant information with accurate file paths and deduplicate by file
            files_seen = set()
            deduped_sources = []
            
            for node in response.source_nodes[:optimal_source_count]:  # Limit to optimal count
                file_path = node.metadata.get("file_path", "unknown")
                # If restrict_files is set, only include those files
                if restrict_files and file_path not in restrict_files:
                    continue
                # Only add if we haven't seen this file before
                if file_path not in files_seen:
                    deduped_sources.append({
                        "file": file_path,
                        "language": node.metadata.get("display_name", "unknown"),
                        "description": node.metadata.get("description", "No description available"),
                        "content": node.text[:5000] + "..." if len(node.text) > 5000 else node.text  # Increased from 1000 to 5000 for better code visibility
                    })
                    files_seen.add(file_path)
            
            context = {
                "response": str(response),
                "sources": deduped_sources,  # Use deduplicated sources
                "repo_info": self.repo_info,
                "search_type": "regular",
                "complexity": complexity,
                "optimal_source_count": optimal_source_count
            }
            
            # Cache the result
            if settings.ENABLE_RAG_CACHING:
                await rag_cache.set(cache_key, context, settings.CACHE_TTL_RAG)
            
            return context
            
        except Exception as e:
            raise Exception(f"Failed to get context: {str(e)}")

    async def get_issue_context(self, issue_title: str, issue_body: str) -> Dict[str, Any]:
        """Get relevant context for a GitHub issue."""
        if not self.query_engine:
            raise Exception("Repository not loaded. Call load_repository first.")
        
        try:
            # Create a query from issue details
            languages = self.repo_info.get("languages", {})
            language_list = ", ".join(languages.values()) if languages else "unknown"
            
            query = f"""
            Issue Title: {issue_title}
            Issue Description: {issue_body}
            
            Based on this issue, find relevant code, documentation, and context that would help understand and solve it.
            Consider the programming languages used in the repository: {language_list}
            """
            
            return await self.get_relevant_context(query)
            
        except Exception as e:
            raise Exception(f"Failed to get issue context: {str(e)}")

    async def _rebuild_indices_from_existing_repo(self) -> None:
        """Rebuild RAG indices from an existing repository without re-cloning."""
        if not self.current_repo_path or not os.path.exists(self.current_repo_path):
            raise Exception(f"Repository path not found: {self.current_repo_path}")
        
        logger.info(f"Rebuilding RAG indices from existing repo at: {self.current_repo_path}")
        
        try:
            # Process repository files (similar to load_repository but without cloning)
            processed_documents = await self._process_repository_files()
            logger.info(f"Processed {len(processed_documents)} documents for re-indexing")

            if not processed_documents:
                raise Exception("No documents found in repository for indexing")

            # Build nodes and index (same logic as in load_repository)
            all_nodes = []
            for doc in processed_documents:
                lang_metadata = get_language_metadata(doc.metadata.get("original_file_path", ""))
                language_key = lang_metadata["language"]
                tree_sitter_lang = TREE_SITTER_LANGUAGE_MAP.get(language_key, language_key)

                if tree_sitter_lang != "unknown":
                    try:
                        query_file_path = os.path.join(LOCAL_QUERY_FILES_PATH, f"tree-sitter-{tree_sitter_lang}-tags.scm")
                        
                        if os.path.exists(query_file_path):
                            code_splitter = CodeSplitter(
                                language=tree_sitter_lang,
                                chunk_lines=80,
                                chunk_lines_overlap=20,
                                max_chars=4000
                            )
                            parser = CodeHierarchyNodeParser(
                                language=tree_sitter_lang,
                                code_splitter=code_splitter
                            )
                            nodes = parser.get_nodes_from_documents([doc])
                            all_nodes.extend(nodes)
                        else:
                            fallback_parser = SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200)
                            nodes = fallback_parser.get_nodes_from_documents([doc])
                            all_nodes.extend(nodes)
                    except Exception as e:
                        logger.warning(f"Tree-sitter parsing failed for {tree_sitter_lang}: {e}")
                        fallback_parser = SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200)
                        nodes = fallback_parser.get_nodes_from_documents([doc])
                        all_nodes.extend(nodes)
                else:
                    fallback_parser = SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200)
                    nodes = fallback_parser.get_nodes_from_documents([doc])
                    all_nodes.extend(nodes)

            logger.info(f"Created {len(all_nodes)} nodes for vector indexing")

            # Set up vector store
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=settings.openai_api_key
            )
            Settings.embed_model = embed_model

            # Create hybrid retriever and index
            self.vector_store = VectorStoreIndex(all_nodes)
            self.bm25_retriever = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=200)
            
            vector_retriever = self.vector_store.as_retriever(similarity_top_k=200)
            self.hybrid_retriever = QueryFusionRetriever(
                [vector_retriever, self.bm25_retriever],
                similarity_top_k=200,
                num_queries=1,
                mode="reciprocal_rerank",
                use_async=True,
                verbose=True,
            )

            # Create reranker and query engine
            self.reranker = LLMRerank(
                top_n=5,
                llm=get_llm_instance()
            )
            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=self.hybrid_retriever,
                node_postprocessors=[self.reranker],
                llm=get_llm_instance()
            )

            logger.info(f"Successfully rebuilt RAG indices for repository at {self.current_repo_path}")

        except Exception as e:
            logger.error(f"Failed to rebuild indices from existing repo: {e}")
            raise

    async def _process_repository_files(self) -> List[Document]:
        """Process files from the existing repository path and return processed documents."""
        if not self.current_repo_path:
            raise Exception("No repository path set")
        
        # Load documents from the local repo using AsyncDirectoryReader
        reader = AsyncDirectoryReader(
            self.current_repo_path,
            exclude_hidden=True,
            recursive=True,
            required_exts=self.all_extensions,
            exclude=["*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.ico", "*.json", "*.ipynb"]
        )
        documents = await reader.load_data()
        
        logger.info(f"Loaded {len(documents)} documents from repository")
        
        # Extract repo info from current path
        # Assume repo path ends with owner/repo format
        path_parts = self.current_repo_path.split(os.path.sep)
        repo = path_parts[-1] if path_parts else "unknown"
        owner = "unknown"  # We'll try to extract this from repo_info if available
        
        if hasattr(self, 'repo_info') and self.repo_info:
            owner = self.repo_info.get("owner", "unknown")
            repo = self.repo_info.get("repo", repo)
        
        # Process each document with language-specific metadata and ensure proper file paths
        processed_documents = []
        for doc in documents:
            # Get the original FULL file path (not just filename)
            original_file_path = doc.metadata.get("file_path", "") or doc.metadata.get("source", "")
            
            # Make the file path relative to the repository root
            if original_file_path.startswith(self.current_repo_path):
                relative_file_path = os.path.relpath(original_file_path, self.current_repo_path)
            else:
                relative_file_path = original_file_path
            
            # Get language metadata
            metadata = get_language_metadata(original_file_path)
            
            # Process content based on language
            processed_content = self._process_file_content(doc.text, metadata, relative_file_path)
            
            # Create a new document with processed content and corrected metadata
            new_doc = Document(
                text=processed_content,
                metadata={
                    **doc.metadata,
                    "file_path": relative_file_path,  # Store the correct relative path
                    "file_name": os.path.basename(original_file_path),  # Keep just the filename
                    "original_file_path": original_file_path,  # Keep original for debugging
                    "owner": owner,
                    "repo": repo,
                    "branch": "main",  # Default branch
                    "language": metadata["language"],
                    "display_name": metadata["display_name"],
                    "description": metadata["description"]
                }
            )
            processed_documents.append(new_doc)
        
        return processed_documents
