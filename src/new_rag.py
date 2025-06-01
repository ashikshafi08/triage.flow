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
from .config import settings
from .local_repo_loader import clone_repo_to_temp, clone_repo_to_temp_persistent
from .language_config import LANGUAGE_CONFIG, get_all_extensions, get_language_metadata
from .llm_client import LLMClient
import re
import Stemmer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import RetrieverTool
from llama_index.core.schema import IndexNode
from llama_index.core import SummaryIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import RelatedNodeInfo, NodeRelationship

import asyncio
from pathlib import Path
import aiofiles # You would need to install aiofiles
import aiofiles.os as aios

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
    "css": "css",  # Added CSS support
    "markdown": "markdown",  # Added Markdown support
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
        
        # Get all required extensions
        self.all_extensions = get_all_extensions()
        
        # Initialize LLM client
        self.llm_client = LLMClient()
    
    def _process_file_content(self, content: str, metadata: Dict[str, Any], file_path: str) -> str:
        """Process file content based on language-specific patterns."""
        if metadata["language"] == "unknown":
            return f"FILE_PATH: {file_path}\n{content}"
            
        # Extract documentation
        if metadata["doc_pattern"]:
            doc_matches = re.findall(metadata["doc_pattern"], content, re.DOTALL | re.MULTILINE)
            docs = "\n".join(doc_matches)
        else:
            docs = "No documentation pattern available"
        
        # Extract imports
        if metadata["import_pattern"]:
            import_matches = re.findall(metadata["import_pattern"], content, re.MULTILINE)
            imports = "\n".join(import_matches)
        else:
            imports = "No import pattern available"
        
        return f"""FILE_PATH: {file_path}

Language: {metadata["display_name"]}
Description: {metadata["description"]}

Imports:
{imports}

Documentation:
{docs}

Code:
{content}
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
            
            # Extract repo info from URL
            url_parts = repo_url.replace(".git", "").split('/')
            owner = url_parts[-2]
            repo = url_parts[-1]
            
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
            
            # --- START: Changes for CodeHierarchyNodeParser ---
            all_nodes = []
            for doc in processed_documents:
                lang_metadata = get_language_metadata(doc.metadata.get("original_file_path", ""))
                language_key = lang_metadata["language"] # e.g., "python", "javascript"
                tree_sitter_lang = TREE_SITTER_LANGUAGE_MAP.get(language_key, language_key)

                if tree_sitter_lang != "unknown": # Only try CodeHierarchyNodeParser for known languages
                    try:
                        # Check if we have a local query file for this language
                        query_file_path = os.path.join(LOCAL_QUERY_FILES_PATH, f"tree-sitter-{tree_sitter_lang}-tags.scm")
                        has_query_file = os.path.exists(query_file_path)
                        
                        # Get custom signature identifiers if available
                        signature_identifiers = CUSTOM_SIGNATURE_IDENTIFIERS.get(tree_sitter_lang)
                        
                        # Initialize CodeHierarchyNodeParser with the specific language
                        node_parser = CodeHierarchyNodeParser(
                            language=tree_sitter_lang,
                            signature_identifiers=signature_identifiers,
                            code_splitter=CodeSplitter(
                                language=tree_sitter_lang,
                                chunk_lines=40, # Default, can be configured
                                chunk_lines_overlap=15, # Default, can be configured
                                max_chars=1500 # Default, can be configured
                            )
                        )
                        nodes_for_doc = node_parser.get_nodes_from_documents([doc])
                        all_nodes.extend(nodes_for_doc)
                    except ImportError:
                        print(f"Warning: tree-sitter-language-pack not installed or language '{tree_sitter_lang}' not supported for CodeHierarchyNodeParser. Falling back to SimpleNodeParser for {doc.metadata.get('file_path')}.")
                        simple_parser = SimpleNodeParser.from_defaults()
                        all_nodes.extend(simple_parser.get_nodes_from_documents([doc]))
                    except ValueError as e:
                        print(f"Warning: Could not parse code for language '{tree_sitter_lang}' in file {doc.metadata.get('file_path')}: {e}. Falling back to SimpleNodeParser.")
                        simple_parser = SimpleNodeParser.from_defaults()
                        all_nodes.extend(simple_parser.get_nodes_from_documents([doc]))
                else:
                    # Fallback for truly unknown languages or non-code files
                    simple_parser = SimpleNodeParser.from_defaults()
                    all_nodes.extend(simple_parser.get_nodes_from_documents([doc]))

            # Fix any None relationships in nodes to prevent validation errors
            nodes = [fix_node_relationships(node) for node in all_nodes]
            # --- END: Changes for CodeHierarchyNodeParser ---
            
            # Setup FAISS vector store
            persist_dir = f".faiss_index_{owner}_{repo}_{branch}"
            os.makedirs(persist_dir, exist_ok=True)
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=None,
                index_store=None
            )
            
            # Create vector index with FAISS
            self.index = VectorStoreIndex(nodes, storage_context=storage_context)
            self.index.storage_context.persist()

            # Dynamically set similarity_top_k based on corpus size
            num_nodes = len(nodes)
            bm25_top_k = min(50, max(1, num_nodes // 2))  # Use half the nodes or 50, whichever is smaller
            dense_top_k = min(40, max(1, num_nodes // 2))  # Use half the nodes or 40, whichever is smaller
            
            print(f"Number of nodes: {num_nodes}, BM25 top_k: {bm25_top_k}, Dense top_k: {dense_top_k}")

            # Create BM25 retriever with proper configuration
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=bm25_top_k,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
                tokenizer=lambda t: re.split(r'[^A-Za-z0-9]', t.lower())
            )

            # Create dense retriever
            dense_retriever = self.index.as_retriever(similarity_top_k=dense_top_k)

            # Create IndexNodes for each retriever
            bm25_node = IndexNode(text="BM25 keyword retriever", index_id="bm25")
            dense_node = IndexNode(text="Dense vector retriever", index_id="dense")

            # Create a SummaryIndex with both nodes
            summary_index = SummaryIndex([bm25_node, dense_node])

            # Create a retriever dict
            retriever_dict = {
                "bm25": bm25_retriever,
                "dense": dense_retriever,
                "root": summary_index.as_retriever()
            }

            # Create RecursiveRetriever to ensemble both
            hybrid_retriever = RecursiveRetriever(
                root_id="root",
                retriever_dict=retriever_dict
            )

            # Add reranker for better results
            reranker = LLMRerank(
                top_n=10,
                llm=self.llm_client._get_openai_llm()  # Use LLM from llm_client
            )
            
            # Create query engine with hybrid retriever and reranker
            self.query_engine = RetrieverQueryEngine(
                retriever=hybrid_retriever,
                node_postprocessors=[reranker]
            )
            
            
            # Get unique languages with their display names
            languages = {}
            for doc in processed_documents:
                lang = doc.metadata.get("language")
                if lang != "unknown":
                    languages[lang] = doc.metadata.get("display_name", lang)
            
            self.repo_info = {
                "owner": owner,
                "repo": repo,
                "branch": branch,
                "url": repo_url,
                "languages": languages,
                "repo_path": repo_path  # Store the repo path for reference
            }
    
        except Exception as e:
            raise Exception(f"Failed to load repository: {str(e)}")
    
    async def get_relevant_context(self, query: str, restrict_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get relevant context from repository for a given query. Optionally restrict to specific files."""
        if not self.query_engine:
            raise Exception("Repository not loaded. Call load_repository first.")
        
        try:
            # Get response from query engine
            response = self.query_engine.query(query)

            # Extract relevant information with accurate file paths and deduplicate by file
            files_seen = set()
            deduped_sources = []
            
            for node in response.source_nodes:
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
                        "content": node.text[:1000] + "..." if len(node.text) > 1000 else node.text  # Limit content length
                    })
                    files_seen.add(file_path)
            
            context = {
                "response": str(response),
                "sources": deduped_sources,  # Use deduplicated sources
                "repo_info": self.repo_info
            }
            
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
