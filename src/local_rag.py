from typing import Optional, List, Dict, Any, Tuple
import os
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.schema import Document
from llama_index.retrievers.bm25 import BM25Retriever
from .config import settings
from .local_repo_loader import clone_repo_to_temp, clone_repo_to_temp_persistent
from .language_config import LANGUAGE_CONFIG, get_all_extensions, get_language_metadata
from .llm_client import LLMClient
import re
import Stemmer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
import asyncio
from pathlib import Path
from llama_index.core.tools import RetrieverTool
from llama_index.core.schema import IndexNode
from llama_index.core import SummaryIndex
from llama_index.core.retrievers import RecursiveRetriever


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
            
            # Load documents from the local repo
            documents = SimpleDirectoryReader(
                repo_path,
                exclude_hidden=True,
                recursive=True,
                required_exts=self.all_extensions,
                exclude=["*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.ico", "*.json", "*.ipynb"]
            ).load_data()
            
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
            
            # Create nodes with metadata
            parser = SimpleNodeParser.from_defaults()
            nodes = parser.get_nodes_from_documents(processed_documents)
            
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

            # Create BM25 retriever with proper configuration
            bm25_retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=50,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
                tokenizer=lambda t: re.split(r'[^A-Za-z0-9]', t.lower())
            )

            # Create dense retriever
            dense_retriever = self.index.as_retriever(similarity_top_k=40)

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
