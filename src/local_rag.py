from typing import Optional, List, Dict, Any
import os
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from .config import settings
from .local_repo_loader import clone_repo_to_temp

class LocalRepoContextExtractor:
    """Extract context from a locally cloned repository"""
    
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your .env file.")
        
        # Initialize with empty index
        self.index = None
        self.query_engine = None
        self.repo_info = None
    
    async def load_repository(self, repo_url: str, branch: str = "main") -> None:
        """Load repository by cloning it locally and creating a vector index"""
        try:
            # Set OpenAI embedding model
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",  # or "text-embedding-3-large"
                api_key=settings.openai_api_key
            )
            Settings.embed_model = embed_model
            d = 1536  # 3072 for "text-embedding-3-large"
            
            # Clone repo to temporary directory
            with clone_repo_to_temp(repo_url, branch) as repo_path:
                print(f"Cloned repository to: {repo_path}")
                
                # Load documents from the local repo
                documents = SimpleDirectoryReader(
                    repo_path,
                    exclude_hidden=True,
                    recursive=True,
                    filename_as_id=True,
                    required_exts=[
                        ".py", ".js", ".ts", ".jsx", ".tsx", ".md", 
                        ".rst", ".txt", ".java", ".c", ".cpp", ".h", 
                        ".go", ".rs", ".rb", ".php", ".html", ".css"
                    ],
                    exclude=["*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.ico", "*.json", "*.ipynb"]
                ).load_data()
                
                print(f"Loaded {len(documents)} documents from repository")
                
                # Extract repo info from URL
                # Format: https://github.com/owner/repo.git or https://github.com/owner/repo
                url_parts = repo_url.replace(".git", "").split('/')
                owner = url_parts[-2]
                repo = url_parts[-1]
                
                # Add metadata to documents
                for doc in documents:
                    doc.metadata.update({
                        "owner": owner,
                        "repo": repo,
                        "branch": branch
                    })
                
                # Create nodes with metadata
                parser = SimpleNodeParser.from_defaults()
                nodes = parser.get_nodes_from_documents(documents)
                
                # Setup FAISS vector store
                persist_dir = f".faiss_index_{owner}_{repo}_{branch}"
                os.makedirs(persist_dir, exist_ok=True)
                faiss_index = faiss.IndexFlatL2(d)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                
                # Create a new storage context without trying to load from persist_dir
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    docstore=None,  # Create new docstore instead of loading
                    index_store=None  # Create new index store instead of loading
                )
                
                # Create vector index with FAISS
                self.index = VectorStoreIndex(nodes, storage_context=storage_context)
                self.index.storage_context.persist()
                self.query_engine = self.index.as_query_engine(
                    similarity_top_k=10,  # Retrieve more chunks for broader context
                    verbose=True
                )
                
                self.repo_info = {
                    "owner": owner,
                    "repo": repo,
                    "branch": branch,
                    "url": repo_url
                }
        
        except Exception as e:
            print(f"Error loading repository: {e}")
            raise Exception(f"Failed to load repository: {str(e)}")
    
    async def get_relevant_context(self, query: str) -> Dict[str, Any]:
        """Get relevant context from repository for a given query."""
        if not self.query_engine:
            raise Exception("Repository not loaded. Call load_repository first.")
        
        try:
            # Get response from query engine (synchronous call)
            response = self.query_engine.query(query)
            
            # Extract relevant information
            context = {
                "response": str(response),
                "sources": [
                    {
                        "file": node.metadata.get("file_name", "unknown"),
                        "content": node.text
                    }
                    for node in response.source_nodes
                ]
            }
            
            return context
            
        except Exception as e:
            print(f"Error during query: {e}")
            raise Exception(f"Failed to get context: {str(e)}")

    async def get_issue_context(self, issue_title: str, issue_body: str) -> Dict[str, Any]:
        """Get relevant context for a GitHub issue."""
        if not self.query_engine:
            raise Exception("Repository not loaded. Call load_repository first.")
        
        try:
            # Create a query from issue details
            query = f"""
            Issue Title: {issue_title}
            Issue Description: {issue_body}
            
            Based on this issue, find relevant code, documentation, and context that would help understand and solve it.
            """
            
            # Note: get_relevant_context is already async, so we can await it
            return await self.get_relevant_context(query)
            
        except Exception as e:
            print(f"Error getting issue context: {e}")
            raise Exception(f"Failed to get issue context: {str(e)}") 