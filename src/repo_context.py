from typing import Optional, List, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.readers.github import GithubRepositoryReader, GithubClient
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document
from .config import settings
import asyncio
import nest_asyncio
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
import faiss
import os
import sys

# Enable nested event loops
nest_asyncio.apply()

class RepoContextExtractor:
    def __init__(self):
        if not settings.github_token:
            raise ValueError("GitHub token is required. Please set GITHUB_TOKEN in your .env file.")
        
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in your .env file.")
        
        self.github_client = GithubClient(
            github_token=settings.github_token,
            verbose=True
        )
        
        # Initialize with empty index
        self.index = None
        self.query_engine = None
        self.repo_info = None

    async def load_repository(self, owner: str, repo: str, branch: str = "main") -> None:
        """Load repository data and create vector index with FAISS and OpenAI Embeddings."""
        try:
            # Set OpenAI embedding model
            embed_model = OpenAIEmbedding(
                model="text-embedding-3-small",  # or "text-embedding-3-large"
                api_key=settings.openai_api_key
            )
            Settings.embed_model = embed_model
            d = 1536  # 3072 for "text-embedding-3-large"

            # Configure repository reader (no directory filter, only file extension filter)
            reader = GithubRepositoryReader(
                github_client=self.github_client,
                owner=owner,
                repo=repo,
                use_parser=False,
                verbose=True,
                filter_file_extensions=(
                    [
                        ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
                        ".json", ".ipynb"
                    ],
                    GithubRepositoryReader.FilterType.EXCLUDE,
                ),
            )

            # Get the branch info first
            branch_info = await self.github_client.get_branch(owner, repo, branch)
            if not branch_info:
                raise Exception(f"Branch {branch} not found")

            # Load documents using the reader (SYNC, not async)
            documents = reader.load_data(branch=branch)
            if not documents:
                print("Warning: No documents loaded from repository")
                return

            # Create nodes with metadata
            parser = SimpleNodeParser.from_defaults(chunk_size=4000, chunk_overlap=200)
            nodes = parser.get_nodes_from_documents(documents)

            # Add metadata to nodes
            for node in nodes:
                node.metadata.update({
                    "owner": owner,
                    "repo": repo,
                    "branch": branch
                })

            # Setup FAISS vector store
            persist_dir = f".faiss_index_{owner}_{repo}_{branch}"
            os.makedirs(persist_dir, exist_ok=True)
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)

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
                "branch": branch
            }

        except Exception as e:
            print(f"Error loading repository: {e}")
            raise Exception(f"Failed to load repository: {str(e)}")

    async def get_relevant_context(self, query: str) -> Dict[str, Any]:
        """Get relevant context from repository for a given query."""
        if not self.query_engine:
            raise Exception("Repository not loaded. Call load_repository first.")
        
        try:
            # Get response from query engine
            response = await self.query_engine.aquery(query)
            
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
            
            return await self.get_relevant_context(query)
            
        except Exception as e:
            raise Exception(f"Failed to get issue context: {str(e)}") 