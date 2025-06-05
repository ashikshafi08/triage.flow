import aiohttp
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

class TriageFlowMCPClient:
    """Client for interacting with the triage.flow MCP server"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def load_repository(self, repo_url: str, branch: str = "main") -> Dict[str, Any]:
        """Load a repository into the MCP server"""
        async with self.session.post(
            f"{self.base_url}/mcp/repositories",
            json={"repo_url": repo_url, "branch": branch}
        ) as response:
            if response.status != 200:
                error = await response.text()
                raise Exception(f"Failed to load repository: {error}")
            return await response.json()
    
    async def list_repositories(self) -> List[Dict[str, Any]]:
        """List all loaded repositories"""
        async with self.session.get(f"{self.base_url}/mcp/repositories") as response:
            if response.status != 200:
                error = await response.text()
                raise Exception(f"Failed to list repositories: {error}")
            return await response.json()
    
    async def query_repository(
        self,
        repo_url: str,
        query: str,
        session_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Query a repository with natural language"""
        async with self.session.post(
            f"{self.base_url}/mcp/query",
            json={
                "repo_url": repo_url,
                "query": query,
                "session_id": session_id,
                "model": model
            }
        ) as response:
            if response.status != 200:
                error = await response.text()
                raise Exception(f"Failed to query repository: {error}")
            return await response.json()
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        async with self.session.get(f"{self.base_url}/mcp/sessions/{session_id}") as response:
            if response.status != 200:
                error = await response.text()
                raise Exception(f"Failed to get session history: {error}")
            return await response.json()

# Example usage:
"""
async def main():
    async with TriageFlowMCPClient() as client:
        # Load a repository
        repo = await client.load_repository("https://github.com/username/repo")
        
        # Query the repository
        response = await client.query_repository(
            repo_url="https://github.com/username/repo",
            query="What is the main purpose of this repository?"
        )
        
        # Get session history
        history = await client.get_session_history(response["session_id"])
        print(history)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
""" 