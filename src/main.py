from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import PromptRequest, PromptResponse
from .github_client import GitHubIssueClient
from .llm_client import LLMClient
from .prompt_generator import PromptGenerator
import nest_asyncio

# Enable nested event loops for Jupyter notebooks
nest_asyncio.apply()

app = FastAPI(
    title="GH Issue Prompt",
    description="Transform GitHub issues into structured LLM prompts with context-aware intelligence",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
github_client = GitHubIssueClient()
llm_client = LLMClient()
prompt_generator = PromptGenerator()

@app.get("/")
async def root():
    return {"message": "GH Issue Prompt API"}

@app.post("/generate-prompt", response_model=PromptResponse)
async def generate_prompt(request: PromptRequest) -> PromptResponse:
    try:
        # Fetch GitHub issue
        issue = await github_client.get_issue(request.issue_url)
        if not issue:
            raise HTTPException(status_code=404, detail="Issue not found")

        # Generate prompt with repository context
        prompt_response = await prompt_generator.generate_prompt(request, issue)
        if prompt_response.status == "error":
            raise HTTPException(status_code=400, detail=prompt_response.error)

        # Process prompt with LLM
        llm_response = await llm_client.process_prompt(
            prompt_response.prompt,
            model=request.model
        )

        return PromptResponse(
            status="success",
            prompt=prompt_response.prompt,
            response=llm_response
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 