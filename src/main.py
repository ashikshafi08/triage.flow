from fastapi import FastAPI, HTTPException
from .github_client import GitHubIssueClient
from .prompt_generator import PromptGenerator
from .llm_client import LLMClient
from .models import PromptRequest, PromptResponse, IssueResponse

app = FastAPI(
    title="GH Issue Prompt",
    description="Transform GitHub issues into structured LLM prompts with context-aware intelligence",
    version="0.1.0"
)

github_client = GitHubIssueClient()
prompt_generator = PromptGenerator()
llm_client = LLMClient()

@app.get("/")
async def root():
    return {"message": "GH Issue Prompt API"}

@app.post("/generate-prompt", response_model=PromptResponse)
async def generate_prompt(request: PromptRequest) -> PromptResponse:
    # First, fetch the issue
    issue_response = await github_client.get_issue(request.issue_url)
    if issue_response.status != "success":
        raise HTTPException(
            status_code=404,
            detail=f"Failed to fetch issue: {issue_response.error}"
        )

    # Generate the prompt
    prompt_response = await prompt_generator.generate_prompt(
        request,
        issue_response.data
    )

    if prompt_response.status != "success":
        raise HTTPException(
            status_code=400,
            detail=f"Failed to generate prompt: {prompt_response.error}"
        )

    # Process the prompt with LLM
    llm_response = await llm_client.process_prompt(
        prompt=prompt_response.prompt,
        prompt_type=request.prompt_type,
        context=request.context,
        model=request.model
    )

    if llm_response.status != "success":
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process prompt with LLM: {llm_response.error}"
        )

    return llm_response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 