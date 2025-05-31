from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from .models import PromptRequest, PromptResponse, ChatMessage, SessionResponse
from .github_client import GitHubIssueClient
from .llm_client import LLMClient
from .prompt_generator import PromptGenerator
from .session_manager import SessionManager
import nest_asyncio
import asyncio

# Enable nested event loops for Jupyter notebooks
nest_asyncio.apply()

app = FastAPI(
    title="GH Issue Prompt",
    description="Transform GitHub issues into structured LLM prompts with context-aware intelligence",
    version="0.2.0"
)

# Add CORS middleware
# It's good practice to list specific origins in production.
# For development, ["*"] is often used, but sometimes explicit origins work better.
allowed_origins = [
    "http://localhost:8080", # Your Vite frontend dev port from screenshot
    "http://localhost:5173", # Common Vite default
    "http://localhost:3000", # Common React dev port
    # Add any other origins you might be using
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, # Use the list here
    allow_credentials=True,
    allow_methods=["*"], # Allows all standard methods
    allow_headers=["*"], # Allows all headers
)

# Initialize clients and services
github_client = GitHubIssueClient()
llm_client = LLMClient()
prompt_generator = PromptGenerator()
session_manager = SessionManager()

# Background task to clean up old sessions
async def cleanup_sessions_periodically():
    while True:
        session_manager.cleanup_sessions()
        await asyncio.sleep(600)  # Clean up every 10 minutes

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_sessions_periodically())

@app.get("/")
async def root():
    return {"message": "GH Issue Prompt API"}

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: PromptRequest):
    try:
        # Create new session
        session_id = session_manager.create_session(request.issue_url, request.prompt_type)
        
        # Initialize session context in background
        await session_manager.initialize_session_context(session_id)
        
        # Get initial prompt
        session = session_manager.get_session(session_id)
        if not session or not session.get("issue_data"):
            raise HTTPException(status_code=404, detail="Issue not found")
            
        prompt_response = await prompt_generator.generate_prompt(
            request, 
            session["issue_data"]
        )
        
        if prompt_response.status == "error":
            raise HTTPException(status_code=400, detail=prompt_response.error)
            
        # Process initial prompt with LLM
        llm_response = await llm_client.process_prompt(
            prompt_response.prompt,
            prompt_type=request.prompt_type,
            model=request.llm_config.name,
            context=request.context
        )
        
        # Add initial messages to session
        session_manager.add_message(session_id, "system", prompt_response.prompt)
        session_manager.add_message(session_id, "assistant", llm_response.prompt)
        
        return {
            "session_id": session_id,
            "initial_message": llm_response.prompt
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/messages", response_model=ChatMessage)
async def handle_chat_message(session_id: str, message: ChatMessage):
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
            
        # Add user message to history
        session_manager.add_message(session_id, "user", message.content)
        
        # Generate response using conversation history
        context = session.get("repo_context", {})
        history = session.get("conversation_history", [])
        
        # Build prompt from conversation history
        conversation = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" 
             for msg in history[-6:]]  # Last 3 exchanges
        )
        
        llm_response = await llm_client.process_prompt(
            conversation,
            prompt_type=session["prompt_type"],
            context=context
        )
        
        # Add assistant response
        session_manager.add_message(session_id, "assistant", llm_response.prompt)
        
        return ChatMessage(
            role="assistant",
            content=llm_response.prompt
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
