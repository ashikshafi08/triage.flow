from typing import Optional, Dict
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from .config import settings
from .models import PromptResponse

class LLMClient:
    def __init__(self):
        # Default model if none specified
        self.default_model = "gpt-4o-mini"
        
        # Define system prompts for different types
        self.system_prompts = {
            "explain": """You are an expert software engineer. Your task is to explain GitHub issues in a clear and technical manner.
            Focus on understanding the problem, its root cause, and potential impact.""",
            
            "fix": """You are an expert software engineer. Your task is to provide detailed solutions for GitHub issues.
            Include code changes, necessary tests, and consider edge cases.""",
            
            "test": """You are an expert software engineer. Your task is to create comprehensive test cases for GitHub issues.
            Focus on verifying the issue and validating potential fixes.""",
            
            "summarize": """You are an expert software engineer. Your task is to summarize GitHub issues concisely.
            Focus on key points, current status, and next steps."""
        }

    def _get_llm(self, model: Optional[str] = None):
        """Get LLM instance with specified model."""
        return OpenAI(
            api_key=settings.openai_api_key,
            model=model or self.default_model
        )

    async def process_prompt(self, prompt: str, prompt_type: str, context: Optional[Dict] = None, model: Optional[str] = None) -> PromptResponse:
        try:
            # Create a prompt template with system message
            template = PromptTemplate(
                template=prompt,
                system_prompt=self.system_prompts.get(prompt_type, "")
            )
            
            # Get LLM instance with specified model
            llm = self._get_llm(model)
            
            # Get response from LLM
            response = await llm.acomplete(template.format())
            
            return PromptResponse(
                status="success",
                prompt=response.text
            )
            
        except Exception as e:
            return PromptResponse(
                status="error",
                error=f"Failed to process prompt: {str(e)}"
            ) 