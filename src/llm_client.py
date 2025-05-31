from typing import Optional, Dict, Any
import httpx
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from .config import settings
from .models import PromptResponse

class LLMClient:
    def __init__(self):
        self.default_model = settings.default_model
        
        # Define system prompts for different types
        self.system_prompts = {
            "explain": """You are an expert software engineer. Your task is to explain GitHub issues clearly, concisely, and technically.
            Begin with a brief summary, then elaborate on the problem, its root cause, and potential impact.
            When relevant, refer to the specific repository details provided in the context.""",
            
            "fix": """You are an expert software engineer. Your task is to provide detailed yet concise solutions for GitHub issues.
            Include essential code changes, necessary tests, and consider edge cases.
            When relevant, refer to the specific repository details provided in the context.""",
            
            "test": """You are an expert software engineer. Your task is to create comprehensive and focused test cases for GitHub issues.
            Focus on verifying the issue and validating potential fixes efficiently.
            When relevant, refer to the specific repository details provided in the context.""",
            
            "summarize": """You are an expert software engineer. Your task is to summarize GitHub issues concisely and accurately.
            Focus on key points, current status, and actionable next steps.
            When relevant, refer to the specific repository details provided in the context."""
            # Add similar modifications for other prompt types if they exist (document, review, prioritize from prompt_generator.py)
        }

    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return settings.model_configs.get(model, settings.model_configs[self.default_model])

    def _get_openai_llm(self, model: Optional[str] = None):
        """Get OpenAI LLM instance."""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key is required")
        return OpenAI(
            api_key=settings.openai_api_key,
            model=model or self.default_model
        )

    async def _get_openrouter_response(self, prompt: str, model: str, system_prompt: str) -> Dict[str, Any]:
        """Get response from OpenRouter API."""
        if not settings.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")

        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/your-repo",  # Replace with your actual repo
            "X-Title": "GH Issue Prompt"  # Your app name
        }

        model_config = self._get_model_config(model)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": model_config["max_tokens"],
                        "temperature": model_config["temperature"]
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract the response and token usage
                return {
                    "text": data["choices"][0]["message"]["content"],
                    "tokens_used": data.get("usage", {}).get("total_tokens")
                }
        except httpx.HTTPStatusError as e:
            error_detail = f"HTTP Status Error: {e.response.status_code} - {e.response.text}"
            print(error_detail)
            raise ValueError(error_detail) from e
        except httpx.RequestError as e:
            error_detail = f"HTTP Request Error: An error occurred while requesting {e.request.url!r} - {str(e)}"
            print(error_detail)
            raise ValueError(error_detail) from e
        except Exception as e:
            error_detail = f"Unexpected error in OpenRouter response: {str(e)}"
            print(error_detail)
            raise ValueError(error_detail) from e

    async def process_prompt(self, prompt: str, prompt_type: str, context: Optional[Dict] = None, model: Optional[str] = None) -> PromptResponse:
        try:
            model = model or self.default_model
            system_prompt = self.system_prompts.get(prompt_type, "")
            
            if settings.llm_provider == "openai":
                # Create a prompt template with system message
                template = PromptTemplate(
                    template=prompt,
                    system_prompt=system_prompt
                )
                
                # Get LLM instance with specified model
                llm = self._get_openai_llm(model)
                
                # Get response from LLM
                response = await llm.acomplete(template.format())
                return PromptResponse(
                    status="success",
                    prompt=response.text,
                    model_used=model
                )
                
            elif settings.llm_provider == "openrouter":
                response_data = await self._get_openrouter_response(prompt, model, system_prompt)
                return PromptResponse(
                    status="success",
                    prompt=response_data["text"],
                    model_used=model,
                    tokens_used=response_data.get("tokens_used")
                )
                
            else:
                return PromptResponse(
                    status="error",
                    error=f"Unsupported LLM provider: {settings.llm_provider}"
                )
            
        except Exception as e:
            print(f"Error in process_prompt: {str(e)}")
            return PromptResponse(
                status="error",
                error=f"Failed to process prompt: {str(e)}"
            )
