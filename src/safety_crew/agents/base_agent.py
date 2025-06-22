"""
Base Agent following CrewAI documentation patterns
https://docs.crewai.com/core-concepts/agents
"""

from typing import List, Optional, Any, Dict
from crewai import Agent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the existing LLM configuration
from src.agent_tools.llm_config import get_llm_instance
from src.config import settings


class LlamaIndexLLMAdapter:
    """
    Adapter to make LlamaIndex LLMs compatible with CrewAI because crewai is a bit of a pain in the ass with llamaindexs openrouter integration
    
    CrewAI expects LLMs to have specific methods and properties and this is a workaround to make it work.
    """
    
    def __init__(self, llamaindex_llm):
        self.llamaindex_llm = llamaindex_llm
        self.model_name = getattr(llamaindex_llm, 'model', 'unknown')
        
    def __call__(self, messages, **kwargs):
        """Make the adapter callable like CrewAI expects"""
        return self.chat(messages, **kwargs)
    
    def chat(self, messages, **kwargs):
        """Handle chat completion requests"""
        try:
            # Convert messages to LlamaIndex format if needed
            if isinstance(messages, list):
                # Handle ChatMessage format
                from llama_index.core.llms import ChatMessage, MessageRole
                
                chat_messages = []
                for msg in messages:
                    if isinstance(msg, dict):
                        role = msg.get('role', 'user')
                        content = msg.get('content', '')
                        
                        # Map role names
                        if role == 'system':
                            role = MessageRole.SYSTEM
                        elif role == 'assistant':
                            role = MessageRole.ASSISTANT
                        else:
                            role = MessageRole.USER
                            
                        chat_messages.append(ChatMessage(role=role, content=content))
                    else:
                        # Assume it's already a ChatMessage
                        chat_messages.append(msg)
                
                # Use chat method for multi-turn conversations
                response = self.llamaindex_llm.chat(chat_messages)
                return response.message.content
                
            elif isinstance(messages, str):
                # Handle simple string prompt
                response = self.llamaindex_llm.complete(messages)
                return response.text
            else:
                # Fallback
                response = self.llamaindex_llm.complete(str(messages))
                return response.text
                
        except Exception as e:
            print(f"LLM Adapter Error: {e}")
            # Return a fallback response
            return f"I apologize, but I encountered an error processing your request: {str(e)}"
    
    def complete(self, prompt, **kwargs):
        """Handle completion requests"""
        try:
            response = self.llamaindex_llm.complete(prompt)
            return response.text
        except Exception as e:
            print(f"LLM Adapter Complete Error: {e}")
            return f"Error: {str(e)}"
    
    # Properties that CrewAI might expect
    @property
    def model(self):
        return self.model_name
    
    def __str__(self):
        return f"LlamaIndexLLMAdapter({self.model_name})"
    
    def __repr__(self):
        return self.__str__()


def get_crewai_compatible_llm(model: Optional[str] = None):
    """
    Get LLM instance compatible with CrewAI.
    
    Uses an adapter to make LlamaIndex LLMs work with CrewAI.
    """
    # Get your existing LlamaIndex LLM
    llamaindex_llm = get_llm_instance(
        llm_provider=settings.llm_provider,
        openrouter_api_key=settings.openrouter_api_key,
        openai_api_key=settings.openai_api_key,
        default_model=model or settings.default_model
    )
    
    # For CrewAI 0.130.0, we can try using string model names with proper environment setup
    target_model = model or settings.default_model
    
    if settings.llm_provider == "openrouter":
        # Set environment variables that CrewAI/LiteLLM can use
        os.environ["OPENROUTER_API_KEY"] = settings.openrouter_api_key
        os.environ["OPENAI_API_KEY"] = settings.openrouter_api_key  # some tools expect this for some reason lol 
        os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
        
        # Return the model string - CrewAI 0.130.0 should handle this via LiteLLM
        return f"openrouter/{target_model}"
    
    elif settings.llm_provider == "openai":
        # Set OpenAI environment variables
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        
        # Return OpenAI model string
        return target_model
    
    else:
        # Fallback to adapter approach for other providers
        return LlamaIndexLLMAdapter(llamaindex_llm)


def create_agent_with_llm(
    role: str,
    goal: str,
    backstory: str,
    tools: Optional[List] = None,
    model: Optional[str] = None,
    **kwargs
) -> Agent:

    llm = get_crewai_compatible_llm(model)
    
    agent_config = {
        "role": role,
        "goal": goal,
        "backstory": backstory,
        "llm": llm,
        "verbose": kwargs.get("verbose", True),
        "allow_delegation": kwargs.get("allow_delegation", False),
        "max_iter": kwargs.get("max_iter", 5),
        **kwargs
    }
    
    if tools:
        agent_config["tools"] = tools
    
    return Agent(**agent_config)


class BaseAgent:
    "Base class for safety agents following CrewAI patterns"
    
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: List[Any],
        llm: Any,
        verbose: bool = True,
        max_iter: int = 5,
        memory: bool = True,
        allow_delegation: bool = False
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
        self.llm = llm
        self.verbose = verbose
        self.max_iter = max_iter
        self.memory = memory
        self.allow_delegation = allow_delegation
        
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create CrewAI agent with configuration"""
        
        # Use the new compatible LLM approach
        llm_config = get_crewai_compatible_llm()
        
        return Agent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            tools=self.tools,
            llm=llm_config,
            verbose=self.verbose,
            max_iter=self.max_iter,
            memory=self.memory,
            allow_delegation=self.allow_delegation
        )
    
    def get_agent(self) -> Agent:
        "Return the configured agent"
        return self.agent