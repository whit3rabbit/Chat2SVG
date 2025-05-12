from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class ProviderSettings(BaseModel):
    provider: LLMProvider
    api_key: str
    api_base: Optional[str] = None
    model: str
    additional_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    

class AppSettings(BaseModel):
    default_provider: LLMProvider
    providers: Dict[LLMProvider, ProviderSettings]
    
    @classmethod
    def from_env(cls, env_dict: Dict[str, str]):
        """Create settings from environment variables"""
        providers = {}
        
        # OpenAI
        if "OPENAI_API_KEY" in env_dict and env_dict["OPENAI_API_KEY"]:
            providers[LLMProvider.OPENAI] = ProviderSettings(
                provider=LLMProvider.OPENAI,
                api_key=env_dict["OPENAI_API_KEY"],
                api_base=env_dict.get("OPENAI_API_BASE"),
                model=env_dict.get("OPENAI_MODEL", "gpt-4")
            )
            
        # Anthropic
        if "ANTHROPIC_API_KEY" in env_dict and env_dict["ANTHROPIC_API_KEY"]:
            providers[LLMProvider.ANTHROPIC] = ProviderSettings(
                provider=LLMProvider.ANTHROPIC,
                api_key=env_dict["ANTHROPIC_API_KEY"],
                api_base=env_dict.get("ANTHROPIC_API_BASE"),
                model=env_dict.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
            )
            
        # OpenRouter
        if "OPENROUTER_API_KEY" in env_dict and env_dict["OPENROUTER_API_KEY"]:
            providers[LLMProvider.OPENROUTER] = ProviderSettings(
                provider=LLMProvider.OPENROUTER,
                api_key=env_dict["OPENROUTER_API_KEY"],
                api_base=env_dict.get("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"),
                model=env_dict.get("OPENROUTER_MODEL", "openai/gpt-4")
            )
            
        # Local
        if "LOCAL_LLM_BASE_URL" in env_dict and env_dict["LOCAL_LLM_BASE_URL"]:
            providers[LLMProvider.LOCAL] = ProviderSettings(
                provider=LLMProvider.LOCAL,
                api_key=env_dict.get("LOCAL_LLM_API_KEY", ""),  # Some local APIs don't need a key
                api_base=env_dict["LOCAL_LLM_BASE_URL"],
                model=env_dict.get("LOCAL_LLM_MODEL", "default")
            )
        
        # Determine default provider - prioritize Anthropic (Claude), then OpenAI, then others
        default_provider = None
        if LLMProvider.ANTHROPIC in providers:
            default_provider = LLMProvider.ANTHROPIC
        elif LLMProvider.OPENAI in providers:
            default_provider = LLMProvider.OPENAI
        elif LLMProvider.OPENROUTER in providers:
            default_provider = LLMProvider.OPENROUTER
        elif LLMProvider.LOCAL in providers:
            default_provider = LLMProvider.LOCAL
        else:
            raise ValueError("No LLM provider configuration found in environment variables")
        
        return cls(
            default_provider=default_provider,
            providers=providers
        ) 