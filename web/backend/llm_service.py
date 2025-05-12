import os
import json
from typing import Dict, Any, Optional, List
import asyncio
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import openai
import anthropic
# Using local OpenRouter implementation
from openrouter import OpenRouter

from models import LLMProvider, ProviderSettings, AppSettings


class LLMService:
    """Service to handle LLM API calls for various providers"""
    
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.clients = {}
        
        # OpenRouter site info - could be loaded from environment variables or config
        site_url = os.environ.get("SITE_URL", "https://chat2svg.ai")
        site_title = os.environ.get("SITE_TITLE", "Chat2SVG")
        
        # Initialize clients for each provider
        for provider_type, provider_settings in settings.providers.items():
            if provider_type == LLMProvider.OPENAI:
                base_url = None
                if provider_settings.api_base and provider_settings.api_base.strip():
                    base_url = provider_settings.api_base.strip()
                
                self.clients[provider_type] = openai.OpenAI(
                    api_key=provider_settings.api_key,
                    base_url=base_url
                )
            elif provider_type == LLMProvider.ANTHROPIC:
                self.clients[provider_type] = anthropic.Anthropic(
                    api_key=provider_settings.api_key
                )
            elif provider_type == LLMProvider.OPENROUTER:
                base_url = "https://openrouter.ai/api/v1"
                if provider_settings.api_base and provider_settings.api_base.strip():
                    base_url = provider_settings.api_base.strip()
                
                self.clients[provider_type] = OpenRouter(
                    api_key=provider_settings.api_key,
                    base_url=base_url,
                    site_url=site_url,
                    site_title=site_title
                )
            elif provider_type == LLMProvider.LOCAL:
                # For local, we'll use the OpenAI client with a custom base URL
                base_url = None
                if provider_settings.api_base and provider_settings.api_base.strip():
                    base_url = provider_settings.api_base.strip()
                
                self.clients[provider_type] = openai.OpenAI(
                    api_key=provider_settings.api_key or "sk-no-key-required",
                    base_url=base_url
                )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, openai.APIError, anthropic.APIError))
    )
    async def generate_completion(
        self, 
        provider: Optional[LLMProvider] = None, 
        prompt: str = "", 
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a completion from an LLM provider"""
        # Determine which provider to use
        provider = provider or self.settings.default_provider
        provider_settings = self.settings.providers[provider]
        
        # Use model from param, settings, or default
        model_name = model or provider_settings.model
        
        # Handle messages or create from prompt
        if not messages and prompt:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [{"role": "user", "content": prompt}]
        
        # Generate based on provider type
        if provider == LLMProvider.OPENAI or provider == LLMProvider.LOCAL:
            client = self.clients[provider]
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return {
                "content": response.choices[0].message.content,
                "id": response.id,
                "model": response.model,
                "provider": provider,
                "full_response": response.model_dump()
            }
            
        elif provider == LLMProvider.ANTHROPIC:
            client = self.clients[provider]
            
            # Convert messages to Anthropic format if needed
            anthropic_messages = []
            system_message = None
            
            # Extract system message if present
            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    anthropic_messages.append(msg)
            
            response = await asyncio.to_thread(
                client.messages.create,
                model=model_name,
                messages=anthropic_messages,
                system=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return {
                "content": response.content[0].text,
                "id": response.id,
                "model": response.model,
                "provider": provider,
                "full_response": response.model_dump()
            }
            
        elif provider == LLMProvider.OPENROUTER:
            client = self.clients[provider]
            
            # Add OpenRouter specific headers if provided
            headers = {}
            if "site_url" in kwargs:
                headers["HTTP-Referer"] = kwargs.pop("site_url")
            if "app_name" in kwargs:
                headers["X-Title"] = kwargs.pop("app_name")
                
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers=headers if headers else None,
                **kwargs
            )
            
            return {
                "content": response.choices[0].message.content,
                "id": response.id,
                "model": response.model,
                "provider": provider,
                "full_response": response.model_dump()
            }
        
        # Fallback response
        return {
            "content": "Error: Provider not supported",
            "id": None,
            "model": None,
            "provider": provider,
            "full_response": {}
        }


# Singleton instance
_llm_service_instance = None

def get_llm_service() -> LLMService:
    """Get or create the LLM service instance"""
    global _llm_service_instance
    
    if _llm_service_instance is None:
        # Load from environment variables
        settings = AppSettings.from_env(dict(os.environ))
        _llm_service_instance = LLMService(settings)
        
    return _llm_service_instance 