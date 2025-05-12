"""
OpenRouter API client implementation.
"""
import openai
from typing import Dict, Optional, Any, Mapping

class OpenRouter(openai.OpenAI):
    """
    OpenRouter API client that extends the OpenAI client.
    Provides a unified interface to access hundreds of AI models through OpenRouter.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: Optional[str] = None,
        site_title: Optional[str] = None,
        **kwargs
    ):
        """Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            base_url: OpenRouter API base URL
            site_url: Optional site URL for rankings on openrouter.ai
            site_title: Optional site title for rankings on openrouter.ai
        """
        # Set up default headers for OpenRouter
        default_headers = {}
        if site_url:
            default_headers["HTTP-Referer"] = site_url
        if site_title:
            default_headers["X-Title"] = site_title
            
        # Initialize the OpenAI client with OpenRouter configuration
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
            **kwargs
        )
        
    def add_extra_headers(self, headers: Dict[str, str]) -> None:
        """Add additional headers to the client.
        
        Args:
            headers: Additional headers to add
        """
        self._extra_headers = {**self._extra_headers, **headers} if hasattr(self, '_extra_headers') else headers 