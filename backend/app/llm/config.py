"""LLM configuration from environment variables."""

import os
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# Use Literal type instead of Enum class for Pydantic compatibility
LLMProvider = Literal["ollama", "huggingface", "openai", "anthropic", "mock"]


class LLMConfig(BaseModel):
    """Configuration for LLM client."""
    
    # Provider selection
    provider: LLMProvider = Field(
        default="ollama",
        description="LLM provider to use"
    )
    
    # Model configuration
    model_name: str = Field(
        default="llama3.1:8b",
        description="Model name/identifier"
    )
    
    # API configuration
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for API (used by Ollama, OpenAI-compatible)"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authenticated providers"
    )
    
    # Generation parameters
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)"
    )
    max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum tokens in response"
    )
    top_p: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter"
    )
    
    # Timeout and retries
    request_timeout: int = Field(
        default=120,
        ge=10,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts"
    )
    
    # Caching
    enable_caching: bool = Field(
        default=True,
        description="Enable response caching"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Directory for caching (default: ~/.cache/research-agent)"
    )
    
    # Structured output
    prefer_structured: bool = Field(
        default=True,
        description="Prefer structured JSON output when supported"
    )
    
    @classmethod
    def from_environment(cls) -> "LLMConfig":
        """Load configuration from environment variables."""
        
        provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        if provider not in [p for p in dir(LLMProvider) if not p.startswith("_")]:
            provider = "ollama"
        
        config_kwargs = {
            "provider": provider,
            "model_name": os.getenv("LLM_MODEL_NAME", cls.model_fields["model_name"].default),
            "base_url": os.getenv("LLM_BASE_URL") or cls.model_fields["base_url"].default,
            "api_key": os.getenv("LLM_API_KEY") or cls.model_fields["api_key"].default,
            "temperature": float(os.getenv("LLM_TEMPERATURE", cls.model_fields["temperature"].default)),
            "max_tokens": int(os.getenv("LLM_MAX_TOKENS", cls.model_fields["max_tokens"].default)),
            "request_timeout": int(os.getenv("LLM_TIMEOUT", cls.model_fields["request_timeout"].default)),
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", cls.model_fields["max_retries"].default)),
            "enable_caching": os.getenv("LLM_ENABLE_CACHING", "true").lower() == "true",
            "prefer_structured": os.getenv("LLM_PREFER_STRUCTURED", "true").lower() == "true",
        }
        
        # Set provider-specific defaults
        if provider == "ollama":
            config_kwargs.setdefault("base_url", "http://localhost:11434")
            config_kwargs.setdefault("model_name", "llama3.1:8b")
        elif provider == "huggingface":
            config_kwargs.setdefault("model_name", "meta-llama/Llama-3.1-8B-Instruct")
        elif provider == "openai":
            config_kwargs.setdefault("model_name", "gpt-4o-mini")
            config_kwargs.setdefault("api_key", os.getenv("OPENAI_API_KEY"))
        elif provider == "anthropic":
            config_kwargs.setdefault("model_name", "claude-3-haiku-20240307")
            config_kwargs.setdefault("api_key", os.getenv("ANTHROPIC_API_KEY"))
        elif provider == "mock":
            config_kwargs.setdefault("model_name", "mock-model")
        
        cache_dir = os.getenv("LLM_CACHE_DIR")
        if cache_dir:
            config_kwargs["cache_dir"] = cache_dir
        
        return cls(**config_kwargs)
    
    def get_cache_path(self) -> str:
        """Get the cache directory path."""
        if self.cache_dir:
            return self.cache_dir
        
        import os
        from pathlib import Path
        return str(Path.home() / ".cache" / "research-agent" / "llm_responses")
    
    def to_langchain_kwargs(self) -> Dict[str, Any]:
        """Convert to LangChain constructor kwargs."""
        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        
        if self.base_url:
            kwargs["base_url"] = self.base_url
        
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        return kwargs


# Global configuration instance (lazy initialization)
_config: Optional[LLMConfig] = None


def get_llm_config() -> LLMConfig:
    """Get the global LLM configuration."""
    global _config
    if _config is None:
        _config = LLMConfig.from_environment()
    return _config


def set_llm_config(config: LLMConfig) -> None:
    """Set the global LLM configuration."""
    global _config
    _config = config


def reset_llm_config() -> None:
    """Reset the global LLM configuration."""
    global _config
    _config = None
