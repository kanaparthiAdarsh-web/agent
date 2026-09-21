"""Factory for creating LLM clients from different providers."""

import logging
from typing import Optional, Any, Type
from functools import lru_cache

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.language_models.chat_models import BaseChatModel

from .config import LLMConfig, get_llm_config, LLMProvider

logger = logging.getLogger(__name__)


def create_llm_client(
    config: Optional[LLMConfig] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    Create an LLM client based on configuration.
    
    Args:
        config: LLM configuration (uses global config if not provided)
        provider: Override provider from config
        model_name: Override model name from config
        **kwargs: Additional arguments passed to the model constructor
    
    Returns:
        Configured LangChain chat model instance
    
    Raises:
        ValueError: If provider is not supported or required dependencies missing
    """
    if config is None:
        config = get_llm_config()
    
    # Apply overrides
    if provider:
        config.provider = provider  # type: ignore
    if model_name:
        config.model_name = model_name
    
    provider = config.provider
    
    logger.info("Creating LLM client for provider: %s, model: %s", provider, config.model_name)
    
    try:
        if provider == "ollama":
            client = _create_ollama_client(config, **kwargs)
        elif provider == "huggingface":
            client = _create_huggingface_client(config, **kwargs)
        elif provider == "openai":
            client = _create_openai_client(config, **kwargs)
        elif provider == "anthropic":
            client = _create_anthropic_client(config, **kwargs)
        elif provider == "mock":
            client = _create_mock_client(config, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        logger.info("Successfully created LLM client for %s", provider)
        return client
        
    except ImportError as e:
        logger.error("Missing dependencies for provider %s: %s", provider, e)
        raise ValueError(
            f"Missing dependencies for {provider}. "
            f"Please install the required package. Error: {e}"
        ) from e
    except Exception as e:
        logger.error("Failed to create LLM client for %s: %s", provider, e)
        raise


def _create_ollama_client(config: LLMConfig, **kwargs) -> BaseChatModel:
    """Create Ollama chat client."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama not installed. Install with: pip install langchain-ollama"
        )
    
    return ChatOllama(
        model=config.model_name,
        base_url=config.base_url,
        temperature=config.temperature,
        num_predict=config.max_tokens // 4,  # Rough conversion
        **kwargs
    )


def _create_huggingface_client(config: LLMConfig, **kwargs) -> BaseChatModel:
    """Create HuggingFace chat client."""
    try:
        from langchain_huggingface import HuggingFacePipeline
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError:
        raise ImportError(
            "Required packages not installed. Install with: pip install transformers torch"
        )
    
    # Check if we have GPU available
    device = 0 if torch.cuda.is_available() else -1
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if device >= 0 else None,
            load_in_8bit=device >= 0,
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
        )
        
        return HuggingFacePipeline(pipeline=pipe, **kwargs)
        
    except Exception as e:
        logger.warning("Failed to load HuggingFace model locally: %s", e)
        logger.info("Falling back to HuggingFace Inference API")
        return _create_huggingface_api_client(config, **kwargs)


def _create_huggingface_api_client(config: LLMConfig, **kwargs) -> BaseChatModel:
    """Create HuggingFace Inference API client."""
    try:
        from langchain_huggingface import HuggingFaceEndpoint
    except ImportError:
        raise ImportError(
            "langchain-huggingface not installed. Install with: pip install langchain-huggingface"
        )
    
    hf_token = config.api_key or kwargs.get("huggingfacehub_api_token")
    if not hf_token:
        raise ValueError(
            "HuggingFace API token required. Set HF_TOKEN environment variable or pass api_key."
        )
    
    return HuggingFaceEndpoint(
        repo_id=config.model_name,
        huggingfacehub_api_token=hf_token,
        temperature=config.temperature,
        max_new_tokens=config.max_tokens,
        top_p=config.top_p,
        **kwargs
    )


def _create_openai_client(config: LLMConfig, **kwargs) -> BaseChatModel:
    """Create OpenAI chat client."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai not installed. Install with: pip install langchain-openai"
        )
    
    api_key = config.api_key
    if not api_key:
        raise ValueError(
            "OpenAI API key required. Set OPENAI_API_KEY environment variable or pass api_key."
        )
    
    return ChatOpenAI(
        model=config.model_name,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
        **kwargs
    )


def _create_anthropic_client(config: LLMConfig, **kwargs) -> BaseChatModel:
    """Create Anthropic chat client."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic not installed. Install with: pip install langchain-anthropic"
        )
    
    api_key = config.api_key
    if not api_key:
        raise ValueError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable or pass api_key."
        )
    
    return ChatAnthropic(
        model=config.model_name,
        anthropic_api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        timeout=config.request_timeout,
        max_retries=config.max_retries,
        **kwargs
    )


def _create_mock_client(config: LLMConfig, **kwargs) -> BaseChatModel:
    """Create mock LLM client for testing without API access."""
    from .mock_client import MockChatModel
    
    logger.warning("Using MOCK LLM client - responses will be simulated")
    return MockChatModel(
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        **kwargs
    )


@lru_cache(maxsize=1)
def get_cached_llm_client(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
) -> BaseChatModel:
    """
    Get a cached LLM client (singleton per provider/model combination).
    
    Args:
        provider: LLM provider (uses config default if not specified)
        model_name: Model name (uses config default if not specified)
    
    Returns:
        Cached LLM client instance
    """
    config = get_llm_config()
    return create_llm_client(
        config=config,
        provider=provider,
        model_name=model_name,
    )


def clear_llm_cache() -> None:
    """Clear the LLM client cache."""
    get_cached_llm_client.cache_clear()


# Convenience function for structured output
async def invoke_with_structured_output(
    llm: BaseChatModel,
    prompt: str,
    output_schema: Optional[Type] = None,
    **kwargs
) -> Any:
    """
    Invoke LLM with structured output if supported.
    
    Args:
        llm: LLM client
        prompt: Input prompt
        output_schema: Pydantic schema for structured output (optional)
        **kwargs: Additional invoke arguments
    
    Returns:
        LLM response (structured if schema provided and supported)
    """
    from .structured_output import parse_structured_response
    
    # Try to use native structured output if available
    if output_schema and hasattr(llm, "with_structured_output"):
        try:
            structured_llm = llm.with_structured_output(output_schema)
            return await structured_llm.ainvoke(prompt, **kwargs)
        except Exception as e:
            logger.warning("Structured output failed, falling back to parsing: %s", e)
    
    # Fall back to parsing JSON from response
    response = await llm.ainvoke(prompt, **kwargs)
    
    if output_schema:
        return parse_structured_response(response, output_schema)
    
    return response
