"""Mock LLM client for testing without API access."""

import json
import logging
from typing import Any, Dict, List, Optional, Iterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)


class MockChatModel(BaseChatModel):
    """
    Mock chat model for testing and development without API access.
    
    This model returns simulated responses based on the input prompt.
    It's designed to test the application flow without requiring external APIs.
    """
    
    model_name: str = "mock-model"
    temperature: float = 0.1
    max_tokens: int = 4096
    
    # Predefined responses for common patterns
    _responses: Dict[str, str] = {
        "discovery": """{
            "papers": [
                {
                    "title": "Mock Paper on Research Topic",
                    "authors": ["Jane Doe", "John Smith"],
                    "year": 2023,
                    "venue": "Mock Conference",
                    "abstract": "This is a mock abstract for testing purposes."
                }
            ]
        }""",
        "paper_card": """{
            "research_problem": "Mock research problem identified from the paper",
            "research_question": "What is the mock research question?",
            "methodology": "Mock methodology description",
            "models": ["MockModel-v1"],
            "datasets": ["MockDataset"],
            "evaluation_metrics": ["accuracy", "F1"],
            "key_results": ["Result 1: Mock improvement of 5%", "Result 2: Baseline comparison"],
            "contributions": ["Contribution 1", "Contribution 2"],
            "limitations": ["Limitation 1: Small dataset", "Limitation 2: Limited evaluation"],
            "future_work": ["Future direction 1", "Future direction 2"]
        }""",
        "evidence": """{
            "claims": [
                {
                    "claim_type": "result",
                    "content": "Mock claim extracted from text",
                    "confidence": 0.85
                }
            ],
            "summary": "Mock evidence summary"
        }""",
        "comparison": """{
            "shared_approaches": ["Both use transformer architecture"],
            "key_differences": ["Method A uses attention, Method B uses convolution"],
            "consistent_findings": ["Both report improvements over baseline"],
            "conflicting_findings": []
        }""",
        "limitations": """{
            "limitations": [
                {
                    "original_text": "Our method has limitations in scalability",
                    "normalized_description": "Limited scalability to large datasets",
                    "category": "Scalability",
                    "confidence": 0.9
                }
            ]
        }""",
        "gaps": """{
            "gap_candidates": [
                {
                    "description": "Limited evaluation on multimodal tasks",
                    "category": "empirical",
                    "supporting_papers": ["paper_001", "paper_002"],
                    "initial_confidence": 0.65
                }
            ]
        }""",
        "verification": """{
            "coverage_status": "limited_evidence",
            "coverage_summary": "Analysis found limited evidence addressing this potential gap",
            "reasoning": "Among analyzed papers, few directly address this specific aspect",
            "assessment_confidence": 0.75,
            "counterevidence": []
        }""",
        "directions": """{
            "research_directions": [
                {
                    "proposed_problem": "Evaluate methods on multimodal benchmarks",
                    "motivation": "Current literature shows limited multimodal evaluation",
                    "suggested_methodology": "Apply existing methods to vision-language tasks",
                    "possible_datasets": ["VQA", "Visual Genome"],
                    "priority": "high",
                    "novelty_potential": "potential"
                }
            ]
        }"""
    }
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a mock response based on the input messages."""
        
        # Extract the last user message
        last_message = messages[-1] if messages else HumanMessage(content="")
        prompt_text = last_message.content if hasattr(last_message, 'content') else ""
        
        # Determine response type based on prompt content
        response_text = self._get_mock_response(prompt_text)
        
        # Try to parse as JSON if it looks like structured output
        try:
            parsed = json.loads(response_text)
            # Return as formatted JSON string for consistency
            response_text = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pass
        
        message = AIMessage(content=response_text)
        generation = ChatGeneration(message=message)
        
        return ChatResult(generations=[generation])
    
    def _get_mock_response(self, prompt: str) -> str:
        """Get appropriate mock response based on prompt content."""
        prompt_lower = prompt.lower()
        
        # Match prompt patterns to response types
        if any(kw in prompt_lower for kw in ["discover", "search", "find paper", "literature"]):
            return self._responses["discovery"]
        elif any(kw in prompt_lower for kw in ["paper card", "extract", "methodology", "dataset"]):
            return self._responses["paper_card"]
        elif any(kw in prompt_lower for kw in ["evidence", "claim", "verify"]):
            return self._responses["evidence"]
        elif any(kw in prompt_lower for kw in ["compare", "comparison", "difference"]):
            return self._responses["comparison"]
        elif any(kw in prompt_lower for kw in ["limitation", "weakness", "constraint"]):
            return self._responses["limitations"]
        elif any(kw in prompt_lower for kw in ["gap", "research gap", "open question"]):
            return self._responses["gaps"]
        elif any(kw in prompt_lower for kw in ["verif", "coverage", "addressed"]):
            return self._responses["verification"]
        elif any(kw in prompt_lower for kw in ["direction", "suggestion", "future work", "recommend"]):
            return self._responses["directions"]
        else:
            # Default generic response
            return self._generic_response(prompt)
    
    def _generic_response(self, prompt: str) -> str:
        """Generate a generic mock response."""
        return f"""{{
            "response": "This is a mock response. The prompt was: {prompt[:100]}...",
            "note": "Mock LLM is designed for testing. Configure a real LLM provider for actual analysis.",
            "provider_hint": "Set LLM_PROVIDER=ollama or install langchain-ollama for local inference"
        }}"""
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of generate - same implementation for mock."""
        return self._generate(messages, stop, run_manager, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM."""
        return "mock-chat"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for caching."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
