"""Structured output parsing for LLM responses."""

import json
import logging
import re
from typing import Any, Type, Optional, Dict
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def parse_structured_response(
    response: Any,
    output_schema: Type[BaseModel],
    max_retries: int = 3,
) -> BaseModel:
    """
    Parse LLM response into a structured Pydantic model.
    
    Args:
        response: Raw LLM response (string or message object)
        output_schema: Target Pydantic model class
        max_retries: Maximum parsing attempts
    
    Returns:
        Parsed Pydantic model instance
    
    Raises:
        ValueError: If response cannot be parsed as valid JSON
        ValidationError: If parsed data doesn't match schema
    """
    # Extract text content from response
    text_content = _extract_text_from_response(response)
    
    # Try to extract JSON from the response
    json_string = _extract_json_from_text(text_content)
    
    if not json_string:
        raise ValueError(
            f"Could not find valid JSON in response. Response preview: {text_content[:200]}..."
        )
    
    # Parse JSON
    try:
        parsed_data = json.loads(json_string)
    except json.JSONDecodeError as e:
        # Try to fix common JSON issues
        parsed_data = _try_fix_json(json_string, e)
        if parsed_data is None:
            raise ValueError(f"Failed to parse JSON: {e}")
    
    # Validate against schema
    try:
        return output_schema.model_validate(parsed_data)
    except ValidationError as e:
        logger.warning("Schema validation failed: %s", e)
        # Try to partially validate by providing defaults
        return _partial_validate(parsed_data, output_schema)


def _extract_text_from_response(response: Any) -> str:
    """Extract text content from various response formats."""
    if isinstance(response, str):
        return response
    
    # LangChain message types
    if hasattr(response, 'content'):
        return str(response.content)
    
    if hasattr(response, 'text'):
        return response.text
    
    # Dict with content key
    if isinstance(response, dict) and 'content' in response:
        return str(response['content'])
    
    # Fallback: convert to string
    return str(response)


def _extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object/array from text that may contain additional content.
    
    Handles cases where LLM wraps JSON in markdown code blocks or adds explanations.
    """
    # First, try to find JSON wrapped in markdown code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    code_block_match = re.search(code_block_pattern, text)
    if code_block_match:
        return code_block_match.group(1).strip()
    
    # Try to find JSON object pattern
    # Look for balanced braces
    brace_start = text.find('{')
    if brace_start != -1:
        json_obj = _extract_balanced_braces(text, brace_start)
        if json_obj:
            return json_obj
    
    # Try to find JSON array pattern
    bracket_start = text.find('[')
    if bracket_start != -1:
        json_arr = _extract_balanced_brackets(text, bracket_start)
        if json_arr:
            return json_arr
    
    # If text looks like pure JSON, return it
    stripped = text.strip()
    if (stripped.startswith('{') and stripped.endswith('}')) or \
       (stripped.startswith('[') and stripped.endswith(']')):
        return stripped
    
    return None


def _extract_balanced_braces(text: str, start: int) -> Optional[str]:
    """Extract a balanced JSON object starting at given position."""
    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    
    return None


def _extract_balanced_brackets(text: str, start: int) -> Optional[str]:
    """Extract a balanced JSON array starting at given position."""
    depth = 0
    in_string = False
    escape_next = False
    
    for i in range(start, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\':
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
    
    return None


def _try_fix_json(json_string: str, error: json.JSONDecodeError) -> Optional[Dict]:
    """Attempt to fix common JSON parsing issues."""
    import ast
    
    # Try using ast.literal_eval for Python-style dicts
    try:
        result = ast.literal_eval(json_string)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    # Try adding missing closing braces
    open_braces = json_string.count('{')
    close_braces = json_string.count('}')
    if open_braces > close_braces:
        fixed = json_string + '}' * (open_braces - close_braces)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
    
    # Try removing trailing commas
    fixed = re.sub(r',(\s*[}\]])', r'\1', json_string)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # Try fixing single quotes
    fixed = json_string.replace("'", '"')
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None


def _partial_validate(data: Dict, schema: Type[BaseModel]) -> BaseModel:
    """
    Attempt partial validation by filling in missing required fields with defaults.
    
    This is a fallback when full validation fails but we want to salvage what we can.
    """
    # Get field info
    model_fields = schema.model_fields
    
    # Fill in missing required fields with defaults
    cleaned_data = dict(data)
    
    for field_name, field_info in model_fields.items():
        if field_name not in cleaned_data:
            # Use default if available
            if field_info.default is not None:
                cleaned_data[field_name] = field_info.default
            elif field_info.default_factory is not None:
                cleaned_data[field_name] = field_info.default_factory()
            else:
                # Set appropriate empty default based on type
                field_type = field_info.annotation
                if field_type == list or (hasattr(field_type, '__origin__') and 
                                           getattr(field_type.__origin__, '_name', None) == 'List'):
                    cleaned_data[field_name] = []
                elif field_type == dict or (hasattr(field_type, '__origin__') and 
                                             getattr(field_type.__origin__, '_name', None) == 'Dict'):
                    cleaned_data[field_name] = {}
                elif field_type == str:
                    cleaned_data[field_name] = ""
                elif field_type in (int, float):
                    cleaned_data[field_name] = 0
                elif field_type == bool:
                    cleaned_data[field_name] = False
                else:
                    cleaned_data[field_name] = None
    
    # Try validation again with filled data
    try:
        return schema.model_validate(cleaned_data)
    except ValidationError as e:
        logger.error("Partial validation also failed: %s", e)
        # Last resort: create with minimal data
        raise e


def format_prompt_for_structured_output(
    prompt: str,
    output_schema: Type[BaseModel],
    include_schema_description: bool = True,
) -> str:
    """
    Format a prompt to encourage structured JSON output.
    
    Args:
        prompt: Base prompt text
        output_schema: Target Pydantic model
        include_schema_description: Whether to include field descriptions
    
    Returns:
        Formatted prompt with JSON output instructions
    """
    # Build schema description
    schema_desc_parts = []
    
    if include_schema_description:
        schema_desc_parts.append("\nRespond with a JSON object matching this schema:")
        
        for field_name, field_info in output_schema.model_fields.items():
            description = field_info.description or "No description"
            is_required = field_info.is_required()
            default = field_info.default if not is_required else None
            
            field_desc = f"  - {field_name}: {description}"
            if not is_required:
                field_desc += f" (optional, default: {default})"
            schema_desc_parts.append(field_desc)
    
    schema_desc_parts.append(
        "\nIMPORTANT: Return ONLY valid JSON. Do not include explanations outside the JSON object."
    )
    
    return prompt + "\n".join(schema_desc_parts)
