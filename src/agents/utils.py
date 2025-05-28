import httpx
import aiofiles
import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from langchain_core.messages import ChatMessage
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field
from markdownify import markdownify
from langchain_core.runnables import RunnableConfig

from core import get_model, settings


class CustomData(BaseModel):
    "Custom data being sent by an agent"

    data: dict[str, Any] = Field(description="The custom data")

    def to_langchain(self) -> ChatMessage:
        return ChatMessage(content=[self.data], role="custom")

    def dispatch(self, writer: StreamWriter) -> None:
        writer(self.to_langchain())

def get_config_value(value: Any) -> str:
    """
    Convert configuration values to string format, handling both string and enum types.
    
    Args:
        value (Any): The configuration value to process. Can be a string or an Enum.
    
    Returns:
        str: The string representation of the value.
        
    Examples:
        >>> get_config_value("tavily")
        'tavily'
        >>> get_config_value(SearchAPI.TAVILY)
        'tavily'
    """
    return value if isinstance(value, str) else value.value

def strip_thinking_tokens(text: str) -> str:
    """
    Remove <think> and </think> tags and their content from the text.
    
    Iteratively removes all occurrences of content enclosed in thinking tokens.
    
    Args:
        text (str): The text to process
        
    Returns:
        str: The text with thinking tokens and their content removed
    """
    while "<think>" in text and "</think>" in text:
        start = text.find("<think>")
        end = text.find("</think>") + len("</think>")
        text = text[:start] + text[end:]
    return text

def deduplicate_and_format_sources(
    search_response: Union[Dict[str, Any], List[Dict[str, Any]]], 
    max_tokens_per_source: int, 
    fetch_full_page: bool = False
) -> str:
    """
    Format and deduplicate search responses from various search APIs.
    
    Takes either a single search response or list of responses from search APIs,
    deduplicates them by URL, and formats them into a structured string.
    
    Args:
        search_response (Union[Dict[str, Any], List[Dict[str, Any]]]): Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
        max_tokens_per_source (int): Maximum number of tokens to include for each source's content
        fetch_full_page (bool, optional): Whether to include the full page content. Defaults to False.
            
    Returns:
        str: Formatted string with deduplicated sources
        
    Raises:
        ValueError: If input is neither a dict with 'results' key nor a list of search results
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source: {source['title']}\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if fetch_full_page:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sources(search_results: Dict[str, Any]) -> str:
    """
    Format search results into a bullet-point list of sources with URLs.
    
    Creates a simple bulleted list of search results with title and URL for each source.
    
    Args:
        search_results (Dict[str, Any]): Search response containing a 'results' key with
                                        a list of search result objects
        
    Returns:
        str: Formatted string with sources as bullet points in the format "* title : url"
    """
    return '\n'.join(
        f"* {source['title']} : {source['url']}"
        for source in search_results['results']
    )

def fetch_raw_content(url: str) -> Optional[str]:
    """
    Fetch HTML content from a URL and convert it to markdown format.
    
    Uses a 10-second timeout to avoid hanging on slow sites or large pages.
    
    Args:
        url (str): The URL to fetch content from
        
    Returns:
        Optional[str]: The fetched content converted to markdown if successful,
                      None if any error occurs during fetching or conversion
    """
    try:                
        # Create a client with reasonable timeout
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return markdownify(response.text)
    except Exception as e:
        print(f"Warning: Failed to fetch full page content for {url}: {str(e)}")
        return None

def get_model_from_config(config: RunnableConfig):
    """Get the model from config or use default model."""
    return get_model(config["configurable"].get("model", settings.DEFAULT_MODEL))

def extract_message_content(messages):
    """Extract content from messages, handling both single messages and lists."""
    if isinstance(messages, list):
        # Handle list of HumanMessage objects
        return " ".join(msg.content if hasattr(msg, 'content') else str(msg) for msg in messages)
    else:
        # Handle single message
        return messages.content if hasattr(messages, 'content') else str(messages)
    
async def load_mcp_config_json(filepath: str = "mcp_config.json") -> Dict[str, Any]:
    """Load and process MCP configuration from JSON file.
    
    Args:
        filepath: Path to config file relative to utils.py
        
    Returns:
        Processed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    config_path = Path(__file__).parent / filepath

    try:
        async with aiofiles.open(config_path, "r") as f:
            config = json.loads(await f.read())

        if "mcpServers" in config:
            for server_config in config["mcpServers"].values():
                if "transport" not in server_config:
                    server_config["transport"] = "stdio" if server_config.get("command") == "npx" else "sse"

        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file", e.doc, e.pos)
