"""Message handling and format conversion for different providers."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union


class MessageRole(Enum):
    """Standard message roles across all providers."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(Enum):
    """Content block types for multi-modal messages."""
    
    TEXT = "text"
    IMAGE_URL = "image_url"
    IMAGE_FILE = "image_file"


@dataclass
class TextBlock:
    """Text content block."""
    
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageBlock:
    """Image content block with URL or base64 data."""
    
    type: Literal["image_url"] = "image_url"
    image_url: str = ""
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


# Union type for content blocks
ContentBlock = Union[TextBlock, ImageBlock]


@dataclass
class Message:
    """Unified message format across all LLM providers."""
    
    role: MessageRole
    content: Union[str, List[ContentBlock]]
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def user(cls, content: Union[str, List[ContentBlock]], **kwargs) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def assistant(cls, content: str, **kwargs) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content, **kwargs)
    
    @classmethod
    def system(cls, content: str, **kwargs) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)
    
    @classmethod
    def tool(cls, content: str, tool_call_id: str, **kwargs) -> "Message":
        """Create a tool response message."""
        return cls(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=tool_call_id,
            **kwargs
        )
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI message format."""
        message = {"role": self.role.value}
        
        if isinstance(self.content, str):
            message["content"] = self.content
        else:
            content_list = []
            for block in self.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    content_list.append({
                        "type": "image_url",
                        "image_url": {
                            "url": block.image_url,
                            "detail": block.detail
                        }
                    })
            message["content"] = content_list
        
        if self.name:
            message["name"] = self.name
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
            
        return message
    
    def to_anthropic_format(self) -> Dict[str, Any]:
        """Convert to Anthropic message format."""
        message = {"role": self.role.value}
        
        if isinstance(self.content, str):
            message["content"] = self.content
        else:
            # Multi-modal content for Anthropic
            content_list = []
            for block in self.content:
                if isinstance(block, TextBlock):
                    content_list.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageBlock):
                    content_list.append({
                        "type": "image",
                        "source": {
                            "type": "base64" if block.image_url.startswith("data:") else "url",
                            "media_type": "image/jpeg",  # Default, could be detected
                            "data": block.image_url
                        }
                    })
            message["content"] = content_list
            
        return message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "role": self.role.value,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            tool_calls=data.get("tool_calls"),
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            metadata=data.get("metadata", {}),
        )
