"""Message handling and format conversion for different providers."""

import mimetypes
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
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
    DOCUMENT = "document"


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


@dataclass
class DocumentBlock:
    """Document content block for PDFs and other documents."""
    
    type: Literal["document"] = "document"
    document_url: str = ""
    document_type: str = "pdf"
    filename: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_url(cls, document_url: str, filename: Optional[str] = None, **kwargs) -> "DocumentBlock":
        """Create DocumentBlock with auto-detected document type from URL."""
        document_type = cls._detect_type_from_url(document_url)
        return cls(
            document_url=document_url,
            document_type=document_type,
            filename=filename,
            **kwargs
        )
    
    @staticmethod
    def _detect_type_from_url(url: str) -> str:
        """Detect document type from URL."""
        if url.startswith('data:'):
            return 'pdf' if 'application/pdf' in url else 'pdf'
        
        extension = Path(urllib.parse.urlparse(url).path).suffix.lower()
        types = {'.pdf': 'pdf', '.doc': 'doc', '.docx': 'docx', '.txt': 'txt', '.csv': 'csv'}
        return types.get(extension, 'pdf')


# Union type for content blocks
ContentBlock = Union[TextBlock, ImageBlock, DocumentBlock]


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
    
    @classmethod
    def from_pdf(cls, text: str, pdf_url: str, filename: Optional[str] = None, **kwargs) -> "Message":
        """Create a user message with PDF attachment."""
        content = [
            TextBlock(text=text),
            DocumentBlock(
                document_url=pdf_url,
                document_type="pdf",
                filename=filename
            )
        ]
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def from_image(cls, text: str, image_url: str, detail: Optional[Literal["auto", "low", "high"]] = "auto", **kwargs) -> "Message":
        """Create a user message with image attachment."""
        content = [
            TextBlock(text=text),
            ImageBlock(
                image_url=image_url,
                detail=detail
            )
        ]
        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    @classmethod
    def from_attachments(cls, text: str, attachments: List[Union[str, Dict[str, Any]]], **kwargs) -> "Message":
        """Create a user message with multiple attachments."""
        content = [TextBlock(text=text)]
        
        for attachment in attachments:
            if isinstance(attachment, str):
                content.append(ImageBlock(image_url=attachment))
            elif isinstance(attachment, dict):
                attachment_type = attachment.get("type", "image")
                if attachment_type == "pdf":
                    content.append(DocumentBlock.from_url(
                        document_url=attachment["url"],
                        filename=attachment.get("filename")
                    ))
                elif attachment_type == "image":
                    content.append(ImageBlock(
                        image_url=attachment["url"],
                        detail=attachment.get("detail", "auto")
                    ))

        return cls(role=MessageRole.USER, content=content, **kwargs)
    
    
    
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
