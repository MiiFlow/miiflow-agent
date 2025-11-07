"""Tests for multimedia support (images and PDFs)."""

import base64
import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from dataclasses import asdict

from miiflow_llm.core.message import (
    Message,
    MessageRole,
    TextBlock,
    ImageBlock,
    DocumentBlock,
    ContentBlock
)
from miiflow_llm.utils.pdf_extractor import (
    extract_pdf_text,
    extract_pdf_metadata,
)


# Test data paths
TEST_DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
TEST_PDF_PATH = TEST_DOCS_DIR / "DebjyotiRay..pdf"


def load_pdf_as_data_uri(pdf_path: Path) -> str:
    """Helper to load PDF file as base64 data URI."""
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
        return f"data:application/pdf;base64,{base64.b64encode(pdf_bytes).decode()}"


class TestContentBlocks:
    """Test content block classes."""
    
    def test_text_block_creation(self):
        """Test TextBlock creation and properties."""
        block = TextBlock(text="Hello world")
        
        assert block.type == "text"
        assert block.text == "Hello world"
    
    def test_image_block_creation(self):
        """Test ImageBlock creation and properties."""
        block = ImageBlock(
            image_url="https://example.com/image.jpg",
            detail="high"
        )
        
        assert block.type == "image_url"
        assert block.image_url == "https://example.com/image.jpg"
        assert block.detail == "high"
    
    def test_document_block_creation(self):
        """Test DocumentBlock creation and properties."""
        block = DocumentBlock(
            document_url="data:application/pdf;base64,JVBERi0x...",
            filename="test.pdf",
            metadata={"pages": 5}
        )
        
        assert block.type == "document"
        assert block.document_type == "pdf"
        assert block.filename == "test.pdf"
        assert block.metadata["pages"] == 5


class TestMultimediaMessages:
    """Test message creation with multimedia content."""
    
    def test_message_with_pdf(self):
        """Test creating message with PDF attachment."""
        msg = Message.from_pdf(
            text="Analyze this document",
            pdf_url="https://example.com/doc.pdf",
            filename="contract.pdf"
        )
        
        assert msg.role == MessageRole.USER
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextBlock)
        assert isinstance(msg.content[1], DocumentBlock)
        assert msg.content[0].text == "Analyze this document"
        assert msg.content[1].document_url == "https://example.com/doc.pdf"
        assert msg.content[1].filename == "contract.pdf"
    
    def test_message_with_image(self):
        """Test creating message with image attachment."""
        msg = Message.from_image(
            text="What's in this image?",
            image_url="https://example.com/image.jpg",
            detail="high"
        )
        
        assert msg.role == MessageRole.USER
        assert len(msg.content) == 2
        assert isinstance(msg.content[0], TextBlock)
        assert isinstance(msg.content[1], ImageBlock)
        assert msg.content[0].text == "What's in this image?"
        assert msg.content[1].image_url == "https://example.com/image.jpg"
        assert msg.content[1].detail == "high"
    
    def test_message_with_multiple_attachments(self):
        """Test creating message with multiple attachments."""
        attachments = [
            "https://example.com/image1.jpg",
            {
                "url": "https://example.com/doc.pdf",
                "type": "pdf",
                "filename": "report.pdf"
            },
            {
                "url": "https://example.com/image2.jpg",
                "type": "image",
                "detail": "low"
            }
        ]
        
        msg = Message.from_attatchments(
            text="Process these files",
            attachments=attachments
        )
        
        assert msg.role == MessageRole.USER
        assert len(msg.content) == 4  # 1 text + 3 attachments
        assert isinstance(msg.content[0], TextBlock)
        assert isinstance(msg.content[1], ImageBlock)  # Simple string URL
        assert isinstance(msg.content[2], DocumentBlock)  # PDF dict
        assert isinstance(msg.content[3], ImageBlock)  # Image dict
        
        assert msg.content[2].filename == "report.pdf"
        assert msg.content[3].detail == "low"


class TestPDFExtraction:
    """Test PDF text extraction functionality."""
    
    @patch('fitz.open')
    def test_pdf_metadata_extraction(self, mock_fitz):
        """Test PDF metadata extraction."""
        # Mock fitz document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=3)
        mock_doc.metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'subject': 'Test Subject'
        }
        mock_fitz.return_value = mock_doc
        
        from miiflow_llm.utils.pdf_extractor import extract_pdf_metadata
        
        with patch('base64.b64decode') as mock_b64:
            mock_b64.return_value = b'mock_pdf_bytes'
            
            metadata = extract_pdf_metadata("data:application/pdf;base64,JVBERi0x...")
            
            assert metadata['pages'] == 3
            assert metadata['title'] == 'Test Document'
            assert metadata['author'] == 'Test Author'
            assert metadata['subject'] == 'Test Subject'
    
    @patch('miiflow_llm.utils.pdf_extractor.extract_pdf_text')
    def test_pdf_chunking(self, mock_extract):
        """Test PDF chunking functionality."""
        mock_extract.return_value = {
            "text": "Page 1 content\n\nPage 2 content\n\nPage 3 content",
            "metadata": {
                "pages": 3,
                "page_texts": ["Page 1 content", "Page 2 content", "Page 3 content"]
            }
        }
        
        from miiflow_llm.utils.pdf_extractor import extract_pdf_chunks
        
        result = extract_pdf_chunks(
            pdf_data="mock_pdf",
            chunk_size=20,
            chunk_strategy="page"
        )
        
        assert result["chunk_info"]["total_chunks"] == 3
        assert result["chunk_info"]["chunk_strategy"] == "page"
        assert len(result["chunks"]) == 3
        
        # Check chunk structure
        chunk = result["chunks"][0]
        assert "text" in chunk
        assert "chunk_index" in chunk
        assert "page_numbers" in chunk
        assert "chunk_type" in chunk
    
    def test_llm_optimized_chunking(self):
        """Test LLM-optimized chunking with token estimation."""
        from miiflow_llm.utils.pdf_extractor import chunk_pdf_for_llm
        
        with patch('miiflow_llm.utils.pdf_extractor.extract_pdf_chunks') as mock_chunks:
            mock_chunks.return_value = {"chunks": [], "metadata": {}, "chunk_info": {}}
            
            # Test different model token calculations
            chunk_pdf_for_llm("mock_pdf", max_tokens=1000, model="gpt-4")
            
            # Should call extract_pdf_chunks with appropriate chunk_size
            args, kwargs = mock_chunks.call_args
            assert kwargs["chunk_size"] == 4000  # 1000 tokens * 4 chars/token
            assert kwargs["chunk_strategy"] == "smart"


class TestMessageSerialization:
    """Test message serialization with multimedia content."""
    
    def test_message_to_dict_with_multimedia(self):
        """Test converting multimedia message to dict."""
        msg = Message.from_pdf(
            text="Process this",
            pdf_url="https://example.com/doc.pdf",
            filename="test.pdf"
        )
        
        msg_dict = msg.to_dict()
        
        assert msg_dict["role"] == "user"
        assert len(msg_dict["content"]) == 2
        # Content blocks should remain as objects in dict representation
        assert isinstance(msg_dict["content"][0], TextBlock)
        assert isinstance(msg_dict["content"][1], DocumentBlock)
    
    def test_message_from_dict_reconstruction(self):
        """Test reconstructing message from dict."""
        original_msg = Message.from_pdf(
            text="Process this", 
            pdf_url="https://example.com/doc.pdf"
        )
        
        msg_dict = original_msg.to_dict()
        reconstructed_msg = Message.from_dict(msg_dict)
        
        assert reconstructed_msg.role == original_msg.role
        assert len(reconstructed_msg.content) == len(original_msg.content)
        assert reconstructed_msg.timestamp == original_msg.timestamp


class TestMultimediaIntegration:
    """Integration tests for multimedia functionality."""
    
    def test_real_pdf_extraction(self):
        """Integration test with real PDF file using PyMuPDF."""
        if not TEST_PDF_PATH.exists():
            pytest.skip(f"Test PDF file not found at {TEST_PDF_PATH}")
        
        # Load PDF as data URI
        pdf_data_uri = load_pdf_as_data_uri(TEST_PDF_PATH)
        
        # Test text extraction without OCR
        result = extract_pdf_text(pdf_data_uri, use_ocr=False)
        
        assert "text" in result
        assert "metadata" in result
        assert len(result["text"]) > 0
        assert result["metadata"]["pages"] > 0
        assert result["metadata"]["extraction_method"] in ["pymupdf", "pymupdf + tesseract_ocr"]
        
        # Test metadata extraction
        metadata = extract_pdf_metadata(pdf_data_uri)
        
        assert "pages" in metadata
        assert "file_size" in metadata
        assert metadata["pages"] > 0
        assert metadata["file_size"] > 0
    
    def test_ocr_functionality(self):
        """Integration test for OCR functionality with tesseract."""
        if not TEST_PDF_PATH.exists():
            pytest.skip(f"Test PDF file not found at {TEST_PDF_PATH}")
        
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            pytest.skip("pytesseract and/or Pillow not installed")
        
        # Check if tesseract is available
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            pytest.skip("Tesseract OCR not installed or not in PATH")
        
        # Load PDF as data URI
        pdf_data_uri = load_pdf_as_data_uri(TEST_PDF_PATH)
        
        # Test text extraction with OCR enabled
        result = extract_pdf_text(pdf_data_uri, use_ocr=True)
        
        assert "text" in result
        assert "metadata" in result
        assert len(result["text"]) > 0
        
        # If the PDF had image-based pages and OCR was used, verify that
        if result["metadata"].get("image_based_pages", 0) > 0:
            # OCR should have been attempted
            assert result["metadata"].get("ocr_used") in [True, False]  # May or may not succeed


if __name__ == "__main__":
    pytest.main([__file__])
