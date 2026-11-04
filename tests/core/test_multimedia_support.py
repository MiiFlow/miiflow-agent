"""Tests for multimedia support (images and PDFs)."""

import base64
import pytest
from unittest.mock import Mock, patch
from dataclasses import asdict

# Skip if PyMuPDF is not installed
pymupdf = pytest.importorskip("pymupdf", reason="PyMuPDF not installed")

from miiflow_agent.core.message import (
    Message,
    MessageRole,
    TextBlock,
    ImageBlock,
    DocumentBlock,
    ContentBlock
)
from miiflow_agent.utils.pdf_extractor import (
    extract_pdf_text,
    extract_pdf_metadata,
)


A4_POINTS = (595, 842)
TYPED_TEXT = "Revenue grew 12 percent."
SCANNED_TEXT = "Invoice 4471 paid"


def build_pdf_data_uri() -> str:
    """A two-page A4 PDF built with PyMuPDF itself, so the integration tests
    run against the installed library instead of skipping for a missing
    fixture.

    Page 1 carries selectable text. Page 2 is a scan: a raster of rendered
    text with no text layer, the case that sends extraction down the OCR path.
    """
    width, height = A4_POINTS
    scan_source = pymupdf.open()
    scan_page = scan_source.new_page(width=width, height=height)
    scan_page.insert_text((72, 144), SCANNED_TEXT, fontsize=36)
    scan = scan_page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    scan_source.close()

    doc = pymupdf.open()
    doc.set_metadata({"title": "Quarterly plan", "author": "Ada"})
    text_page = doc.new_page(width=width, height=height)
    text_page.insert_text((72, 72), TYPED_TEXT)
    image_page = doc.new_page(width=width, height=height)
    image_page.insert_image(image_page.rect, pixmap=scan)
    pdf_bytes = doc.tobytes()
    doc.close()
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
        
        msg = Message.from_attachments(
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
    
    @patch('pymupdf.open')
    def test_pdf_metadata_extraction(self, mock_open):
        """Test PDF metadata extraction."""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=3)
        mock_doc.metadata = {
            'title': 'Test Document',
            'author': 'Test Author',
            'subject': 'Test Subject'
        }
        mock_open.return_value = mock_doc
        
        from miiflow_agent.utils.pdf_extractor import extract_pdf_metadata
        
        with patch('base64.b64decode') as mock_b64:
            mock_b64.return_value = b'mock_pdf_bytes'
            
            metadata = extract_pdf_metadata("data:application/pdf;base64,JVBERi0x...")
            
            assert metadata['pages'] == 3
            assert metadata['title'] == 'Test Document'
            assert metadata['author'] == 'Test Author'
            assert metadata['subject'] == 'Test Subject'
    
    @patch('miiflow_agent.utils.pdf_extractor.extract_pdf_text')
    def test_pdf_chunking(self, mock_extract):
        """Test PDF chunking functionality."""
        mock_extract.return_value = {
            "text": "Page 1 content\n\nPage 2 content\n\nPage 3 content",
            "metadata": {
                "pages": 3,
                "page_texts": ["Page 1 content", "Page 2 content", "Page 3 content"]
            }
        }

        from miiflow_agent.utils.pdf_extractor import extract_pdf_chunks

        result = extract_pdf_chunks(
            pdf_data="mock_pdf",
            chunk_size=20,
            chunk_strategy="smart"  # Now always uses smart chunking
        )

        assert result["chunk_info"]["total_chunks"] == 3
        assert result["chunk_info"]["chunk_strategy"] == "smart"  # Always smart now
        assert len(result["chunks"]) == 3

        # Check chunk structure
        chunk = result["chunks"][0]
        assert "text" in chunk
        assert "chunk_index" in chunk
        assert "page_numbers" in chunk
        assert "chunk_type" in chunk
    
    def test_llm_optimized_chunking(self):
        """Test LLM-optimized chunking with token estimation."""
        from miiflow_agent.utils.pdf_extractor import chunk_pdf_for_llm
        
        with patch('miiflow_agent.utils.pdf_extractor.extract_pdf_chunks') as mock_chunks:
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
    """Integration tests that run the installed PyMuPDF on a real PDF."""

    def test_real_pdf_extraction(self):
        pdf_data_uri = build_pdf_data_uri()

        result = extract_pdf_text(pdf_data_uri, use_ocr=False)

        metadata = result["metadata"]
        assert TYPED_TEXT in result["text"]
        assert metadata["pages"] == 2
        assert metadata["extraction_method"] == "pymupdf"
        assert metadata["text_based_pages"] == 1
        assert metadata["image_based_pages"] == 1
        assert metadata["has_images"] is True
        assert metadata["ocr_used"] is False

        pdf_metadata = extract_pdf_metadata(pdf_data_uri)

        assert pdf_metadata["pages"] == 2
        assert pdf_metadata["title"] == "Quarterly plan"
        assert pdf_metadata["author"] == "Ada"
        assert pdf_metadata["file_size"] > 0

    def test_ocr_renders_image_only_pages(self):
        """The OCR path rasterizes each page with PyMuPDF and hands Pillow a
        PNG. The OCR engine is stubbed so this runs without tesseract."""
        pytest.importorskip("PIL", reason="Pillow not installed")
        from PIL import Image

        rendered_sizes = []

        def image_to_string(image, config=""):
            assert isinstance(image, Image.Image)
            rendered_sizes.append(image.size)
            return "stub ocr text"

        fake_tesseract = Mock(image_to_string=image_to_string)
        with patch.dict("sys.modules", {"pytesseract": fake_tesseract}):
            result = extract_pdf_text(build_pdf_data_uri(), use_ocr=True)

        metadata = result["metadata"]
        assert metadata["ocr_used"] is True
        assert metadata["extraction_method"] == "pymupdf + tesseract_ocr"
        # OCR fills only the page that had no text layer.
        assert metadata["page_texts"] == [TYPED_TEXT, "stub ocr text"]
        assert "--- Page 2 (OCR) ---\nstub ocr text" in result["text"]
        # Pages are rendered at Matrix(2, 2).
        doubled = (A4_POINTS[0] * 2, A4_POINTS[1] * 2)
        assert rendered_sizes == [doubled, doubled]

    def test_ocr_functionality(self):
        """End to end with the real tesseract binary, when it is installed."""
        pytesseract = pytest.importorskip("pytesseract", reason="pytesseract not installed")
        pytest.importorskip("PIL", reason="Pillow not installed")
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            pytest.skip("Tesseract OCR not installed or not in PATH")

        result = extract_pdf_text(build_pdf_data_uri(), use_ocr=True)

        metadata = result["metadata"]
        assert metadata["ocr_used"] is True
        assert metadata["page_texts"][0] == TYPED_TEXT
        # Compared without whitespace: tesseract drops word spaces under the
        # extractor's character whitelist.
        assert SCANNED_TEXT.replace(" ", "") in "".join(metadata["page_texts"][1].split())


if __name__ == "__main__":
    pytest.main([__file__])
