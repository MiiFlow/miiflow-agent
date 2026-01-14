"""PDF text extraction with OCR support."""

import base64
import io
import logging
import os
import tempfile
from typing import Dict, Any, Optional, List

import fitz
import httpx

from miiflow_agent.utils.url_validator import validate_external_url, URLSecurityError

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_data: str, use_ocr: bool = True, max_size_mb: int = 100) -> Dict[str, Any]:
    """Extract text from PDF with OCR support for scanned documents."""
    
    try:
        if pdf_data.startswith("data:application/pdf;base64,"):
            pdf_bytes = base64.b64decode(pdf_data.split(",")[1])
            
            if len(pdf_bytes) > max_size_mb * 1024 * 1024:
                raise ValueError(f"PDF too large: {len(pdf_bytes)} bytes (max: {max_size_mb}MB)")
                
        elif pdf_data.startswith("data:"):
            raise ValueError("Unsupported data URI format. Expected PDF base64.")
        else:
            is_valid, error_msg = validate_external_url(pdf_data)
            if not is_valid:
                raise URLSecurityError(f"URL security check failed: {error_msg}")
            
            pdf_bytes = b''
            with httpx.stream('GET', pdf_data, timeout=30, follow_redirects=False) as response:
                response.raise_for_status()
                
                content_length = response.headers.get('content-length')
                if content_length:
                    size = int(content_length)
                    if size > max_size_mb * 1024 * 1024:
                        raise ValueError(f"PDF too large: {size} bytes (max: {max_size_mb}MB)")
                
                for chunk in response.iter_bytes(chunk_size=8192):
                    pdf_bytes += chunk
                    if len(pdf_bytes) > max_size_mb * 1024 * 1024:
                        raise ValueError(f"PDF size limit exceeded during download (max: {max_size_mb}MB)")
        
        doc = fitz.open("pdf", pdf_bytes)
        
        text_content = ""
        metadata = {
            "pages": len(doc),
            "extraction_method": "pymupdf",
            "has_images": False,
            "page_texts": [],
            "file_size": len(pdf_bytes),
            "ocr_used": False,
            "text_based_pages": 0,
            "image_based_pages": 0,
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text().strip()
            
            if page_text:
                text_content += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
                metadata["page_texts"].append(page_text)
                metadata["text_based_pages"] += 1
            else:
                metadata["page_texts"].append("")
                metadata["image_based_pages"] += 1
            
            if page.get_images():
                metadata["has_images"] = True
        
        if use_ocr and metadata["image_based_pages"] > 0:
            logger.info(f"Attempting OCR on {metadata['image_based_pages']} image-based pages")
            ocr_results = _extract_with_ocr(doc, pdf_bytes)
            
            if ocr_results["success"]:
                metadata["ocr_used"] = True
                metadata["extraction_method"] = "pymupdf + tesseract_ocr"
                
                for page_num, page_text in enumerate(metadata["page_texts"]):
                    if not page_text and page_num < len(ocr_results["page_texts"]):
                        ocr_text = ocr_results["page_texts"][page_num]
                        if ocr_text:
                            metadata["page_texts"][page_num] = ocr_text
                            text_content += f"--- Page {page_num + 1} (OCR) ---\n{ocr_text}\n\n"
        
        doc.close()
        text_content = text_content.strip()
        
        if not text_content:
            logger.warning("No text content extracted from PDF")
            text_content = "[PDF appears to contain no extractable text]"
            if metadata["has_images"] and not use_ocr:
                text_content += " - PDF contains images but OCR is disabled"
        
        return {
            "text": text_content,
            "metadata": metadata
        }
        
    except Exception as e:
        logger.error(f"PDF text extraction failed: {str(e)}")
        return {
            "text": f"[Error extracting PDF text: {str(e)}]",
            "metadata": {
                "extraction_method": "failed",
                "error": str(e),
                "success": False
            }
        }


def _extract_with_ocr(doc, pdf_bytes: bytes) -> Dict[str, Any]:
    """Extract text using OCR for image-based pages."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        logger.warning(
            "pytesseract and/or Pillow not available for OCR. "
            "Install with: pip install pytesseract pillow"
        )
        return {"success": False, "page_texts": []}
    
    try:
        page_texts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            try:
                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:\-()[]{}\'\"/@#$%^&*+=<>~`| '
                ocr_text = pytesseract.image_to_string(img, config=custom_config)
                page_texts.append(ocr_text.strip())
                
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num + 1}: {str(e)}")
                page_texts.append("")
        
        return {
            "success": True,
            "page_texts": page_texts,
            "method": "tesseract_ocr"
        }
        
    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}")
        return {"success": False, "page_texts": []}


def extract_pdf_text_simple(pdf_data: str) -> str:
    """Simple interface that returns just the extracted text."""
    result = extract_pdf_text(pdf_data)
    return result["text"]


def is_pdf_scanned(pdf_data: str) -> bool:
    """Detect if PDF is primarily scanned/image-based (>50% image pages)."""
    try:
        result = extract_pdf_text(pdf_data, use_ocr=False)
        metadata = result["metadata"]
        
        if metadata.get("pages", 0) == 0:
            return False
            
        image_based_ratio = metadata.get("image_based_pages", 0) / metadata["pages"]
        return image_based_ratio > 0.5
        
    except Exception:
        return False


def extract_pdf_metadata(pdf_data: str) -> Dict[str, Any]:
    """Extract PDF metadata only."""
    try:
        import fitz
        
        if pdf_data.startswith("data:application/pdf;base64,"):
            pdf_bytes = base64.b64decode(pdf_data.split(",")[1])
        else:
            response = httpx.get(pdf_data, timeout=30)
            pdf_bytes = response.content
        
        doc = fitz.open("pdf", pdf_bytes)
        
        metadata = {
            "pages": len(doc),
            "file_size": len(pdf_bytes),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
        }
        
        doc.close()
        return metadata
        
    except Exception as e:
        logger.error(f"PDF metadata extraction failed: {str(e)}")
        return {"error": str(e)}


def extract_pdf_chunks(
    pdf_data: str, 
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    chunk_strategy: str = "smart",
    use_ocr: bool = True
) -> Dict[str, Any]:
    """Extract PDF text as manageable chunks for LLM processing using smart chunking."""
    extraction_result = extract_pdf_text(pdf_data, use_ocr=use_ocr)

    if extraction_result["text"].startswith("[Error"):
        return {
            "chunks": [],
            "metadata": extraction_result["metadata"],
            "chunk_info": {
                "total_chunks": 0,
                "chunk_strategy": "smart",
                "error": "PDF extraction failed"
            }
        }
    
    full_text = extraction_result["text"]
    chunks = _smart_chunk_text(full_text, chunk_size, chunk_overlap)
    
    chunk_info = {
        "total_chunks": len(chunks),
        "chunk_strategy": "smart",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "avg_chunk_length": sum(len(c["text"]) for c in chunks) / len(chunks) if chunks else 0,
        "total_characters": len(full_text)
    }
    
    return {
        "chunks": chunks,
        "metadata": extraction_result["metadata"],
        "chunk_info": chunk_info
    }


def _smart_chunk_text(text: str, chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    """Smart chunking that respects sentence and paragraph boundaries."""
    chunks = []
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    current_chunk = ""
    current_pages = []
    chunk_index = 0

    for paragraph in paragraphs:
        page_num = _extract_page_number(paragraph)
        if page_num:
            current_pages.append(page_num)

        if current_chunk and len(current_chunk) + len(paragraph) > chunk_size:
            chunks.append({
                "text": current_chunk.strip(),
                "chunk_index": chunk_index,
                "page_numbers": list(set(current_pages)) if current_pages else [],
                "chunk_type": "smart",
                "char_count": len(current_chunk)
            })

            if overlap > 0 and len(current_chunk) > overlap:
                overlap_text = current_chunk[-overlap:]
                sentence_end = overlap_text.rfind('. ')
                if sentence_end > overlap // 2:
                    overlap_text = overlap_text[sentence_end + 2:]

                current_chunk = overlap_text + "\n\n" + paragraph
                # Reset current_pages and extract page numbers from overlap + new paragraph
                current_pages = _extract_page_numbers_from_text(overlap_text)
                if page_num:
                    current_pages.append(page_num)
            else:
                current_chunk = paragraph
                current_pages = [page_num] if page_num else []

            chunk_index += 1
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

    if current_chunk.strip():
        chunks.append({
            "text": current_chunk.strip(),
            "chunk_index": chunk_index,
            "page_numbers": list(set(current_pages)) if current_pages else [],
            "chunk_type": "smart",
            "char_count": len(current_chunk)
        })

    return chunks




def _extract_page_number(text: str) -> Optional[int]:
    """Extract page number from text like '--- Page 5 ---'."""
    import re
    match = re.search(r'--- Page (\d+) ---', text)
    return int(match.group(1)) if match else None


def _extract_page_numbers_from_text(text: str) -> List[int]:
    """Extract all page numbers from a text chunk."""
    import re
    matches = re.findall(r'--- Page (\d+) ---', text)
    return [int(match) for match in matches]


def chunk_pdf_for_llm(
    pdf_data: str,
    max_tokens: int = 3000,
    model: str = "gpt-4"
) -> Dict[str, Any]:
    """Chunk PDF optimized for specific LLM token limits."""
    chars_per_token = 4
    if model.startswith("gpt"):
        chars_per_token = 4
    elif model.startswith("claude"):
        chars_per_token = 3.5
    elif model.startswith("gemini"):
        chars_per_token = 3
    
    chunk_size = int(max_tokens * chars_per_token)
    overlap = min(200, chunk_size // 10)
    
    return extract_pdf_chunks(
        pdf_data=pdf_data,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        chunk_strategy="smart"
    )
