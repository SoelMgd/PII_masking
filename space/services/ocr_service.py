#!/usr/bin/env python3
"""
OCR Service for PDF processing using Mistral OCR API.
"""

import os
import sys
import logging
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from mistralai import Mistral

logger = logging.getLogger(__name__)

class OCRService:
    """OCR service for extracting text from PDF documents using Mistral OCR API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.client = Mistral(api_key=self.api_key)
        self.is_initialized = True
        
        logger.info("OCR service initialized with Mistral API")
    
    async def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content using Mistral OCR."""
        try:
            base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
            
            logger.info(f"Processing PDF ({len(pdf_content)} bytes) with Mistral OCR...")
            
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                include_image_base64=False
            )
            
            logger.info("OCR processing completed")
            
            extracted_text = ""
            
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                logger.info(f"Found {len(ocr_response.pages)} pages")
                
                for i, page in enumerate(ocr_response.pages):
                    if hasattr(page, 'markdown') and page.markdown:
                        page_text = page.markdown
                        extracted_text += page_text + "\n\n"
                        logger.debug(f"Page {i+1}: {len(page_text)} characters")
                
                logger.info(f"Total extracted text: {len(extracted_text)} characters")
                
                if not extracted_text.strip():
                    logger.warning("No text extracted from PDF")
                    return "No text could be extracted from this PDF."
                
                return extracted_text.strip()
                
            else:
                logger.warning("No pages found in OCR response")
                return "No text could be extracted from this PDF."
                
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information for monitoring."""
        return {
            "service_type": "ocr",
            "model": "mistral-ocr-latest",
            "initialized": self.is_initialized,
            "description": "Mistral OCR service for PDF text extraction"
        }

async def create_ocr_service(api_key: Optional[str] = None) -> OCRService:
    """Factory function to create OCR service."""
    try:
        service = OCRService(api_key)
        return service
    except Exception as e:
        logger.error(f"Failed to create OCR service: {e}")
        raise

async def test_ocr_service():
    """Test function for development."""
    try:
        service = await create_ocr_service()
        logger.info("OCR service test - service created successfully")
        
        with open("test.pdf", "rb") as f:
            pdf_content = f.read()
            
        text = await service.extract_text_from_pdf(pdf_content)
        logger.info(f"Extracted text length: {len(text)}")
        logger.info(f"First 200 chars: {text[:200]}")
        
    except Exception as e:
        logger.error(f"OCR test failed: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_ocr_service()) 