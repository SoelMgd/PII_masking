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

# Load environment variables
load_dotenv()

from mistralai import Mistral

# Setup logging
logger = logging.getLogger(__name__)

class OCRService:
    """
    OCR service for extracting text from PDF documents using Mistral OCR API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OCR service.
        
        Args:
            api_key: Mistral API key (if None, will read from environment)
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        
        self.client = Mistral(api_key=self.api_key)
        self.is_initialized = True
        
        logger.info("OCR service initialized with Mistral API")
    
    async def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content using Mistral OCR.
        
        Args:
            pdf_content: Raw PDF file content as bytes
            
        Returns:
            Extracted text content
        """
        try:
            # Encode PDF content to base64
            base64_pdf = base64.b64encode(pdf_content).decode('utf-8')
            
            logger.info(f"Processing PDF ({len(pdf_content)} bytes) with Mistral OCR...")
            
            # Process the PDF with OCR
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                include_image_base64=False  # Don't include images to save bandwidth
            )
            
            logger.info("OCR processing completed")
            
            # Extract text from all pages
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
            "service_name": "OCRService",
            "is_initialized": self.is_initialized,
            "api_provider": "Mistral",
            "model": "mistral-ocr-latest",
            "description": "PDF text extraction using Mistral OCR API"
        }

# Factory function for easy initialization
async def create_ocr_service(api_key: Optional[str] = None) -> OCRService:
    """
    Factory function to create and initialize OCR service.
    
    Args:
        api_key: Mistral API key (if None, will read from environment)
        
    Returns:
        Initialized OCRService
    """
    try:
        service = OCRService(api_key)
        logger.info("OCR service created successfully")
        return service
    except Exception as e:
        logger.error(f"Failed to create OCR service: {e}")
        raise 