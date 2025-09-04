#!/usr/bin/env python3
"""
FastAPI application for PII Masking Demo - HuggingFace Space.

Simple version using only Mistral Prompting service.
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Import our inference services
from services.mistral_prompting import create_mistral_service, MistralPromptingService
from services.bert_classif import create_bert_service, BERTInferenceService
from services.ocr_service import create_ocr_service, OCRService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
mistral_base_service: MistralPromptingService = None
mistral_finetuned_service: MistralPromptingService = None
bert_service: BERTInferenceService = None
ocr_service: OCRService = None

# Model configurations
MODELS = {
    "base": "mistral-large-latest",
    "finetuned": "ft:ministral-8b-latest:c6d4dfa8:20250831:pii-1e-4-200:57d93df9"
}

# BERT model path - HuggingFace Hub repository
BERT_MODEL_PATH = "SoelMgd/bert-pii-detection"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global mistral_base_service, mistral_finetuned_service, bert_service, ocr_service
    
    # Startup
    logger.info("Starting PII Masking Demo application...")
    
    try:
        # Initialize base Mistral service
        logger.info("Initializing base Mistral service...")
        mistral_base_service = await create_mistral_service(model_name=MODELS["base"])
        logger.info("Base Mistral service initialized successfully")
        
        # Initialize fine-tuned Mistral service
        logger.info("Initializing fine-tuned Mistral service...")
        mistral_finetuned_service = await create_mistral_service(model_name=MODELS["finetuned"])
        logger.info("Fine-tuned Mistral service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Mistral services: {e}")
        # Don't raise exception - let app start but handle gracefully in endpoints
    
    try:
        # Initialize BERT service
        logger.info("Initializing BERT service...")
        bert_service = await create_bert_service(model_path=BERT_MODEL_PATH)
        logger.info("BERT service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize BERT service: {e}")
        # Don't raise exception - let app start but handle gracefully in endpoints
    
    try:
        # Initialize OCR service
        logger.info("Initializing OCR service...")
        ocr_service = await create_ocr_service()
        logger.info("OCR service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize OCR service: {e}")
        # Don't raise exception - let app start but handle gracefully in endpoints
        
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# Create FastAPI app
app = FastAPI(
    title="🔒 PII Masking Demo",
    description="Personal Identifiable Information masking using Mistral AI",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class PredictionRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for PII", min_length=1, max_length=5000)
    method: str = Field(default="mistral", description="Method to use: 'mistral' or 'bert'")
    model: str = Field(default="base", description="Model to use: 'base' for mistral-large-latest or 'finetuned' for fine-tuned model (ignored for BERT)")
    pii_entities: List[str] = Field(default=[], description="List of PII entity types to mask (empty list means mask all detected entities)")

class PredictionResponse(BaseModel):
    masked_text: str = Field(description="Text with PII entities masked")
    entities: Dict[str, list[str]] = Field(description="Detected PII entities")
    processing_time: float = Field(description="Processing time in seconds")
    method_used: str = Field(description="Method used for prediction")
    num_entities: int = Field(description="Total number of entities found")
    selected_entities: List[str] = Field(description="List of entity types that were selected for masking")

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, Any]
    timestamp: float

# Helper function to get the appropriate service
def get_mistral_service(model: str) -> MistralPromptingService:
    """Get the appropriate Mistral service based on model type."""
    if model == "base":
        if mistral_base_service is None:
            raise HTTPException(
                status_code=503, 
                detail="Base Mistral service not available. Please check API key configuration."
            )
        return mistral_base_service
    elif model == "finetuned":
        if mistral_finetuned_service is None:
            raise HTTPException(
                status_code=503, 
                detail="Fine-tuned Mistral service not available. Please check API key configuration."
            )
        return mistral_finetuned_service
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model}' not supported. Use 'base' or 'finetuned'."
        )

# Mount static files for frontend
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    try:
        return FileResponse("static/index.html")
    except Exception:
        # Fallback HTML if static files not available
        return HTMLResponse("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PII Masking Demo</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
                textarea { width: 100%; height: 100px; margin: 10px 0; padding: 10px; }
                button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
                .result { margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔒 PII Masking Demo</h1>
                <p>Enter text below to detect and mask Personal Identifiable Information:</p>
                
                <textarea id="inputText" placeholder="Enter your text here... (e.g., Hi, my name is John Smith and my email is john.smith@company.com)"></textarea>
                <br>
                <button onclick="processText()">Process Text</button>
                
                <div id="result" class="result" style="display:none;">
                    <h3>Results:</h3>
                    <p><strong>Masked Text:</strong> <span id="maskedText"></span></p>
                    <p><strong>Entities Found:</strong> <span id="entities"></span></p>
                    <p><strong>Processing Time:</strong> <span id="processingTime"></span>s</p>
                </div>
            </div>
            
            <script>
                async function processText() {
                    const text = document.getElementById('inputText').value;
                    if (!text.trim()) {
                        alert('Please enter some text');
                        return;
                    }
                    
                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ text: text, method: 'mistral' })
                        });
                        
                        const result = await response.json();
                        
                        if (response.ok) {
                            document.getElementById('maskedText').textContent = result.masked_text;
                            document.getElementById('entities').textContent = JSON.stringify(result.entities, null, 2);
                            document.getElementById('processingTime').textContent = result.processing_time.toFixed(3);
                            document.getElementById('result').style.display = 'block';
                        } else {
                            alert('Error: ' + result.detail);
                        }
                    } catch (error) {
                        alert('Error: ' + error.message);
                    }
                }
            </script>
        </body>
        </html>
        """)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict PII entities and return masked text.
    
    Supports Mistral models (base and fine-tuned) and BERT.
    """
    # Validate method
    if request.method not in ["mistral", "bert"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Method '{request.method}' not supported. Use 'mistral' or 'bert'."
        )
    
    start_time = time.time()
    
    try:
        if request.method == "mistral":
            # Get the appropriate Mistral service
            service = get_mistral_service(request.model)
            model_type = "Fine-tuned" if request.model == "finetuned" else "Base"
            logger.info(f"🔍 Processing text with {model_type} Mistral model: {request.text[:100]}...")
            
            # Call Mistral service
            prediction = await service.predict(request.text, request.pii_entities)
            method_used = f"{request.method}-{request.model}"
            
        elif request.method == "bert":
            # Check BERT service availability
            if bert_service is None:
                raise HTTPException(
                    status_code=503, 
                    detail="BERT service not available. Please check model configuration."
                )
            
            logger.info(f"🔍 Processing text with BERT model: {request.text[:100]}...")
            
            # Call BERT service
            prediction = await bert_service.predict(request.text, request.pii_entities)
            method_used = "bert"
        
        processing_time = time.time() - start_time
        
        # Count total entities
        num_entities = sum(len(entities) for entities in prediction.entities.values())
        
        logger.info(f"Prediction completed in {processing_time:.3f}s - found {num_entities} entities")
        
        return PredictionResponse(
            masked_text=prediction.masked_text,
            entities=prediction.entities,
            processing_time=processing_time,
            method_used=method_used,
            num_entities=num_entities,
            selected_entities=request.pii_entities
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-pdf", response_model=PredictionResponse)
async def predict_pdf(
    file: UploadFile = File(...),
    method: str = "mistral",
    model: str = "base",
    pii_entities: str = "[]"
):
    """
    Extract text from PDF using OCR, then predict PII entities and return masked text.
    
    Supports the same methods as /predict: Mistral (base/fine-tuned) and BERT.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Check OCR service availability
    if ocr_service is None:
        raise HTTPException(
            status_code=503,
            detail="OCR service not available. Please check API key configuration."
        )
    
    # Validate method
    if method not in ["mistral", "bert"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Method '{method}' not supported. Use 'mistral' or 'bert'."
        )
    
    try:
        # Parse PII entities list
        import json
        pii_entities_list = json.loads(pii_entities) if pii_entities else []
        
        start_time = time.time()
        
        # Read PDF content
        pdf_content = await file.read()
        logger.info(f"Received PDF file: {file.filename} ({len(pdf_content)} bytes)")
        
        # Extract text using OCR
        logger.info("Extracting text from PDF using Mistral OCR...")
        extracted_text = await ocr_service.extract_text_from_pdf(pdf_content)
        
        if not extracted_text or len(extracted_text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Could not extract sufficient text from PDF"
            )
        
        logger.info(f"Extracted {len(extracted_text)} characters from PDF")
        
        # Now process the extracted text with the selected method
        if method == "mistral":
            # Get the appropriate Mistral service
            service = get_mistral_service(model)
            prediction = await service.predict(extracted_text, pii_entities_list)
            method_used = f"{method}-{model}"
            
        elif method == "bert":
            # Check BERT service availability
            if bert_service is None:
                raise HTTPException(
                    status_code=503, 
                    detail="BERT service not available. Please check model configuration."
                )
            
            prediction = await bert_service.predict(extracted_text, pii_entities_list)
            method_used = "bert"
        
        processing_time = time.time() - start_time
        
        # Count total entities
        num_entities = sum(len(entities) for entities in prediction.entities.values())
        
        logger.info(f"PDF processing completed in {processing_time:.3f}s - found {num_entities} entities")
        
        return PredictionResponse(
            masked_text=prediction.masked_text,
            entities=prediction.entities,
            processing_time=processing_time,
            method_used=f"pdf-{method_used}",
            num_entities=num_entities,
            selected_entities=pii_entities_list
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global mistral_base_service, mistral_finetuned_service, bert_service, ocr_service
    
    services_status = {
        "mistral_base": {
            "available": mistral_base_service is not None,
            "initialized": mistral_base_service.is_initialized if mistral_base_service else False,
            "model": MODELS["base"],
            "info": mistral_base_service.get_service_info() if mistral_base_service else None
        },
        "mistral_finetuned": {
            "available": mistral_finetuned_service is not None,
            "initialized": mistral_finetuned_service.is_initialized if mistral_finetuned_service else False,
            "model": MODELS["finetuned"],
            "info": mistral_finetuned_service.get_service_info() if mistral_finetuned_service else None
        },
        "bert": {
            "available": bert_service is not None,
            "initialized": bert_service.is_initialized if bert_service else False,
            "model": BERT_MODEL_PATH,
            "info": bert_service.get_service_info() if bert_service else None
        },
        "ocr": {
            "available": ocr_service is not None,
            "initialized": ocr_service.is_initialized if ocr_service else False,
            "model": "mistral-ocr-latest",
            "info": ocr_service.get_service_info() if ocr_service else None
        }
    }
    
    # Overall status
    base_healthy = mistral_base_service and mistral_base_service.is_initialized
    finetuned_healthy = mistral_finetuned_service and mistral_finetuned_service.is_initialized
    bert_healthy = bert_service and bert_service.is_initialized
    ocr_healthy = ocr_service and ocr_service.is_initialized
    
    healthy_services = sum([base_healthy, finetuned_healthy, bert_healthy, ocr_healthy])
    
    if healthy_services == 4:
        overall_status = "healthy"
    elif healthy_services >= 2:
        overall_status = "partial"
    else:
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        services=services_status,
        timestamp=time.time()
    )

@app.get("/api/info")
async def api_info():
    """Get API information."""
    return {
        "name": "PII Masking Demo API",
        "version": "1.0.0",
        "description": "Personal Identifiable Information masking using Mistral AI",
        "available_methods": ["mistral", "bert"],
        "available_models": {
            "base": {
                "name": MODELS["base"],
                "description": "Base Mistral model with detailed prompting"
            },
            "finetuned": {
                "name": MODELS["finetuned"],
                "description": "Fine-tuned Mistral model specialized for PII detection"
            },
            "bert": {
                "name": BERT_MODEL_PATH,
                "description": "BERT token classification model for fast PII detection"
            }
        },
        "endpoints": {
            "predict": "POST /predict - Analyze text for PII (supports 'model' parameter: 'base' or 'finetuned')",
            "health": "GET /health - Health check",
            "info": "GET /api/info - API information"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": f"Endpoint {request.url.path} not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Add CORS middleware for development
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    
    # For local development
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=True,
        log_level="info"
    ) 