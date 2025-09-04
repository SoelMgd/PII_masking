#!/usr/bin/env python3
"""
Service setup for performance benchmarks.
Initializes BERT and Mistral services with local model loading.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "space"))

from services.bert_classif import BERTInferenceService
from services.mistral_prompting import MistralPromptingService
from services.base_service import BasePIIInferenceService

logger = logging.getLogger(__name__)

class BenchmarkBERTService(BERTInferenceService):
    """BERT service modified to load from local model directory."""
    
    def __init__(self, local_model_path: str, config: Dict = None):
        # Override the model_path to use local directory
        super().__init__(local_model_path, config)
        self.local_model_path = local_model_path
        
    async def initialize(self) -> bool:
        """Initialize the model from local directory."""
        try:
            logger.info(f"Loading BERT model from local path: {self.local_model_path}")
            
            # Import here to avoid issues
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import torch
            
            # Load from local directory instead of HuggingFace Hub
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.local_model_path,
                torch_dtype=torch.float32,
                trust_remote_code=False,
                local_files_only=True  # Force local loading
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.local_model_path,
                trust_remote_code=False,
                local_files_only=True  # Force local loading
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            torch.set_num_threads(2)
            
            self.is_initialized = True
            
            logger.info("BERT model loaded successfully from local directory")
            logger.info(f"Model parameters: {self.model.num_parameters():,}")
            logger.info(f"Number of labels: {self.model.config.num_labels}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT model from local path: {e}")
            return False

async def create_bert_service(model_path: str = None, config: Dict = None) -> BenchmarkBERTService:
    """
    Create and initialize BERT service for benchmarking.
    
    Args:
        model_path: Path to local BERT model directory
        config: Configuration dictionary
        
    Returns:
        Initialized BERT service
    """
    if model_path is None:
        model_path = "/Users/twin/Documents/pii-masking-200k/models/bert_classic_token_classif"
    
    if config is None:
        config = {
            'max_length': 512,
            'batch_size': 4
        }
    
    service = BenchmarkBERTService(model_path, config)
    
    if not await service.initialize():
        raise RuntimeError("Failed to initialize BERT service")
    
    logger.info("BERT service ready for benchmarking")
    return service

async def create_mistral_service(
    api_key: Optional[str] = None, 
    model_name: str = "mistral-large-latest",
    enable_batching: bool = True
) -> MistralPromptingService:
    """
    Create and initialize Mistral service for benchmarking.
    
    Args:
        api_key: Mistral API key (if None, reads from environment)
        model_name: Model to use
        enable_batching: Whether to enable batch processing
        
    Returns:
        Initialized Mistral service
    """
    if api_key is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not provided and not found in environment variables")
    
    service = MistralPromptingService(api_key, model_name, enable_batching)
    
    if not await service.initialize():
        raise RuntimeError("Failed to initialize Mistral service")
    
    logger.info(f"Mistral service ready for benchmarking (model: {model_name})")
    return service

async def create_services_for_benchmark() -> Dict[str, BasePIIInferenceService]:
    """
    Create all services needed for benchmarking.
    
    Returns:
        Dictionary of initialized services
    """
    services = {}
    
    try:
        # Load BERT service
        logger.info("Initializing BERT service...")
        services['bert'] = await create_bert_service()
        
        # Load Mistral services
        logger.info("Initializing Mistral services...")
        services['mistral_large'] = await create_mistral_service(
            model_name="mistral-large-latest",
            enable_batching=True
        )
        
        # Check if fine-tuned model is available
        fine_tuned_model = os.getenv("MISTRAL_FINE_TUNED_MODEL")
        if fine_tuned_model:
            services['mistral_finetuned'] = await create_mistral_service(
                model_name=fine_tuned_model,
                enable_batching=True
            )
        else:
            logger.warning("MISTRAL_FINE_TUNED_MODEL not found in environment, skipping fine-tuned model")
        
        logger.info(f"Successfully initialized {len(services)} services for benchmarking")
        return services
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

def get_test_texts() -> Dict[str, str]:
    """
    Generate test texts of different lengths for benchmarking.
    
    Returns:
        Dictionary of test texts with different characteristics
    """

    text_3k = """CONFIDENTIAL EMPLOYEE RECORD
        
        Personal Information:
        - Full Name: Alexander Michael Johnson
        - Date of Birth: March 15, 1985
        - Social Security Number: 123-45-6789
        - Phone Number: (555) 123-4567
        - Email Address: alex.johnson@company.com
        - Home Address: 1234 Elm Street, Apartment 5B, Springfield, IL 62701
        - Emergency Contact: Maria Johnson (spouse) - (555) 987-6543
        
        Employment Details:
        - Employee ID: EMP-2023-001
        - Job Title: Senior Software Engineer
        - Department: Engineering
        - Manager: Jennifer Smith (jennifer.smith@company.com)
        - Start Date: January 10, 2020
        - Salary: $95,000 annually
        - Direct Phone: (555) 234-5678
        - Office Location: Building A, Floor 3, Cube 42
        
        Banking Information:
        - Bank Name: First National Bank
        - Account Number: 1234567890
        - Routing Number: 987654321
        - Account Type: Checking
        
        Medical Information:
        - Primary Care Physician: Dr. Robert Wilson
        - Medical Record Number: MRN-789456
        - Insurance Provider: HealthPlus Insurance
        - Insurance ID: HP-123456789
        - Blood Type: O+
        - Allergies: Penicillin, Shellfish
        
        Additional Contact Information:
        - LinkedIn Profile: linkedin.com/in/alexjohnson85
        - Personal Website: www.alexjohnson.dev
        - GitHub Username: alex_johnson_dev
        - Twitter Handle: @alexjohnsontech
        - Secondary Email: alex.personal@gmail.com
        
        Family Information:
        - Spouse: Maria Elena Johnson (DOB: July 22, 1987)
        - Children: Emma Johnson (DOB: May 10, 2015), Liam Johnson (DOB: August 3, 2018)
        - Father: Michael Robert Johnson (Phone: 555-456-7890)
        - Mother: Patricia Ann Johnson (Phone: 555-567-8901)
        
        Education History:
        - University: Stanford University
        - Degree: Bachelor of Science in Computer Science
        - Graduation Date: June 2007
        - Student ID: STU-20030145
        - GPA: 3.8/4.0
        
        Previous Employment:
        - Company: TechStart Inc.
        - Position: Software Developer
        - Duration: 2007-2020
        - Reference: David Miller (david.miller@techstart.com, 555-678-9012)
        
        Financial Information:
        - Credit Score: 750
        - Annual Income: $95,000
        - Tax ID: 123-45-6789
        - 401k Account: 401K-987654321
        - Stock Options: 1000 shares vested
        
        Legal Information:
        - Driver's License: DL-IL-123456789
        - Passport Number: 123456789
        - Voter Registration: VR-IL-987654321
        - Professional License: PE-IL-456789 (Professional Engineer)
        
        Technology Access:
        - Company Laptop: LAPTOP-001234
        - IP Address (Work): 192.168.1.100
        - VPN Username: ajohnson_vpn
        - Active Directory: COMPANY\\ajohnson
        - Badge Number: BADGE-001234
        
        This document contains highly sensitive personal and financial information. 
        Access is restricted to authorized personnel only. Any unauthorized disclosure 
        is strictly prohibited and may result in legal action.
        
        Document ID: DOC-CONF-20231201
        Classification: CONFIDENTIAL
        Last Updated: December 1, 2023
        Updated By: HR Administrator (hradmin@company.com)
        """.strip()
    return {
        "short": "Hi, my name is John Smith and my email is john.smith@company.com. I live at 123 Main Street.",
        
        "medium": """
        Dear Mr. Johnson,
        
        I hope this email finds you well. My name is Sarah Davis and I work at TechCorp Inc. 
        You can reach me at sarah.davis@techcorp.com or call me at (555) 123-4567.
        
        I wanted to discuss the project we talked about. Our office is located at 456 Oak Avenue, 
        Springfield, IL 62701. We also have a secondary location at 789 Pine Street, Suite 200.
        
        My employee ID is EMP-2023-001 and my direct phone number is 555-987-6543.
        
        Best regards,
        Sarah Davis
        """.strip(),
        
        "long_3k": text_3k,
        
        "very_long_6k": text_3k * 2
    }

if __name__ == "__main__":
    import asyncio
    
    async def test_setup():
        """Test the service setup."""
        logging.basicConfig(level=logging.INFO)
        
        try:
            services = await create_services_for_benchmark()
            print(f"Successfully initialized {len(services)} services:")
            
            for name, service in services.items():
                info = service.get_service_info()
                print(f"  - {name}: {info.get('service_name', 'Unknown')}")
            
            # Test with a simple text
            test_text = "Hi, my name is John Smith and my email is john@example.com"
            
            for name, service in services.items():
                print(f"\nTesting {name}...")
                result = await service.predict(test_text)
                print(f"  Found {len(result.entities)} entity types")
                print(f"  Masked: {result.masked_text}")
                
        except Exception as e:
            print(f"Setup test failed: {e}")
            
    asyncio.run(test_setup()) 