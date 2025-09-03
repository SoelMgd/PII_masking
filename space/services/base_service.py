#!/usr/bin/env python3
"""
Base service interface for PII inference services.

This module defines the common interface that all PII inference services should implement.
"""

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import our existing module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from pii_masking.text_processing import PIIPrediction, reconstruct_masked_text


class BasePIIInferenceService(ABC):
    """
    Abstract base class for PII inference services.
    
    This ensures a consistent interface across different PII detection approaches:
    - Mistral API-based services (prompting, fine-tuning)
    - BERT token classification services
    - Other future implementations
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the base service.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the service (load models, connect to APIs, etc.).
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def predict(self, text: str, pii_entities: List[str] = None) -> PIIPrediction:
        """
        Predict PII entities for a single text.
        
        Args:
            text: Input text to analyze
            pii_entities: List of PII entity types to mask (if None, mask all detected entities)
            
        Returns:
            PIIPrediction object with entities, spans, and masked text
        """
        pass
    
    @abstractmethod
    def get_service_info(self) -> Dict[str, Any]:
        """
        Get service information for monitoring and debugging.
        
        Returns:
            Dictionary with service metadata (name, model, status, etc.)
        """
        pass
    
    def _filter_prediction_by_entities(self, prediction: PIIPrediction, allowed_entities: List[str]) -> PIIPrediction:
        """
        Filter prediction to only include specified entity types.
        
        This is a common method that can be shared across all services.
        
        Args:
            prediction: Original PIIPrediction object
            allowed_entities: List of entity types to keep
            
        Returns:
            Filtered PIIPrediction object
        """
        if not allowed_entities:
            return prediction
        
        # Convert to set for faster lookup
        allowed_set = set(allowed_entities)
        
        # Filter entities dictionary
        filtered_entities = {
            entity_type: entities 
            for entity_type, entities in prediction.entities.items() 
            if entity_type in allowed_set
        }
        
        # Filter spans
        filtered_spans = [
            span for span in prediction.spans 
            if span.entity_type in allowed_set
        ]
        
        # Reconstruct masked text with filtered entities
        filtered_masked_text = reconstruct_masked_text(prediction.original_text, filtered_entities)
        
        return PIIPrediction(
            entities=filtered_entities,
            spans=filtered_spans,
            masked_text=filtered_masked_text,
            original_text=prediction.original_text
        )


class PIIServiceManager:
    """
    Manager class for handling multiple PII inference services.
    
    This class provides a unified interface to work with different PII detection services
    (Mistral, BERT, etc.) through a common API.
    """
    
    def __init__(self):
        """Initialize the service manager."""
        self.services: Dict[str, BasePIIInferenceService] = {}
    
    def register_service(self, name: str, service: BasePIIInferenceService):
        """
        Register a PII inference service.
        
        Args:
            name: Unique name for the service (e.g., 'mistral-base', 'bert', 'mistral-finetuned')
            service: The service instance
        """
        self.services[name] = service
    
    def get_service(self, name: str) -> Optional[BasePIIInferenceService]:
        """
        Get a registered service by name.
        
        Args:
            name: Name of the service
            
        Returns:
            Service instance or None if not found
        """
        return self.services.get(name)
    
    def list_services(self) -> List[str]:
        """
        List all registered service names.
        
        Returns:
            List of service names
        """
        return list(self.services.keys())
    
    def get_services_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered services.
        
        Returns:
            Dictionary mapping service names to their info
        """
        return {
            name: service.get_service_info() 
            for name, service in self.services.items()
        }
    
    async def predict_with_service(self, service_name: str, text: str, 
                                 pii_entities: List[str] = None) -> PIIPrediction:
        """
        Make a prediction using a specific service.
        
        Args:
            service_name: Name of the service to use
            text: Input text to analyze
            pii_entities: List of PII entity types to mask
            
        Returns:
            PIIPrediction object
            
        Raises:
            ValueError: If service not found or not initialized
        """
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service '{service_name}' not found")
        
        if not service.is_initialized:
            raise ValueError(f"Service '{service_name}' is not initialized")
        
        return await service.predict(text, pii_entities) 