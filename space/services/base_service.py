#!/usr/bin/env python3
"""
Base service interface for PII inference services.
"""

import sys
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from pii_masking.text_processing import PIIPrediction, reconstruct_masked_text


class BasePIIInferenceService(ABC):
    """
    Abstract base class for PII inference services.
    
    Supports both sync (BERT) and async (Mistral API) service types
    with automatic routing through a unified interface.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service (load models, connect to APIs, etc.)."""
        pass
    
    def predict_sync(self, text: str, pii_entities: List[str] = None) -> PIIPrediction:
        """Synchronous prediction method for CPU-bound models like BERT."""
        raise NotImplementedError("This service doesn't support synchronous prediction")
    
    async def predict_async_native(self, text: str, pii_entities: List[str] = None) -> PIIPrediction:
        """Native async prediction method for I/O-bound models like Mistral API."""
        raise NotImplementedError("This service doesn't support native async prediction")
    
    async def predict(self, text: str, pii_entities: List[str] = None) -> PIIPrediction:
        """
        Unified async prediction method that routes to the appropriate implementation.
        
        - For sync models (BERT): calls predict_sync in a thread pool
        - For async models (Mistral): calls predict_async_native directly
        """
        try:
            return await self.predict_async_native(text, pii_entities)
        except NotImplementedError:
            pass
        
        try:
            return await asyncio.to_thread(self.predict_sync, text, pii_entities)
        except NotImplementedError:
            raise NotImplementedError("Service must implement either predict_sync or predict_async_native")
    
    @abstractmethod
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information for health checks and debugging."""
        pass
    
    def _filter_prediction_by_entities(self, prediction: PIIPrediction, allowed_entities: List[str]) -> PIIPrediction:
        """Filter prediction to only include specified entity types."""
        if not allowed_entities:
            return prediction
        
        allowed_set = set(allowed_entities)
        
        filtered_entities = {
            entity_type: entities 
            for entity_type, entities in prediction.entities.items() 
            if entity_type in allowed_set
        }
        
        filtered_spans = [
            span for span in prediction.spans 
            if span.entity_type in allowed_set
        ]
        
        filtered_masked_text = reconstruct_masked_text(prediction.original_text, filtered_entities)
        
        return PIIPrediction(
            entities=filtered_entities,
            spans=filtered_spans,
            masked_text=filtered_masked_text,
            original_text=prediction.original_text
        )


class PIIServiceManager:
    """Manager class for handling multiple PII inference services."""
    
    def __init__(self):
        self.services: Dict[str, BasePIIInferenceService] = {}
    
    def register_service(self, name: str, service: BasePIIInferenceService):
        """Register a PII inference service."""
        self.services[name] = service
    
    def get_service(self, name: str) -> Optional[BasePIIInferenceService]:
        """Get a registered service by name."""
        return self.services.get(name)
    
    def list_services(self) -> List[str]:
        """List all registered service names."""
        return list(self.services.keys())
    
    def get_services_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all registered services."""
        return {
            name: service.get_service_info() 
            for name, service in self.services.items()
        }
    
    async def predict_with_service(self, service_name: str, text: str, 
                                 pii_entities: List[str] = None) -> PIIPrediction:
        """Make a prediction using a specific service."""
        service = self.get_service(service_name)
        if not service:
            raise ValueError(f"Service '{service_name}' not found")
        
        if not service.is_initialized:
            raise ValueError(f"Service '{service_name}' is not initialized")
        
        return await service.predict(text, pii_entities) 