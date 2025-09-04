"""
Base model interface for PII masking experiments.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass

from .data_loader import PIIExample
from .text_processing import PIIPrediction, reconstruct_masked_text_from_prediction as _reconstruct

logger = logging.getLogger(__name__)

# Legacy utility function for backward compatibility
def reconstruct_masked_text_from_prediction(text: str, prediction: Union[str, PIIPrediction]) -> str:
    """
    Reconstruct masked text from a prediction.
    
    Args:
        text: Original unmasked text
        prediction: Either a JSON string or PIIPrediction object
        
    Returns:
        Text with PII replaced by [ENTITY_TYPE_X] placeholders
    """
    return _reconstruct(text, prediction)

class BasePIIModel(ABC):
    """
    Abstract base class for all PII masking models.
    
    This ensures a consistent interface across different approaches:
    - Prompting-based models (Mistral, OpenAI, etc.)
    - Fine-tuned models (BERT, RoBERTa, etc.)
    - Hybrid approaches
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        """
        Initialize the model.
        
        Args:
            model_name: Name/identifier for the model
            config: Configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.is_initialized = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the model (load weights, connect to API, etc.).
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict_single(self, text: str, **kwargs) -> PIIPrediction:
        """
        Predict PII entities for a single text.
        
        Args:
            text: Input text containing PII
            **kwargs: Additional model-specific parameters
            
        Returns:
            PIIPrediction object with entities, spans, and masked text pre-computed
            
        Raises:
            NotImplementedError: If model is not initialized
            Exception: If prediction fails
        """
        pass
    
    def predict_batch(self, texts: List[str], **kwargs) -> List[PIIPrediction]:
        """
        Predict PII entities for a batch of texts.
        
        Default implementation calls predict_single for each text.
        Override for models that support efficient batch processing.
        
        Args:
            texts: List of input texts
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of PIIPrediction objects with pre-computed entities, spans, and masked text.
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
        
        predictions = []
        for i, text in enumerate(texts):
            try:
                prediction = self.predict_single(text, **kwargs)
                predictions.append(prediction)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error predicting text {i}: {e}")
                # Empty prediction on error
                empty_prediction = PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
                predictions.append(empty_prediction)
        
        return predictions
    
    def predict_examples(self, examples: List[PIIExample], **kwargs) -> List[PIIPrediction]:
        """
        Predict PII entities for a list of PIIExample objects.
        
        Args:
            examples: List of PIIExample objects
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of PIIPrediction objects with pre-computed entities, spans, and masked text.
        """
        texts = [example.unmasked_text for example in examples]
        return self.predict_batch(texts, **kwargs)
    
    def predict_dataset(self, examples: List[PIIExample], 
                       max_samples: Optional[int] = None,
                       use_batch_inference: bool = False,
                       **kwargs) -> List[PIIPrediction]:
        """
        Predict PII entities for a dataset with optional batching and sampling.
        
        This method provides a high-level interface for processing datasets
        with support for:
        - Sampling a subset of examples
        - Batch inference for efficiency (model-specific)
        - Progress tracking for large datasets
        
        Args:
            examples: List of PIIExample objects
            max_samples: Maximum number of samples to process (None = all)
            use_batch_inference: Whether to use batch inference if available
            **kwargs: Additional model-specific parameters
            
        Returns:
            List of PIIPrediction objects with pre-computed entities, spans, and masked text.
        """
        # Sample examples if requested
        if max_samples is not None and len(examples) > max_samples:
            import random
            examples = random.sample(examples, max_samples)
            logger.info(f"Sampled {max_samples} examples from {len(examples)} total")
        
        # Use batch inference if available and requested
        if use_batch_inference and hasattr(self, 'predict_batch_inference'):
            logger.info(f"Using batch inference for {len(examples)} examples")
            return self.predict_batch_inference(examples, **kwargs)
        
        # Fallback to standard batch processing
        logger.info(f"Using standard batch processing for {len(examples)} examples")
        return self.predict_examples(examples, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'config': self.config,
            'is_initialized': self.is_initialized,
            'model_type': self.__class__.__name__
        }
    
    def cleanup(self):
        """
        Cleanup resources (close connections, free memory, etc.).
        
        Override if your model needs specific cleanup.
        """
        self.is_initialized = False
        logger.info(f"Model {self.model_name} cleaned up")

class PromptBasedModel(BasePIIModel):
    """
    Base class for prompt-based models (API calls).
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        self.api_client = None
        self.prompt_template = None
        
    def create_prompt(self, text: str, few_shot_examples: Optional[List[PIIExample]] = None) -> str:
        """
        Create a prompt for PII masking.
        
        Args:
            text: Input text to mask
            few_shot_examples: Optional examples for few-shot learning
            
        Returns:
            Formatted prompt string
        """
        if self.prompt_template is None:
            raise NotImplementedError("Prompt template not defined")
        
        # Basic template substitution
        prompt = self.prompt_template.format(text=text)
        
        # Add few-shot examples if provided
        if few_shot_examples:
            examples_text = self._format_few_shot_examples(few_shot_examples)
            prompt = prompt.replace("{examples}", examples_text)
        else:
            prompt = prompt.replace("{examples}", "")
        
        return prompt
    
    def _format_few_shot_examples(self, examples: List[PIIExample]) -> str:
        """Format few-shot examples for the prompt."""
        formatted = "\nExamples:\n\n"
        
        for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples
            formatted += f"Example {i}:\n"
            formatted += f"Input: {example.unmasked_text}\n"
            formatted += f"Output: {example.masked_text}\n\n"
        
        return formatted