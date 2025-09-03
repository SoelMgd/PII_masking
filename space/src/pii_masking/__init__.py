"""
PII Masking Package

A comprehensive toolkit for evaluating and comparing PII masking approaches
using various LLM techniques including prompting and fine-tuning.
"""

__version__ = "0.1.0"
__author__ = "Applied AI Engineer Candidate"

from .data_loader import PIIDataLoader, PIIExample
from .base_model import (
    BasePIIModel, 
    reconstruct_masked_text_from_prediction
)
from .text_processing import PIIPrediction, EntitySpan, reconstruct_masked_text

__all__ = [
    "PIIDataLoader",
    "PIIExample", 
    "BasePIIModel",
    "PIIPrediction",
    "reconstruct_masked_text_from_prediction",
    "EntitySpan",
    "reconstruct_masked_text"
] 