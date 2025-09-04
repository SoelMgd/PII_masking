#!/usr/bin/env python3
"""
Mistral Prompting Inference Service for HuggingFace Space.

Production service with batch inference for long texts.
"""

import os
import sys
import json
import logging
import asyncio
import time
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from pii_masking.text_processing import PIIPrediction, EntitySpan, reconstruct_masked_text

from .base_service import BasePIIInferenceService

from mistralai import Mistral as MistralClient, File
from mistralai.models import UserMessage

logger = logging.getLogger(__name__)

class MistralPromptingService(BasePIIInferenceService):
    """Production-ready Mistral prompting service for PII detection."""
    
    def __init__(self, api_key: str, model_name: str = "mistral-large-latest", enable_batching: bool = True):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.enable_batching = enable_batching
        self.client = None
        
        self.is_fine_tuned = model_name.startswith("ft:")
        
        self.config = {
            'temperature': 0.1,
            'max_tokens': 1000,
            'max_retries': 3,
            'batch_threshold': 2000,
            'chunk_size': 1500,
            'chunk_overlap': 200,
            'batch_timeout': 300,
            'batch_poll_interval': 2
        }
        
        if self.is_fine_tuned:
            self.prompt_template = """Please extract all Personal Identifiable Information (PII) from the text.
Text to analyze:
{text}"""
        else:
            self.prompt_template = """You are an expert in Personal Identifiable Information (PII) detection.

Your task is to analyze the provided text and identify ALL PII entities, returning them as a structured JSON.

PII Entity Types to detect:
PREFIX, FIRSTNAME, LASTNAME, DATE, TIME, USERNAME, GENDER, CITY, STATE, URL, EMAIL, 
JOBTYPE, COMPANYNAME, JOBTITLE, STREET, SECONDARYADDRESS, COUNTY, AGE, ACCOUNTNAME, 
PHONENUMBER, SEX, IP, MIDDLENAME, DOB, BUILDINGNUMBER, ZIPCODE, SSN

Instructions:
1. Identify each PII element in the text
2. Extract the EXACT substring from the original text (preserve case, punctuation, etc.)
3. Return ONLY a JSON object with the format: {{"PII": {{"ENTITY_TYPE": ["substring1", "substring2", ...]}}}}
4. If no PII is found, return: {{"PII": {{}}}}
5. Do NOT include non-PII text or explanations - only actual PII entities

Text to analyze:
{text}

JSON Output:"""
        
        logger.info(f"Mistral service initialized with model: {model_name}")
        logger.info(f"Fine-tuned model: {self.is_fine_tuned}")
        logger.info(f"Batch processing: {'enabled' if enable_batching else 'disabled'}")

    async def initialize(self) -> bool:
        """Initialize the Mistral client."""
        try:
            logger.info("Initializing Mistral client...")
            self.client = MistralClient(api_key=self.api_key)
            self.is_initialized = True
            logger.info("Mistral client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            return False
    
    def create_prompt(self, text: str) -> str:
        """Create a prompt for PII detection."""
        return self.prompt_template.replace("{text}", text)
    
    def _chunk_text(self, text: str) -> List[Tuple[str, int]]:
        """Split text into overlapping chunks for batch processing."""
        if len(text) <= self.config['chunk_size']:
            return [(text, 0)]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.config['chunk_size']
            
            if end >= len(text):
                chunk = text[start:]
                chunks.append((chunk, start))
                break
            
            chunk_end = end
            for i in range(end, start + self.config['chunk_size'] - self.config['chunk_overlap'], -1):
                if text[i] in ' \n\t.!?;,':
                    chunk_end = i + 1
                    break
            
            chunk = text[start:chunk_end]
            chunks.append((chunk, start))
            
            start = chunk_end - self.config['chunk_overlap']
            if start < 0:
                start = 0
        
        logger.debug(f"Text chunked into {len(chunks)} pieces (sizes: {[len(c[0]) for c in chunks]})")
        return chunks
    
    async def _create_batch_file(self, chunks: List[Tuple[str, int]]) -> File:
        """Create a batch input file from text chunks."""
        buffer = BytesIO()
        
        for i, (chunk_text, start_pos) in enumerate(chunks):
            prompt = self.create_prompt(chunk_text)
            request = {
                "custom_id": f"chunk_{i}_{start_pos}",
                "body": {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config['temperature'],
                    "max_tokens": self.config['max_tokens'],
                    "response_format": {"type": "json_object"}
                }
            }
            buffer.write(json.dumps(request).encode("utf-8"))
            buffer.write("\n".encode("utf-8"))
        
        file_name = f"pii_batch_{uuid.uuid4().hex[:8]}.jsonl"
        batch_file = self.client.files.upload(
            file=File(file_name=file_name, content=buffer.getvalue()),
            purpose="batch"
        )
        
        logger.debug(f"Batch file uploaded: {batch_file.id}")
        return batch_file
    
    async def _run_batch_job(self, batch_file: File) -> Dict[str, Any]:
        """Run a batch job and wait for completion."""
        batch_job = self.client.batch.jobs.create(
            input_files=[batch_file.id],
            model=self.model_name,
            endpoint="/v1/chat/completions",
            metadata={"job_type": "pii_detection"}
        )
        
        logger.info(f"Batch job created: {batch_job.id}")
        
        start_time = time.time()
        while batch_job.status in ["QUEUED", "RUNNING"]:
            if time.time() - start_time > self.config['batch_timeout']:
                logger.error(f"Batch job {batch_job.id} timed out")
                raise TimeoutError(f"Batch job timed out after {self.config['batch_timeout']} seconds")
            
            await asyncio.sleep(self.config['batch_poll_interval'])
            batch_job = self.client.batch.jobs.get(job_id=batch_job.id)
            
            total = batch_job.total_requests or 0
            completed = (batch_job.succeeded_requests or 0) + (batch_job.failed_requests or 0)
            if total > 0:
                progress = (completed / total) * 100
                logger.debug(f"Batch progress: {completed}/{total} ({progress:.1f}%)")
        
        logger.info(f"Batch job completed with status: {batch_job.status}")
        
        if batch_job.status != "SUCCESS":
            raise RuntimeError(f"Batch job failed with status: {batch_job.status}")
        
        output_file_stream = self.client.files.download(file_id=batch_job.output_file)
        results = {}
        
        for line in output_file_stream.read().decode('utf-8').strip().split('\n'):
            if line:
                result = json.loads(line)
                custom_id = result.get('custom_id')
                if custom_id and result.get('response'):
                    content = result['response']['body']['choices'][0]['message']['content']
                    results[custom_id] = content
        
        logger.debug(f"Downloaded {len(results)} batch results")
        return results
    
    def _merge_chunk_predictions(self, chunk_results: List[Tuple[PIIPrediction, int]], original_text: str) -> PIIPrediction:
        """
        Merge predictions from multiple chunks, handling overlaps and adjusting positions.
        
        Args:
            chunk_results: List of (PIIPrediction, start_position) tuples
            original_text: Original full text
            
        Returns:
            Combined PIIPrediction for the full text
        """
        all_entities = {}
        all_spans = []
        seen_entities = set()
        
        for prediction, chunk_start in chunk_results:
            for span in prediction.spans:
                adjusted_start = span.start + chunk_start
                adjusted_end = span.end + chunk_start
                
                entity_key = (adjusted_start, adjusted_end, span.entity_type, span.text)
                if entity_key in seen_entities:
                    continue
                
                seen_entities.add(entity_key)
                
                adjusted_span = EntitySpan(
                    entity_type=span.entity_type,
                    start=adjusted_start,
                    end=adjusted_end,
                    text=span.text
                )
                all_spans.append(adjusted_span)
                
                if span.entity_type not in all_entities:
                    all_entities[span.entity_type] = []
                if span.text not in all_entities[span.entity_type]:
                    all_entities[span.entity_type].append(span.text)
        
        all_spans.sort(key=lambda x: x.start)
        
        masked_text = reconstruct_masked_text(original_text, all_entities)
        
        final_prediction = PIIPrediction(
            entities=all_entities,
            spans=all_spans,
            masked_text=masked_text,
            original_text=original_text
        )
        
        logger.debug(f"Merged {len(all_spans)} entities from {len(chunk_results)} chunks")
        return final_prediction

    async def predict_async_native(self, text: str, pii_entities: List[str] = None) -> PIIPrediction:
        """
        Native async prediction method for Mistral (I/O-bound API calls).
        This is the natural way to call external APIs.
        """
        if not self.is_initialized:
            raise RuntimeError("Service not initialized. Call initialize() first.")
        
        if not text or not text.strip():
            return PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
        
        if self.enable_batching and len(text) > self.config['batch_threshold']:
            return await self._predict_batch(text, pii_entities)
        else:
            return await self._predict_single(text, pii_entities)

    async def _predict_batch(self, text: str, pii_entities: List[str] = None) -> PIIPrediction:
        """Handle batch processing for long texts."""
        logger.info(f"Using batch processing for long text ({len(text)} chars)")
        
        try:
            chunks = self._chunk_text(text)
            
            batch_file = await self._create_batch_file(chunks)
            
            batch_results = await self._run_batch_job(batch_file)
            
            chunk_predictions = []
            for i, (chunk_text, start_pos) in enumerate(chunks):
                custom_id = f"chunk_{i}_{start_pos}"
                if custom_id in batch_results:
                    json_prediction = batch_results[custom_id]
                    prediction = PIIPrediction.from_json_and_text(json_prediction, chunk_text)
                    chunk_predictions.append((prediction, start_pos))
                else:
                    empty_prediction = PIIPrediction(
                        entities={}, spans=[], masked_text=chunk_text, original_text=chunk_text
                    )
                    chunk_predictions.append((empty_prediction, start_pos))
            
            final_prediction = self._merge_chunk_predictions(chunk_predictions, text)
            
            if pii_entities is not None:
                final_prediction = self._filter_prediction_by_entities(final_prediction, pii_entities)
            
            logger.info(f"Batch processing completed: {len(final_prediction.spans)} entities found")
            return final_prediction
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            logger.info("Falling back to single request processing")
            return await self._predict_single(text, pii_entities)

    async def _predict_single(self, text: str, pii_entities: List[str] = None) -> PIIPrediction:
        """
        Predict PII entities using single API request (original method).
        
        Args:
            text: Input text to analyze
            pii_entities: List of PII entity types to mask (if None, mask all detected entities)
            
        Returns:
            PIIPrediction object
        """
        try:
            prompt = self.create_prompt(text.strip())
            logger.debug(f"Processing text: {text[:100]}...")
            
            messages = [UserMessage(role="user", content=prompt)]
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
                response_format={"type": "json_object"}
            )
            
            json_prediction = response.choices[0].message.content.strip()
            logger.debug(f"LLM output: {json_prediction}")
            
            prediction = PIIPrediction.from_json_and_text(json_prediction, text)
            logger.debug(f"Found {len(prediction.spans)} PII entities")
            
            if pii_entities is not None:
                prediction = self._filter_prediction_by_entities(prediction, pii_entities)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in Mistral API call: {e}")
            return PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
    
    async def predict_batch(self, texts: list[str]) -> list[PIIPrediction]:
        """
        Predict PII entities for multiple texts with rate limiting.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of PIIPrediction objects
        """
        if not texts:
            return []
        
        logger.info(f"Processing batch of {len(texts)} texts")
        predictions = []
        
        for i, text in enumerate(texts):
            try:
                prediction = await self.predict(text)
                predictions.append(prediction)
                
                if (i + 1) % 5 == 0 or i == len(texts) - 1:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                
                if i < len(texts) - 1:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error processing text {i}: {e}")
                # Add empty prediction on error
                empty_prediction = PIIPrediction(
                    entities={}, 
                    spans=[], 
                    masked_text=text, 
                    original_text=text
                )
                predictions.append(empty_prediction)
        
        logger.info(f"Batch processing completed: {len(predictions)} predictions")
        return predictions
    


    def get_service_info(self) -> Dict[str, Any]:
        """Get service information for monitoring."""
        return {
            "service_name": "MistralPromptingService",
            "model_name": self.model_name,
            "is_initialized": self.is_initialized,
            "enable_batching": self.enable_batching,
            "is_fine_tuned": self.is_fine_tuned,
            "config": self.config,
            "description": "Mistral API-based PII detection with JSON output and batch processing support"
        }

async def create_mistral_service(api_key: Optional[str] = None, model_name: str = "mistral-large-latest", enable_batching: bool = True) -> MistralPromptingService:
    """
    Factory function to create and initialize Mistral service.
    
    Args:
        api_key: Mistral API key (if None, will read from environment)
        model_name: Model to use (base or fine-tuned model)
        enable_batching: Whether to enable batch processing for long texts
        
    Returns:
        Initialized MistralPromptingService
    """
    if api_key is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not provided and not found in environment variables")
    
    service = MistralPromptingService(api_key, model_name, enable_batching)
    
    if not await service.initialize():
        raise RuntimeError("Failed to initialize Mistral service")
    
    return service

# Test function for development
async def test_service():
    """Test function for development."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        for batching_enabled in [True, False]:
            service = await create_mistral_service(enable_batching=batching_enabled)
            batch_status = "enabled" if batching_enabled else "disabled"
            
            test_texts = [
                "Hi, my name is John Smith and my email is john.smith@company.com.",
                "Call me at 555-1234 or visit 123 Main Street.",
                "This is a normal sentence with no PII information.",
                " ".join([
                    "This is a long document containing multiple PII entities.",
                    "My name is Alice Johnson and I work at TechCorp Inc.",
                    "You can reach me at alice.johnson@techcorp.com or call 555-0123.",
                    "I live at 456 Oak Avenue, Springfield, IL 62701.",
                    "My date of birth is March 15, 1985 and my SSN is 123-45-6789.",
                    "I also have a secondary address at 789 Pine Street, Unit 4B.",
                    "My username is alice_j85 and my account number is ACC-2023-001.",
                ]) * 10
            ]
            
            print(f"\nTesting Mistral Prompting Service (batching {batch_status})")
            print("=" * 60)
            
            for i, text in enumerate(test_texts, 1):
                print(f"\nTest {i}: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"Text length: {len(text)} characters")
                
                start_time = time.time()
                prediction = await service.predict(text)
                processing_time = time.time() - start_time
                
                print(f"Entities: {len(prediction.entities)} types")
                print(f"Spans: {len(prediction.spans)} entities")
                print(f"Processing time: {processing_time:.2f}s")
                print(f"Masked preview: {prediction.masked_text[:100]}{'...' if len(prediction.masked_text) > 100 else ''}")
        
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_service()) 