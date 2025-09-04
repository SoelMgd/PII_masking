#!/usr/bin/env python3
"""
Fine-tuned model testing script.

This script:
1. Tests a fine-tuned Mistral model on PII detection
2. Uses the same evaluation framework as the efficient approach
3. Supports batch inference for efficient processing
4. Compares performance with baseline models
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from io import BytesIO

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pii_masking import PIIDataLoader, PIIExample, PIIPrediction
from pii_masking.base_model import BasePIIModel
from pii_masking.custom_evaluator import CustomPIIEvaluator, CustomEvaluationResult

from mistralai import Mistral, File
from mistralai.models import UserMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FineTunedMistralModel(BasePIIModel):
    """Fine-tuned Mistral model for PII detection."""
    
    def __init__(self, model_name: str, api_key: str, config: Dict[str, Any] = None):
        super().__init__(model_name, config)
        self.api_key = api_key
        self.client = None
        
    def initialize(self) -> bool:
        """Initialize the Mistral client for fine-tuned model."""
        try:
            self.client = Mistral(api_key=self.api_key)
            self.is_initialized = True
            logger.info(f"Fine-tuned model client initialized: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize fine-tuned model client: {e}")
            return False
    
    def predict_single(self, text: str, **kwargs) -> PIIPrediction:
        """Predict using fine-tuned model via Mistral API."""
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")
        
        try:
            
            prompt = f"""Please extract all Personal Identifiable Information (PII) from the text.
Text to analyze:
{text}"""

            messages = [UserMessage(role="user", content=prompt)]
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 1000),
                response_format={"type": "json_object"}
            )
            
            json_prediction = response.choices[0].message.content.strip()
            logger.debug(f"Fine-tuned model output: {str(json_prediction)}")
            
            return PIIPrediction.from_json_and_text(json_prediction, text)
            
        except Exception as e:
            logger.error(f"Error in fine-tuned model API call: {e}")
            return PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
    
    def predict_dataset(self, examples: List[PIIExample], 
                       max_samples: Optional[int] = None,
                       use_batch_inference: bool = False,
                       **kwargs) -> List[PIIPrediction]:
        """
        Predict PII entities for a dataset with optional batching and sampling.
        """
        if max_samples is not None and len(examples) > max_samples:
            import random
            examples = random.sample(examples, max_samples)
            logger.info(f"Sampled {max_samples} examples from {len(examples)} total")
        
        if use_batch_inference and hasattr(self, 'predict_batch_inference'):
            logger.info(f"Using batch inference for {len(examples)} examples")
            return self.predict_batch_inference(examples, **kwargs)
        
        logger.info(f"Using standard batch processing for {len(examples)} examples")
        return self.predict_examples(examples, **kwargs)
    
    def predict_batch_inference(self, examples: List[PIIExample], **kwargs) -> List[PIIPrediction]:
        """
        Predict using Mistral's batch inference API for efficient processing.
        
        Returns:
            List of complete PIIPrediction objects
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        logger.info(f"Starting batch inference for {len(examples)} examples")

        batch_file = self._create_batch_input_file(examples)
        logger.info(f"Created batch input file: {batch_file.id}")

        batch_job = self._run_batch_job(batch_file)
        logger.info(f"Batch job completed: {batch_job.id}")

        json_predictions = self._parse_batch_results(batch_job, len(examples))
        logger.info(f"Parsed {len(json_predictions)} JSON predictions from batch results")
        
        predictions = []
        for i, json_pred in enumerate(json_predictions):
            try:
                prediction = PIIPrediction.from_json_and_text(json_pred, examples[i].unmasked_text)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error creating PIIPrediction for example {i}: {e}")
                empty_prediction = PIIPrediction(
                    entities={}, 
                    spans=[], 
                    masked_text=examples[i].unmasked_text, 
                    original_text=examples[i].unmasked_text
                )
                predictions.append(empty_prediction)

        return predictions
    
    def _create_batch_input_file(self, examples: List[PIIExample]) -> File:
        """Create a JSONL file for batch inference."""
        buffer = BytesIO()
        
        for idx, example in enumerate(examples):
            try:
                prompt = f"""Please extract all Personal Identifiable Information (PII) from the text.
Text to analyze:
{example.unmasked_text}"""
            except Exception as e:
                logger.error(f"Error creating prompt for example {idx}: {e}")
                raise
            
            request = {
                "custom_id": str(idx),
                "body": {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": self.config.get('temperature', 0.1),
                    "max_tokens": self.config.get('max_tokens', 1000),
                    "response_format": {"type": "json_object"}
                }
            }
            
            buffer.write(json.dumps(request).encode("utf-8"))
            buffer.write("\n".encode("utf-8"))
        
        return self.client.files.upload(
            file=File(
                file_name=f"pii_finetuned_batch_{len(examples)}.jsonl",
                content=buffer.getvalue()
            ),
            purpose="batch"
        )
    
    def _run_batch_job(self, input_file: File):
        """Run the batch job and wait for completion."""
        batch_job = self.client.batch.jobs.create(
            input_files=[input_file.id],
            model=self.model_name,
            endpoint="/v1/chat/completions",
            metadata={"job_type": "pii_finetuned", "framework": "pii-masking-finetuned"}
        )
        
        logger.info(f"Created batch job {batch_job.id}, status: {batch_job.status}")
        
        while batch_job.status in ["QUEUED", "RUNNING"]:
            batch_job = self.client.batch.jobs.get(job_id=batch_job.id)
            
            if hasattr(batch_job, 'total_requests') and batch_job.total_requests > 0:
                completed = (batch_job.succeeded_requests or 0) + (batch_job.failed_requests or 0)
                progress = (completed / batch_job.total_requests) * 100
                logger.info(f"Batch progress: {progress:.1f}% ({completed}/{batch_job.total_requests})")
            
            time.sleep(5)
        
        if batch_job.status != "SUCCESS":
            logger.error(f"Batch job failed with status: {batch_job.status}")
            raise RuntimeError(f"Batch job failed with status: {batch_job.status}")
        
        return batch_job
    
    def _parse_batch_results(self, batch_job, expected_count: int) -> List[str]:
        """Parse batch job results and return predictions in order."""
        if not batch_job.output_file:
            raise RuntimeError("No output file available from batch job")
        
        output_stream = self.client.files.download(file_id=batch_job.output_file)
        results_content = output_stream.read().decode('utf-8')
        
        results_by_id = {}
        for line in results_content.strip().split('\n'):
            if line:
                result = json.loads(line)
                custom_id = int(result['custom_id'])
                
                if 'response' in result and 'body' in result['response']:
                    choices = result['response']['body'].get('choices', [])
                    if choices:
                        prediction = choices[0]['message']['content'].strip()
                        results_by_id[custom_id] = prediction
                    else:
                        logger.warning(f"No choices in result for ID {custom_id}")
                        results_by_id[custom_id] = '{"PII": {}}'
                else:
                    logger.warning(f"Invalid result format for ID {custom_id}")
                    results_by_id[custom_id] = '{"PII": {}}'
        
        predictions = []
        for i in range(expected_count):
            if i in results_by_id:
                predictions.append(results_by_id[i])
            else:
                logger.warning(f"Missing result for index {i}")
                predictions.append('{"PII": {}}')
        
        return predictions

def run_finetuned_experiment(
    fine_tuned_model_name: str,
    dataset_path: str,
    num_samples: int = 100,
    config: Dict[str, Any] = None
) -> CustomEvaluationResult:
    """
    Run the fine-tuned model experiment.
    
    Args:
        fine_tuned_model_name: Name of the fine-tuned model
        dataset_path: Path to the dataset file
        num_samples: Number of samples to evaluate
        config: Configuration dictionary
        
    Returns:
        CustomEvaluationResult object with evaluation metrics
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("MISTRAL_API_KEY")
    
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    data_loader = PIIDataLoader(data_dir=Path(dataset_path).parent)
    evaluator = CustomPIIEvaluator()
    
    logger.info(f"Loading dataset from {dataset_path}")
    examples = data_loader.load_dataset(
        language='english',
        max_samples=num_samples,
        shuffle=True,
        seed=42
    )
    
    if not examples:
        raise ValueError("No examples loaded from dataset")
    
    logger.info(f"Loaded {len(examples)} examples")
    
    model_config = config.get('model_config', {}) if config else {}
    model = FineTunedMistralModel(
        model_name=fine_tuned_model_name,
        api_key=api_key,
        config=model_config
    )
    
    if not model.initialize():
        raise RuntimeError("Failed to initialize fine-tuned Mistral model")
    
    logger.info("Generating predictions...")
    
    if 'use_batch_inference' in config:
        use_batch_inference = config['use_batch_inference']
    else:
        use_batch_inference = len(examples) > 10
    
    if use_batch_inference:
        logger.info("Using Mistral batch inference API for efficient processing")
        predictions = model.predict_dataset(
            examples=examples,
            use_batch_inference=True
        )
    else:
        logger.info("Using standard sequential API calls")
        texts = [example.unmasked_text for example in examples]
        predictions = model.predict_batch(texts)
    
    logger.info("Evaluating results...")
    experiment_config = {
        'num_samples': len(examples),
        'use_batch_inference': use_batch_inference,
        'model_config': model_config,
        'fine_tuned_model': fine_tuned_model_name
    }
    
    result = evaluator.evaluate_dataset(
        examples=examples,
        predictions=predictions,
        experiment_name="fine_tuned_mistral",
        model_name=fine_tuned_model_name,
        config=experiment_config
    )
    
    evaluator.print_evaluation_report(result)
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"fine_tuned_{fine_tuned_model_name.replace('/', '_')}.json"
    
    result_dict = result.to_dict()
    
    with open(results_file, 'w') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    
    model.cleanup()
    
    return result

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Test fine-tuned Mistral model on PII detection")
    
    parser.add_argument(
        "--model",
        required=True,
        help="Fine-tuned model name (e.g., 'ft:ministral-8b-latest:suffix')"
    )
    parser.add_argument(
        "--dataset", 
        default="../data/english_pii_43k.jsonl",
        help="Path to dataset file"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--batch-inference", 
        action="store_true",
        help="Force use of batch inference"
    )
    parser.add_argument(
        "--no-batch-inference", 
        action="store_true",
        help="Force use of sequential API calls"
    )
    
    args = parser.parse_args()
    
    config = {
        'model_config': {
            'temperature': 0.1,
            'max_tokens': 1000,
            'max_retries': 3,
            'rate_limit_delay': 0.1
        }
    }
    
    if args.batch_inference:
        config['use_batch_inference'] = True
    elif args.no_batch_inference:
        config['use_batch_inference'] = False
    
    try:
        result = run_finetuned_experiment(
            fine_tuned_model_name=args.model,
            dataset_path=args.dataset,
            num_samples=args.samples,
            config=config
        )
        
        print(f"\nFine-tuned model evaluation completed successfully!")
        print(f"F1-Score: {result.f1_score:.3f}")
        print(f"Precision: {result.precision:.3f}")
        print(f"Recall: {result.recall:.3f}")
        print(f"Results saved to: ../results/fine_tuned_{args.model.replace('/', '_')}.json")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 