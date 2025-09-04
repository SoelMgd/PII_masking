#!/usr/bin/env python3
"""
Mistral Efficient NER-based PII masking experiment.

This script uses a more efficient approach:
1. Prompts the model to return a JSON with identified PII entities
2. Uses regex matching to reconstruct masked text
3. Evaluates using exact span matching
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
from pii_masking.base_model import PromptBasedModel
from pii_masking.custom_evaluator import CustomPIIEvaluator, CustomEvaluationResult

from mistralai import Mistral as MistralClient, File
from mistralai.models import UserMessage

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MistralEfficientNERModel(PromptBasedModel):
    """Mistral model for efficient PII detection using JSON output."""
    
    def __init__(self, api_key: str, model_name: str = "mistral-large-latest", config: dict = None):
        super().__init__(model_name, config)
        self.api_key = api_key
        self.client = None
        
        self.prompt_template = """You are an expert in Personal Identifiable Information (PII) detection.

Your task is to analyze the provided text and identify ALL PII entities, returning them as a structured JSON.

PII Entity Types to detect:
PREFIX, FIRSTNAME, LASTNAME, DATE, TIME, PHONEIMEI, USERNAME, GENDER, CITY, STATE, URL, JOBAREA, EMAIL, 
JOBTYPE, COMPANYNAME, JOBTITLE, STREET, SECONDARYADDRESS, COUNTY, AGE, USERAGENT, ACCOUNTNAME, ACCOUNTNUMBER, 
CURRENCYSYMBOL, AMOUNT, CREDITCARDISSUER, CREDITCARDNUMBER, CREDITCARDCVV, PHONENUMBER, SEX, IP, ETHEREUMADDRESS, 
BITCOINADDRESS, MIDDLENAME, IBAN, VEHICLEVRM, DOB, PIN, CURRENCY, PASSWORD, CURRENCYNAME, LITECOINADDRESS, CURRENCYCODE, 
BUILDINGNUMBER, ORDINALDIRECTION, MASKEDNUMBER, ZIPCODE, BIC, IPV4, IPV6, MAC, NEARBYGPSCOORDINATE, VEHICLEVIN, EYECOLOR, 
HEIGHT, SSN

Instructions:
1. Identify each PII element in the text
2. Extract the EXACT substring from the original text (preserve case, punctuation, etc.)
3. Return ONLY a JSON object with the format: {"PII": {"ENTITY_TYPE": ["substring1", "substring2", ...]}}
4. If no PII is found, return: {"PII": {}}
5. Do NOT include non-PII text or "O" labels - only actual PII entities

{examples}

Text to analyze:
{text}

JSON Output:"""
    
    def create_prompt(self, text: str, few_shot_examples: Optional[List[PIIExample]] = None) -> str:
        """Create a prompt for JSON-based PII detection."""
        examples_text = ""
        if few_shot_examples:
            examples_text = "\nExamples:\n\n"
            
            for i, example in enumerate(few_shot_examples[-3:], 1):  # Limit to 3 examples
                pii_json = self._spans_to_json(example.unmasked_text, example.span_labels)
                
                examples_text += f"Example {i}:\n"
                examples_text += f"Text: {example.unmasked_text}\n"
                examples_text += f"JSON: {json.dumps(pii_json, ensure_ascii=False)}\n\n"
        
        prompt = self.prompt_template.replace("{examples}", examples_text)
        prompt = prompt.replace("{text}", text)
        
        return prompt
    
    def _spans_to_json(self, text: str, span_labels: List[List]) -> Dict[str, Dict[str, List[str]]]:
        """
        Convert span labels to JSON format for few-shot examples.
        
        Returns:
            Dictionary in PIIPrediction format: {"PII": {"ENTITY_TYPE": ["substring", ...]}}
        """
        pii_dict = {}
        
        for span in span_labels:
            if len(span) >= 3:
                start, end, label = span[0], span[1], span[2]
                
                entity_type = label.replace('B-', '').replace('I-', '')
                if '_' in entity_type:
                    entity_type = entity_type.split('_')[0]
                
                if entity_type == 'O':
                    continue
                
                substring = text[start:end]
                
                if entity_type not in pii_dict:
                    pii_dict[entity_type] = []
                
                if substring not in pii_dict[entity_type]:
                    pii_dict[entity_type].append(substring)
        
        return {"PII": pii_dict}
    
    def initialize(self) -> bool:
        """Initialize the Mistral client."""
        try:
            self.client = MistralClient(api_key=self.api_key)
            self.is_initialized = True
            logger.info(f"Mistral client initialized with model: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Mistral client: {e}")
            return False
    
    def predict_single(self, text: str, few_shot_examples: Optional[List[PIIExample]] = None, **kwargs) -> PIIPrediction:
        """
        Predict PII entities for a single text using Mistral API.
        
        Returns:
            Complete PIIPrediction object with entities, spans, and masked text
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")
        
        prompt = self.create_prompt(text, few_shot_examples)
        logger.info(f"Prompt: {prompt}")
        
        try:
            messages = [UserMessage(role="user", content=prompt)]
            response = self.client.chat.complete(
                model=self.model_name,
                messages=messages,
                temperature=self.config.get('temperature', 0.1),
                max_tokens=self.config.get('max_tokens', 1000),
                response_format={"type": "json_object"}
            )
            
            json_prediction = response.choices[0].message.content.strip()
            logger.debug(f"LLM output: {str(json_prediction)}")
            
            return PIIPrediction.from_json_and_text(json_prediction, text)
            
        except Exception as e:
            logger.error(f"Error in Mistral API call: {e}")
            return PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
    
    def predict_batch(self, texts: List[str], **kwargs) -> List[PIIPrediction]:
        """
        Predict batch with rate limiting.
        
        Returns:
            List of complete PIIPrediction objects
        """
        predictions = []
        rate_limit_delay = self.config.get('rate_limit_delay', 0.1)
        
        for i, text in enumerate(texts):
            try:
                prediction = self.predict_single(text, **kwargs)
                predictions.append(prediction)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(texts)} texts")
                
                if i < len(texts) - 1:
                    time.sleep(rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Error predicting text {i}: {e}")
                # Empty prediction on error
                empty_prediction = PIIPrediction(entities={}, spans=[], masked_text=text, original_text=text)
                predictions.append(empty_prediction)
        
        return predictions
    
    def predict_batch_inference(self, examples: List[PIIExample], 
                               few_shot_examples: Optional[List[PIIExample]] = None,
                               **kwargs) -> List[PIIPrediction]:
        """
        Predict using Mistral's batch inference API for efficient processing.
        
        Returns:
            List of complete PIIPrediction objects
        """
        if not self.is_initialized:
            raise RuntimeError("Model not initialized")

        logger.info(f"Starting batch inference for {len(examples)} examples")

        batch_file = self._create_batch_input_file(examples, few_shot_examples)
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
                # Empty prediction on error
                empty_prediction = PIIPrediction(
                    entities={}, 
                    spans=[], 
                    masked_text=examples[i].unmasked_text, 
                    original_text=examples[i].unmasked_text
                )
                predictions.append(empty_prediction)

        return predictions
    
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
    
    def _create_batch_input_file(self, examples: List[PIIExample], 
                                few_shot_examples: Optional[List[PIIExample]] = None) -> File:
        """Create a JSONL file for batch inference."""
        buffer = BytesIO()
        
        for idx, example in enumerate(examples):
            try:
                prompt = self.create_prompt(example.unmasked_text, few_shot_examples)
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
                file_name=f"pii_efficient_batch_{len(examples)}.jsonl",
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
            metadata={"job_type": "pii_efficient", "framework": "pii-masking-efficient"}
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

def run_mistral_efficient_experiment(
    dataset_path: str,
    num_samples: int = 100,
    use_few_shot: bool = False,
    num_few_shot: int = 3,
    config: Dict[str, Any] = None
) -> CustomEvaluationResult:
    """
    Run the Mistral efficient NER experiment.
    
    Args:
        dataset_path: Path to the dataset file
        num_samples: Number of samples to evaluate
        use_few_shot: Whether to use few-shot examples
        num_few_shot: Number of few-shot examples to use
        config: Configuration dictionary
        
    Returns:
        CustomEvaluationResult object with evaluation metrics
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = config.get("model_name", "mistral-large-latest")
    
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")
    
    data_loader = PIIDataLoader(data_dir=Path(dataset_path).parent)
    evaluator = CustomPIIEvaluator()
    
    logger.info(f"Loading dataset from {dataset_path}")
    examples = data_loader.load_dataset(
        language='english',
        max_samples=num_samples + (num_few_shot if use_few_shot else 0),
        shuffle=True,
        seed=42
    )
    
    if not examples:
        raise ValueError("No examples loaded from dataset")
    
    logger.info(f"Loaded {len(examples)} examples")
    
    model_config = config.get('model_config', {}) if config else {}
    model = MistralEfficientNERModel(api_key=api_key, model_name=model_name, config=model_config)
    
    if not model.initialize():
        raise RuntimeError("Failed to initialize Mistral model")
    
    few_shot_examples = None
    if use_few_shot and num_few_shot > 0:
        few_shot_examples = examples[:num_few_shot]
        examples = examples[num_few_shot:]
        logger.info(f"Using {num_few_shot} few-shot examples")
    
    logger.info("Generating predictions...")
    
    if 'use_batch_inference' in config:
        use_batch_inference = config['use_batch_inference']
    else:
        use_batch_inference = len(examples) > 50
    
    if use_batch_inference:
        logger.info("Using Mistral batch inference API for efficient processing")
        predictions = model.predict_dataset(
            examples=examples,
            use_batch_inference=True,
            few_shot_examples=few_shot_examples
        )
    else:
        logger.info("Using standard sequential API calls")
        texts = [example.unmasked_text for example in examples]
        predictions = model.predict_batch(texts, few_shot_examples=few_shot_examples)
    
    logger.info("Evaluating results...")
    experiment_config = {
        'num_samples': len(examples),
        'use_few_shot': use_few_shot,
        'num_few_shot': num_few_shot if use_few_shot else 0,
        'use_batch_inference': use_batch_inference,
        'model_config': model_config
    }
    
    result = evaluator.evaluate_dataset(
        examples=examples,
        predictions=predictions,
        experiment_name="mistral_efficient_ner",
        model_name=model_name,
        config=experiment_config
    )
    
    evaluator.print_evaluation_report(result)
    
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "mistral_efficient_ner.json"
    
    result_dict = result.to_dict()
    
    with open(results_file, 'w') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    
    model.cleanup()
    
    return result

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Mistral Efficient NER-based PII masking")
    
    parser.add_argument(
        "--dataset", 
        default="../data/english_pii_43k.jsonl",
        help="Path to dataset file"
    )
    parser.add_argument(
        "--model",
        default="mistral-large-latest",
        help="Model to use (mistral-medium-latest, mistral-large-latest, mistral-small-latest)"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--few-shot", 
        action="store_true",
        help="Use few-shot examples"
    )
    parser.add_argument(
        "--num-few-shot", 
        type=int, 
        default=3,
        help="Number of few-shot examples"
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
        'model_name': args.model,
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
        result = run_mistral_efficient_experiment(
            dataset_path=args.dataset,
            num_samples=args.samples,
            use_few_shot=args.few_shot,
            num_few_shot=args.num_few_shot,
            config=config
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"F1-Score: {result.f1_score:.3f}")
        print(f"Results saved to: ../results/mistral_efficient_ner.json")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 