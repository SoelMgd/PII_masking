#!/usr/bin/env python3
"""
Fine-tuning script for Mistral models on PII detection.

This script:
1. Uploads training and validation datasets to Mistral
2. Creates and starts a fine-tuning job
3. Monitors the job progress
4. Provides the fine-tuned model name when completed
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional

from mistralai import Mistral
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MistralFineTuner:
    """Fine-tuning manager for Mistral models."""
    
    def __init__(self, api_key: str):
        """
        Initialize the fine-tuner.
        
        Args:
            api_key: Mistral API key
        """
        self.client = Mistral(api_key=api_key)
        self.api_key = api_key
    
    def upload_file(self, file_path: Path, file_name: str) -> str:
        """
        Upload a file to Mistral for fine-tuning.
        
        Args:
            file_path: Path to the file to upload
            file_name: Name for the uploaded file
            
        Returns:
            File ID from Mistral
        """
        logger.info(f"Uploading {file_name} from {file_path}")
        
        with open(file_path, 'rb') as f:
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": file_name,
                    "content": f,
                }
            )
        
        logger.info(f"Uploaded {file_name} with ID: {uploaded_file.id}")
        return uploaded_file.id
    
    def create_fine_tuning_job(
        self,
        model: str,
        training_file_id: str,
        validation_file_id: str,
        hyperparameters: Dict[str, Any] = None,
        auto_start: bool = False,
        suffix: str = None
    ) -> str:
        """
        Create a fine-tuning job.
        
        Args:
            model: Model to fine-tune
            training_file_id: ID of uploaded training file
            validation_file_id: ID of uploaded validation file
            hyperparameters: Training hyperparameters
            auto_start: Whether to start training immediately
            suffix: Optional suffix for the fine-tuned model name
            
        Returns:
            Job ID
        """
        if hyperparameters is None:
            hyperparameters = {
                "training_steps": 100,
                "learning_rate": 1e-4
            }
        
        logger.info(f"Creating fine-tuning job for model: {model}")
        logger.info(f"Hyperparameters: {hyperparameters}")
        
        job_params = {
            "model": model,
            "training_files": [{"file_id": training_file_id, "weight": 1}],
            "validation_files": [validation_file_id],
            "hyperparameters": hyperparameters,
            "auto_start": auto_start
        }
        
        if suffix:
            job_params["suffix"] = suffix
        
        created_job = self.client.fine_tuning.jobs.create(**job_params)
        
        logger.info(f"Created fine-tuning job: {created_job.id}")
        logger.info(f"Job status: {created_job.status}")
        
        return created_job.id
    
    def start_job(self, job_id: str):
        """
        Start a fine-tuning job.
        
        Args:
            job_id: ID of the job to start
        """
        logger.info(f"Starting fine-tuning job: {job_id}")
        self.client.fine_tuning.jobs.start(job_id=job_id)
        logger.info("Job started successfully")
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a fine-tuning job.
        
        Args:
            job_id: ID of the job
            
        Returns:
            Job information dictionary
        """
        job = self.client.fine_tuning.jobs.get(job_id=job_id)
        
        return {
            "id": job.id,
            "status": job.status,
            "model": job.model,
            "fine_tuned_model": getattr(job, 'fine_tuned_model', None),
            "created_at": job.created_at,
            "training_files": job.training_files,
            "validation_files": job.validation_files,
            "hyperparameters": job.hyperparameters,
            "trained_tokens": getattr(job, 'trained_tokens', None),
            "epochs": getattr(job, 'epochs', None)
        }
    
    def monitor_job(self, job_id: str, check_interval: int = 30) -> Dict[str, Any]:
        """
        Monitor a fine-tuning job until completion.
        
        Args:
            job_id: ID of the job to monitor
            check_interval: Seconds between status checks
            
        Returns:
            Final job information
        """
        logger.info(f"Monitoring job {job_id} (checking every {check_interval}s)")
        
        while True:
            job_info = self.get_job_status(job_id)
            status = job_info["status"]
            
            logger.info(f"Job {job_id} status: {status}")
            
            if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                logger.info(f"Job completed with status: {status}")
                
                if status == "SUCCEEDED":
                    logger.info(f"Fine-tuned model: {job_info['fine_tuned_model']}")
                elif status == "FAILED":
                    logger.error("Fine-tuning job failed")
                elif status == "CANCELLED":
                    logger.warning("Fine-tuning job was cancelled")
                
                return job_info
            
            if job_info.get("trained_tokens"):
                logger.info(f"Trained tokens: {job_info['trained_tokens']}")
            if job_info.get("epochs"):
                logger.info(f"Epochs: {job_info['epochs']}")
            
            time.sleep(check_interval)
    
    def list_jobs(self, limit: int = 10) -> list:
        """
        List recent fine-tuning jobs.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job information
        """
        jobs = self.client.fine_tuning.jobs.list(page_size=limit)
        return [self.get_job_status(job.id) for job in jobs.data]
    
    def cancel_job(self, job_id: str):
        """
        Cancel a fine-tuning job.
        
        Args:
            job_id: ID of the job to cancel
        """
        logger.info(f"Cancelling job: {job_id}")
        self.client.fine_tuning.jobs.cancel(job_id=job_id)
        logger.info("Job cancelled successfully")

def main():
    """Main function for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Mistral models on PII detection")
    
    parser.add_argument(
        "--train-file",
        default="../data/fine_tuning/train.jsonl",
        help="Path to training JSONL file"
    )
    parser.add_argument(
        "--val-file",
        default="../data/fine_tuning/validation.jsonl",
        help="Path to validation JSONL file"
    )
    parser.add_argument(
        "--model",
        default="ministral-8b-latest",
        choices=["open-mistral-7b", "mistral-small-latest", "ministral-8b-latest", "ministral-3b-latest"],
        help="Model to fine-tune"
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=100,
        help="Number of training steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--suffix",
        help="Suffix for the fine-tuned model name"
    )
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Start training immediately after job creation"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor job progress until completion"
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List recent fine-tuning jobs"
    )
    parser.add_argument(
        "--job-id",
        help="Job ID for monitoring or starting a specific job"
    )
    parser.add_argument(
        "--start-job",
        action="store_true",
        help="Start a previously created job"
    )
    parser.add_argument(
        "--cancel-job",
        action="store_true",
        help="Cancel a job"
    )
    
    args = parser.parse_args()
    
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    
    if not api_key:
        logger.error("MISTRAL_API_KEY environment variable not set")
        return
    
    fine_tuner = MistralFineTuner(api_key)
    
    try:
        if args.list_jobs:
            logger.info("Listing recent fine-tuning jobs...")
            jobs = fine_tuner.list_jobs()
            
            print(f"\n📋 Recent Fine-tuning Jobs:")
            print("-" * 80)
            for job in jobs:
                print(f"ID: {job['id']}")
                print(f"Status: {job['status']}")
                print(f"Model: {job['model']}")
                if job['fine_tuned_model']:
                    print(f"Fine-tuned Model: {job['fine_tuned_model']}")
                print(f"Created: {job['created_at']}")
                print("-" * 80)
            return
        
        if args.cancel_job:
            if not args.job_id:
                logger.error("--job-id required for cancelling a job")
                return
            fine_tuner.cancel_job(args.job_id)
            return
        
        if args.start_job:
            if not args.job_id:
                logger.error("--job-id required for starting a job")
                return
            fine_tuner.start_job(args.job_id)
            
            if args.monitor:
                final_job = fine_tuner.monitor_job(args.job_id)
                print(f"\nJob completed!")
                print(f"Status: {final_job['status']}")
                if final_job['fine_tuned_model']:
                    print(f"Fine-tuned Model: {final_job['fine_tuned_model']}")
            return
        
        if args.job_id and args.monitor:
            final_job = fine_tuner.monitor_job(args.job_id)
            print(f"\nJob completed!")
            print(f"Status: {final_job['status']}")
            if final_job['fine_tuned_model']:
                print(f"Fine-tuned Model: {final_job['fine_tuned_model']}")
            return
        
        train_file = Path(args.train_file)
        val_file = Path(args.val_file)
        
        if not train_file.exists():
            logger.error(f"Training file not found: {train_file}")
            return
        
        if not val_file.exists():
            logger.error(f"Validation file not found: {val_file}")
            return
        
        logger.info("Uploading training and validation files...")
        train_file_id = fine_tuner.upload_file(train_file, "pii_train.jsonl")
        val_file_id = fine_tuner.upload_file(val_file, "pii_validation.jsonl")
        
        hyperparameters = {
            "training_steps": args.training_steps,
            "learning_rate": args.learning_rate
        }
        
        job_id = fine_tuner.create_fine_tuning_job(
            model=args.model,
            training_file_id=train_file_id,
            validation_file_id=val_file_id,
            hyperparameters=hyperparameters,
            auto_start=args.auto_start,
            suffix=args.suffix
        )
        
        print(f"\nFine-tuning job created successfully!")
        print(f"Job ID: {job_id}")
        print(f"Model: {args.model}")
        print(f"Training steps: {args.training_steps}")
        print(f"Learning rate: {args.learning_rate}")
        
        if not args.auto_start:
            print(f"\nJob created but not started. To start:")
            print(f"python fine_tuning.py --start-job --job-id {job_id}")
            
            if args.monitor:
                print(f"\nTo monitor progress:")
                print(f"python fine_tuning.py --monitor --job-id {job_id}")
        else:
            print(f"\nJob started automatically!")
            
            if args.monitor:
                final_job = fine_tuner.monitor_job(job_id)
                print(f"\nJob completed!")
                print(f"Status: {final_job['status']}")
                if final_job['fine_tuned_model']:
                    print(f"Fine-tuned Model: {final_job['fine_tuned_model']}")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 