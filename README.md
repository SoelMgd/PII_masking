# PII Masking Evaluation Framework

**Mistral Take-Home Project** 

A comprehensive framework for evaluating and comparing PII (Personally Identifiable Information) masking approaches using various LLM techniques including prompting and fine-tuning.

### Live Demo
**[Try it live](https://huggingface.co/spaces/SoelMgd/pii_masking)**

## Project Overview

This project evaluates whether **prompt engineering** can compete with **fine-tuned models** and **token-classiciation** approach for PII masking tasks, enabling data-driven decisions about the optimal approach.

### Dataset: AI4Privacy PII
- **48k english and 64k french examples**
- **54 PII classes** covering comprehensive privacy scenarios
- **Human-validated** synthetic data with no privacy violations

## Project Structure

```
pii-masking/
├── src/pii_masking/           # Core framework package
│   ├── data_loader.py
│   ├── text_processing.py
│   ├── custom_evaluator.py
│   └── base_model.py
│
├── experiments/               # Research experiments & analysis
│   ├── mistral_prompting_baseline.py
│   ├── mistral_finetuning.py
│   ├── bert_token_classification/
│   │   ├── bert_finetuning_kaggle.ipynb
│   │   └── eval.py
│   ├── llm_token_classification/
│   │   ├── dataset_processing_token.py
│   │   ├── mistral-token-classif.ipynb
│   │   └── eval.py
│   ├── performance_benchmarks/    # Production performance validation
│   │   ├── speed_benchmark.py
│   │   ├── concurrency_benchmark.py
│   │   └── results/
│   └── visualization/
│
├── space/                     # Production deployment
│   ├── app.py                 # FastAPI server with async/sync architecture
│   ├── services/              # Inference services
│   │   ├── base_service.py    # Unified async/sync interface
│   │   ├── mistral_prompting.py
│   │   ├── bert_classif.py
│   │   └── ocr_service.py
│   └── static/
│
└── presentation.pdf           # Presentation
```

### Repository Components

#### **Research & Development** (`src/` + `experiments/`)
- **Core framework**: Reusable PII detection and evaluation components
- **Experiments**: Comparative studies between prompting, fine-tuning, and token classification
- **Benchmarking**: Systematic evaluation on AI4Privacy PII dataset

#### **Production Deployment** (`space/`)
- **Web application**: FastAPI server with modern frontend
- **Multi-method support**: Mistral prompting, fine-tuning, and BERT classification
- **Advanced features**: PDF processing with OCR, entity selection, drag-and-drop UI
- **HuggingFace Space**: Ready for production deployment at scale

## Quick Start

### 1. Installation (avec UV - recommended)

```bash
# Clone and navigate to project
cd pii-masking-200k

# Synchronize dependencies with uv
uv sync

# Copy environment template
cp .env.example .env
# Edit .env with your MISTRAL_API_KEY
```

### 2. Run Baseline Experiment

```bash
cd experiments

uv run mistral_baseline.py --samples 500 --few-shot --output ../results/mistral_baseline_full.json
```


## Evaluation Methodology

### Strict Position Matching 
- **Exact position requirement**: Entities must match type AND exact character positions
- **No partial credit**: Any position mismatch = error (prevents text rewriting issues)
- **Per-class metrics**: F1-score calculated for each PII type individually

### Standardized Output Format
All experiments return `EvaluationResult` objects with:
- **Global metrics**: Precision, Recall, F1-Score, Exact Match Accuracy
- **Per-class metrics**: Performance breakdown by PII type
- **Detailed results**: Individual predictions for error analysis

## Available Experiments

### 1. Mistral API Prompting (`mistral_prompting_baseline.py`)
- **Approach**: Sophisticated prompt engineering + few-shot learning
- **Models**: mistral-large-latest, fine-tuned models
- **Features**: Rate limiting, retry logic, batch processing, configurable prompts
- **Languages**: English & french

```bash
cd experiments
uv run mistral_prompting_baseline.py \
    --samples 100 \
    --few-shot \
    --num-few-shot 3 \
    --temperature 0.1 \
    --output ../results/mistral_prompting.json
```

### 2. Mistral Fine-tuning (`mistral_finetuning.py`)
- **Approach**: Fine-tuned Mistral models via API
- **Features**: Custom model training, specialized PII detection
- **Performance**: Enhanced accuracy on domain-specific patterns
- **Languages**: English & french

### 3. BERT Token Classification (`bert_test.py`)
- **Approach**: Fine-tuned BERT for token-level PII detection
- **Models**: DistilBERT, BERT-base optimized for CPU inference
- **Features**: Fast inference, offline deployment capability
- **Languages**: English only

### 4. Mistral Token Classification (`llm_token_classification/`)
- **Approach**: Fine-tuned Mistral-8B for token-level PII detection
- **Models**: Ministral-8B-Instruct-2410 with frozen backbone + classification head
- **Languages**: English only

```bash
cd experiments/llm_token_classification

# Process dataset for token classification
uv run dataset_processing_token.py \
    --data-dir ../../data \
    --output-dir ./data \
    --max-english 30000

# Evaluate trained model (entity-level metrics)
uv run eval.py \
    --model-path ./ministral_token_classifier \
    --val-dataset ./data/val_dataset.pkl \
    --output-file ./evaluation_results.json
```

### 5. Cross-Method Evaluation (`evaluation_comparison.py`)
- **Comprehensive comparison** across all three approaches
- **Performance metrics**: Precision, Recall, F1-Score by entity type
- **Cost analysis**: API costs vs. fine-tuning investment

### 6. Performance Benchmarks (`performance_benchmarks/`)
- **Speed benchmarking**: Inference time & throughput testing across different text lengths
- **Concurrency validation**: Sequential vs concurrent processing comparison

```bash
cd experiments/performance_benchmarks

# Run comprehensive performance benchmarks
python speed_benchmark.py
python concurrency_benchmark.py 
```


## Production Deployment (HuggingFace Space)

The `space/` directory contains a complete production-ready web application deployed on HuggingFace Spaces.

### Features
- **Multi-method PII detection**: Mistral prompting, fine-tuning, and BERT classification
- **PDF processing**: Drag-and-drop PDF upload with Mistral OCR integration
- **Entity selection**: Granular control over which PII types to mask
- **Smart async/sync architecture**: Optimal performance for different service types
- **Concurrent processing**: Multiple users served simultaneously without blocking
- **Optimized inference**: CPU-optimized BERT, batched Mistral API calls
- **Modern UI**: Responsive interface with real-time feedback

### Quick Deploy
```bash
cd space
uv sync
uv run app.py
# Visit http://localhost:7860
```


### Architecture
- **Backend**: FastAPI with smart async/sync processing architecture
- **Services**: Unified interface with automatic routing (async for Mistral, thread pool for BERT)
- **Frontend**: Modern HTML/CSS/JS with drag-and-drop
- **Models**: HuggingFace Hub integration for BERT, Mistral API
- **OCR**: Mistral OCR service for PDF text extraction
- **Concurrency**: Non-blocking event loop, multiple users served simultaneously

## Technical Architecture: Smart Async/Sync Pattern

### The Challenge
Different AI services have different computational characteristics:
- **BERT**: CPU-bound operations (PyTorch inference) - naturally synchronous
- **Mistral API**: I/O-bound operations (network calls) - naturally asynchronous
- **FastAPI**: Async framework requiring non-blocking operations

### The solution: Unified Interface with Automatic Routing

```python
class BasePIIInferenceService(ABC):
    async def predict(self, text: str) -> PIIPrediction:
        # Try native async first (for API-based models)
        try:
            return await self.predict_async_native(text)
        except NotImplementedError:
            # Fall back to sync in thread pool (for local models)
            return await asyncio.to_thread(self.predict_sync, text)

# BERT Service - implements sync pattern
class BERTInferenceService(BasePIIInferenceService):
    def predict_sync(self, text: str) -> PIIPrediction:
        # Synchronous PyTorch operations
        outputs = self.model(**inputs)
        return self.process_predictions(outputs)

# Mistral Service - implements async pattern  
class MistralPromptingService(BasePIIInferenceService):
    async def predict_async_native(self, text: str) -> PIIPrediction:
        # Native async API calls
        response = await self.client.chat(...)
        return self.process_json_response(response)
```

### Performance Benefits
- **6.0x faster** BERT inference under concurrent load
- **Scalable**: Multiple users served simultaneously (22.1 req/s vs 3.7 req/s)
- **Maintainable**: Clean separation of sync/async concerns
- **Production-validated**: Real performance metrics from comprehensive benchmarks


### Environment Variables
```bash
MISTRAL_API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

## Results Analysis

### JSON Output Format
```json
{
  "experiment_name": "mistral_baseline",
  "model_name": "mistral-large-latest",
  "metrics": {
    "precision": 0.847,
    "recall": 0.793,
    "f1_score": 0.819,
    "exact_match_accuracy": 0.340
  },
  "per_class_metrics": {
    "FIRSTNAME": {"precision": 0.95, "recall": 0.92, "f1_score": 0.93},
    "EMAIL": {"precision": 0.88, "recall": 0.85, "f1_score": 0.86}
  }
}
```

### Visualization 
- Performance comparison charts
- Error analysis by PII type
- Position accuracy heatmaps

## Business Impact

### Privacy Compliance
- **GDPR/HIPAA** automated compliance
- **Real-time** PII detection and masking
- **Multi-language** support for global deployment

### Cost-Benefit Analysis
- **Costs reduction**
- **Faster PII masking time**

