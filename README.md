# PII Masking Evaluation Framework

**Mistral AI Take-Home Project** - Applied AI Engineer Position

A comprehensive framework for evaluating and comparing PII (Personally Identifiable Information) masking approaches using various LLM techniques including prompting and fine-tuning.

## Project Overview

This project evaluates whether sophisticated **prompt engineering** with Mistral API can compete with **fine-tuned models** for PII masking tasks, enabling data-driven decisions about the optimal approach.

### Dataset: AI4Privacy PII-200k
- **209k examples** across 4 languages (English, French, German, Italian)
- **54 PII classes** covering comprehensive privacy scenarios
- **Human-validated** synthetic data with no privacy violations

## 🏗️ Project Structure

```
pii-masking-200k/
├── src/pii_masking/           # Core framework package
│   ├── __init__.py               # Package exports
│   ├── data_loader.py            # Dataset loading utilities
│   ├── text_processing.py       # Text processing and reconstruction
│   ├── custom_evaluator.py      # Evaluation with exact position matching
│   └── base_model.py             # Abstract model interfaces
│
├── experiments/               # Research experiments
│   ├── mistral_prompting_baseline.py    # Mistral API prompting
│   ├── mistral_finetuning.py           # Mistral fine-tuning experiments
│   ├── bert_test.py                     # BERT token classification
│   └── evaluation_comparison.py        # Cross-method comparison
│
├── space/                     # Production deployment (HuggingFace Space)
│   ├── app.py                    # FastAPI production server
│   ├── inference/                # Production inference services
│   │   ├── mistral_prompting.py  # Mistral API service with batching
│   │   ├── bert_classif.py       # BERT inference service (CPU-optimized)
│   │   └── ocr_service.py        # PDF OCR processing (Mistral OCR)
│   ├── static/                   # Frontend assets
│   │   └── index.html            # Web interface with PDF drag-and-drop
│   ├── src/                      # Shared core package (symlink)
│   ├── pyproject.toml           # Space dependencies
│   ├── Dockerfile               # Container configuration
│   └── README.md                # Space deployment guide
│
├── data/                      # Dataset files (AI4Privacy PII-200k)
├── configs/                  # Configuration files
│
├── 📋 pyproject.toml             # Project configuration
├── 🔧 .env.example               # Environment variables template
└── 📖 README.md                  # This file
```

### Repository Components

#### 🔬 **Research & Development** (`src/` + `experiments/`)
- **Core framework**: Reusable PII detection and evaluation components
- **Experiments**: Comparative studies between prompting, fine-tuning, and token classification
- **Benchmarking**: Systematic evaluation on AI4Privacy PII-200k dataset

#### 🚀 **Production Deployment** (`space/`)
- **Web application**: FastAPI server with modern frontend
- **Multi-method support**: Mistral prompting, fine-tuning, and BERT classification
- **Advanced features**: PDF processing with OCR, entity selection, drag-and-drop UI
- **HuggingFace Space**: Ready for production deployment at scale

## 🚀 Quick Start

### 1. Installation (avec UV - recommandé)

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

# Quick test (20 samples)
uv run mistral_baseline.py --samples 20 --few-shot

# Full evaluation (500 samples)
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

### 3. BERT Token Classification (`bert_test.py`)
- **Approach**: Fine-tuned BERT for token-level PII detection
- **Models**: DistilBERT, BERT-base optimized for CPU inference
- **Features**: Fast inference, offline deployment capability

### 4. Cross-Method Evaluation (`evaluation_comparison.py`)
- **Comprehensive comparison** across all three approaches
- **Performance metrics**: Precision, Recall, F1-Score by entity type
- **Cost analysis**: API costs vs. fine-tuning investment


## Production Deployment (HuggingFace Space)

The `space/` directory contains a complete production-ready web application deployed on HuggingFace Spaces.

### Features
- **Multi-method PII detection**: Mistral prompting, fine-tuning, and BERT classification
- **PDF processing**: Drag-and-drop PDF upload with Mistral OCR integration
- **Entity selection**: Granular control over which PII types to mask
- **Optimized inference**: CPU-optimized BERT, batched Mistral API calls
- **Modern UI**: Responsive interface with real-time feedback

### Quick Deploy
```bash
cd space
uv sync
uv run app.py
# Visit http://localhost:7860
```

### Live Demo
**[Try the live demo on HuggingFace Spaces](https://huggingface.co/spaces/SoelMgd/pii_masking)**

### Architecture
- **Backend**: FastAPI with async processing
- **Frontend**: Modern HTML/CSS/JS with drag-and-drop
- **Models**: HuggingFace Hub integration for BERT, Mistral API
- **OCR**: Mistral OCR service for PDF text extraction


### Adding New Experiments

1. **Inherit from base classes**:
   ```python
   from pii_masking.base_model import PromptBasedModel
   ```

2. **Use standard evaluation**:
   ```python
   from pii_masking import PIIEvaluator, EvaluationResult
   
   evaluator = PIIEvaluator(strict_matching=True)
   result = evaluator.evaluate_dataset(examples, predictions, ...)
   ```

3. **Follow naming convention**: `{model_type}_{approach}.py`

## 📋 Configuration

### Environment Variables
```bash
MISTRAL_API_KEY=your_api_key_here
MISTRAL_MODEL=mistral-large-latest
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
- **API Approach**: ~$0.05-0.15 per prediction, immediate deployment
- **Fine-tuning**: ~$50-100 initial cost, 2-3 days development
- **Break-even**: ~1000 predictions/month


## Contributing

This is a take-home project, but the framework is designed for extensibility:

1. **Add new models**: Inherit from `BasePIIModel`
2. **Extend evaluation**: Modify `PIIEvaluator` for new metrics
3. **New datasets**: Extend `PIIDataLoader` for different formats

## 📄 License

MIT License - See LICENSE file for details.

---

**Ready for Production Deployment!**

*This framework demonstrates a systematic approach to AI model evaluation, combining technical rigor with business pragmatism - essential skills for an Applied AI Engineer at Mistral AI.*
