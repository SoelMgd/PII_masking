# Token Classification for PII Detection - Technical Context

## 🎯 **Project Overview**

This project implements **token-level classification** for **Personal Identifiable Information (PII) detection** using a fine-tuned Mistral model. The goal is to classify each token in a text as either non-PII ("O") or a specific PII type (EMAIL, PERSON, etc.).

## 📊 **Dataset Structure & Processing**

### **Original Data Format**
- **Source**: PII datasets with English and French examples
- **Format**: Each example contains:
  - `unmasked_text`: Original text with PII
  - `span_labels`: List of `[start, end, label_type]` indicating PII locations
  - Example: `"Hello John Smith"` with span `[6, 16, "PERSON"]`

### **Token Classification Dataset Creation**

#### **Step 1: Tokenization with Mistral Tekken v3**
```python
# Input text: "Hello John Smith, email: john@example.com"
# Tekken v3 tokenization produces:
tokens = ["Hello", " John", " Smith", ",", " email", ":", " j", "ohn", "@example", ".com"]
token_ids = [12345, 6789, 1234, 567, 8901, 234, 567, 890, 1234, 567]  # Actual token IDs
```

#### **Step 2: Span Alignment**
```python
# Original spans: [[6, 16, "PERSON"], [24, 40, "EMAIL"]]
# Align spans with token boundaries:
# "John Smith" (chars 6-16) → tokens 1-2
# "john@example.com" (chars 24-40) → tokens 6-9
```

#### **Step 3: Label Assignment**
```python
# Assign labels to each token:
labels = ["O", "PERSON", "PERSON", "O", "O", "O", "EMAIL", "EMAIL", "EMAIL", "EMAIL"]
label_ids = [0, 15, 15, 0, 0, 0, 11, 11, 11, 11]  # Converted to IDs
```

### **Final Dataset Structure**
Each processed example contains:
```python
{
    'texts': ["Hello John Smith, email: john@example.com"],
    'tokens': [["Hello", " John", " Smith", ",", " email", ":", " j", "ohn", "@example", ".com"]],
    'token_ids': [[12345, 6789, 1234, 567, 8901, 234, 567, 890, 1234, 567]],  # ✅ PRE-COMPUTED
    'labels': [["O", "PERSON", "PERSON", "O", "O", "O", "EMAIL", "EMAIL", "EMAIL", "EMAIL"]],
    'label_ids': [[0, 15, 15, 0, 0, 0, 11, 11, 11, 11]],  # ✅ PRE-ALIGNED
    'label_to_id': {"O": 0, "PERSON": 15, "EMAIL": 11, ...},
    'id_to_label': {0: "O", 15: "PERSON", 11: "EMAIL", ...},
    'num_labels': 57
}
```

## 🏗️ **Model Architecture**

### **Base Model**: Ministral-8B-Instruct-2410
- **Type**: Mistral transformer model (8B parameters)
- **Precision**: bfloat16 for stability
- **Backbone**: Frozen (only classification head is trained)

### **Classification Head**
```python
class MinistralTokenClassifier(nn.Module):
    def __init__(self, model_name, num_labels=57):
        # Frozen backbone
        self.backbone = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Trainable classification head
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(4096, 57)  # hidden_size → num_labels
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get hidden states for all tokens
        outputs = self.backbone(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, 4096]
        
        # Classify each token
        logits = self.classifier(hidden_states)  # [batch, seq_len, 57]
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, 57), labels.view(-1))
            return {'loss': loss, 'logits': logits}
        
        return {'logits': logits}
```

## 🚀 **Training Process**

### **Key Optimizations Applied**
1. **Pre-computed Token IDs**: No re-tokenization during training (major speedup)
2. **Frozen Backbone**: Only ~230k parameters trained vs 8B total
3. **bfloat16**: More stable than float16
4. **Efficient Dataset**: FastPIITokenDataset with minimal overhead

### **Training Configuration**
```python
batch_size = 12
max_length = 96  # Sequence length
learning_rate = 1e-5
gradient_accumulation_steps = 3  # Effective batch = 36
num_epochs = 1
```

### **Data Flow During Training**
```python
# Input batch: [batch_size, max_length] = [12, 96]
input_ids = [[12345, 6789, ...], [...], ...]      # Pre-computed token IDs
labels = [[0, 15, 15, ...], [...], ...]           # Pre-aligned label IDs

# Forward pass:
hidden_states = backbone(input_ids)  # [12, 96, 4096]
logits = classifier(hidden_states)   # [12, 96, 57]

# Loss calculation:
loss = CrossEntropyLoss(logits.view(-1, 57), labels.view(-1))  # Flatten for loss
# 12 * 96 = 1,152 token predictions per batch
```

## 📈 **Dataset Statistics**
- **Training examples**: 94,463
- **Validation examples**: 10,495
- **Total labels**: 57 PII types
- **Average sequence length**: 58.1 tokens
- **Non-O token ratio**: 43.7% (good balance)

## 🔧 **Technical Challenges & Solutions**

### **Challenge 1: Tokenizer Mismatch**
- **Problem**: Dataset uses Mistral Tekken v3, but training might use different tokenizer
- **Solution**: Pre-compute token IDs during dataset creation, use them directly

### **Challenge 2: Label Alignment**
- **Problem**: PII spans don't align perfectly with token boundaries
- **Solution**: Character-level mapping → token-level alignment with overlap handling

### **Challenge 3: Memory & Speed**
- **Problem**: Token classification = 96x more predictions than text classification
- **Solution**: Frozen backbone + pre-computed tokens + optimized batch sizes

### **Challenge 4: Device Consistency**
- **Problem**: Multi-GPU distribution causes device mismatch errors
- **Solution**: Force all components on same device with explicit `.to(device)`

## 🎯 **Expected Performance**
- **Training time**: 30-45 minutes per epoch (vs 7+ hours without optimizations)
- **Memory usage**: ~15GB GPU memory
- **Convergence**: F1-score should improve from ~0.3 to 0.7+ within 1 epoch

## 🐛 **Common Issues & Debugging**

### **Device Errors**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!
```
**Solution**: Use `model.to(device)` after initialization

### **Forward Method Errors**
```
TypeError: _forward_unimplemented() got an unexpected keyword argument 'input_ids'
```
**Solution**: Check class definition syntax, indentation, and method signature

### **NaN Loss**
**Causes**: dtype mismatch (float16 instability), invalid labels, extreme gradients
**Solutions**: Use bfloat16, validate label ranges, gradient clipping

## 📁 **File Structure**
```
experiments/token_classification/
├── dataset_processing.py          # Creates token classification dataset
├── fast_dataset.py               # Optimized PyTorch Dataset class
├── kaggle_finetuning.py          # Training script (structured for notebooks)
├── dataset_check.py              # Dataset validation and visualization
└── token-classification-finetuning.ipynb  # Kaggle notebook
```

## 🎯 **Current Status**
The project has been optimized for speed and stability. The main remaining issue is a `TypeError` in the forward method during training, likely due to a syntax error in the model class definition or device placement issues.

---

**This context should provide complete understanding of the token classification pipeline for debugging purposes.** 