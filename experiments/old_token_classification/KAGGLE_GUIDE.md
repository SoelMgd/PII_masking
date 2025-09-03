# 🎯 Guide Kaggle : Fine-tuning Mistral pour Classification de Tokens PII

Ce guide explique comment utiliser les scripts de classification par token sur Kaggle pour fine-tuner un modèle Mistral avec une approche "BERT-like".

## 📋 Vue d'ensemble

Cette approche diffère du fine-tuning API en :
- **Gel du backbone Mistral** : Seule la tête de classification est entraînée
- **Classification par token** : Chaque token reçoit un label PII (comme BERT NER)
- **Coût réduit** : Moins de paramètres à entraîner
- **Contrôle total** : Entraînement local avec PyTorch

## 🛠️ Étape 1 : Préparation des Données (Local)

### Génération du dataset

```bash
# Depuis le répertoire experiments/token_classification/
uv run dataset_processing.py --data-dir ../../data --max-english 1000 --max-french 1000
```

### Fichiers générés
- `../../data/token_classification/train_dataset.pkl` : Dataset d'entraînement
- `../../data/token_classification/val_dataset.pkl` : Dataset de validation  
- `../../data/token_classification/label_mappings.json` : Mapping des labels
- `../../data/token_classification/dataset_stats.json` : Statistiques

### Upload vers Kaggle
1. Créer un dataset Kaggle avec ces fichiers
2. Ou utiliser l'API Kaggle pour upload automatique

## 🚀 Étape 2 : Fine-tuning sur Kaggle

### Configuration Kaggle requise
- **GPU** : P100 ou T4 minimum (16GB VRAM recommandé)
- **Internet** : Activé pour télécharger le modèle Mistral
- **Durée** : 2-4 heures selon la taille du dataset

### Script d'entraînement

```python
# Dans un notebook Kaggle
!pip install transformers torch accelerate sentencepiece

# Copier le contenu de kaggle_finetuning.py
# Puis exécuter :

python kaggle_finetuning.py \
    --train-dataset /kaggle/input/pii-token-data/train_dataset.pkl \
    --val-dataset /kaggle/input/pii-token-data/val_dataset.pkl \
    --output-dir ./mistral_token_classifier \
    --model-name mistralai/Mistral-7B-v0.1 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --num-epochs 3 \
    --max-length 512
```

### Paramètres recommandés

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `batch_size` | 4-8 | Selon la VRAM disponible |
| `learning_rate` | 2e-5 | Taux d'apprentissage pour la tête |
| `num_epochs` | 3-5 | Nombre d'époques |
| `max_length` | 512 | Longueur max des séquences |
| `gradient_accumulation_steps` | 4 | Pour simuler des batch plus grands |

### Monitoring de l'entraînement

Le script affiche :
- **Loss par epoch** : Doit diminuer progressivement
- **F1-Score validation** : Métrique principale à optimiser
- **Paramètres entraînables** : Seulement ~1% du modèle total

## 📊 Étape 3 : Évaluation sur Kaggle

### Script d'évaluation

```python
# Après l'entraînement, évaluer le modèle
python eval.py \
    --model-dir ./mistral_token_classifier \
    --test-dataset /kaggle/input/pii-token-data/val_dataset.pkl \
    --output-file evaluation_results.json \
    --batch-size 16
```

### Métriques calculées
- **F1-Score** (Weighted, Macro, Micro)
- **Précision/Rappel par classe**
- **Matrice de confusion**
- **Analyse d'erreurs détaillée**

## 🔧 Structure des Scripts

### `dataset_processing.py`
```python
# Fonctionnalités principales :
- Tokenisation avec Mistral tokenizer
- Alignement des labels PII avec les tokens
- Création de datasets PyTorch compatibles
- Gestion multilingue (anglais/français)
```

### `kaggle_finetuning.py`
```python
# Architecture du modèle :
class MistralTokenClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        self.backbone = AutoModel.from_pretrained(model_name)  # Gelé
        self.classifier = nn.Linear(hidden_size, num_labels)   # Entraînable
```

### `eval.py`
```python
# Évaluation complète :
- Chargement du modèle entraîné
- Prédictions sur dataset de test
- Calcul de métriques détaillées
- Analyse d'erreurs et exemples
```

## 💡 Optimisations Kaggle

### Gestion mémoire
```python
# Dans kaggle_finetuning.py
torch_dtype=torch.float16,  # Précision réduite
device_map="auto",          # Distribution automatique
gradient_accumulation_steps=4  # Batch virtuel plus grand
```

### Accélération
```python
# Utiliser gradient checkpointing si nécessaire
model.gradient_checkpointing_enable()

# Optimiser le DataLoader
num_workers=2,  # Parallélisation
pin_memory=True  # Transfert GPU plus rapide
```

## 📈 Résultats Attendus

### Performance cible
- **F1-Score** : 0.75-0.85 (selon la taille du dataset)
- **Temps d'entraînement** : 2-4 heures sur T4
- **Mémoire utilisée** : ~12-14GB VRAM

### Comparaison avec les approches
| Approche | F1-Score | Coût | Temps |
|----------|----------|------|-------|
| API Fine-tuning | 0.87 | $15-25 | 30min |
| Token Classification | 0.75-0.85 | Gratuit | 2-4h |
| Few-shot Baseline | 0.67 | $2-5 | 10min |

## 🐛 Dépannage Kaggle

### Erreurs communes

1. **CUDA Out of Memory**
   ```python
   # Réduire batch_size à 2 ou 4
   # Augmenter gradient_accumulation_steps
   # Utiliser torch.float16
   ```

2. **Modèle trop lent à charger**
   ```python
   # Utiliser un modèle plus petit
   model_name = "mistralai/Mistral-7B-v0.1"  # Au lieu de Mistral-8x7B
   ```

3. **Alignement des tokens**
   ```python
   # Le script utilise un fallback robuste
   # Vérifier les logs pour les warnings
   ```

### Optimisations de performance
```python
# Dans le notebook Kaggle
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Éviter les warnings
torch.backends.cudnn.benchmark = True  # Optimiser CUDNN
```

## 📝 Exemple de Notebook Kaggle

```python
# Cellule 1: Installation
!pip install transformers torch accelerate sentencepiece scikit-learn tqdm

# Cellule 2: Imports et configuration
import torch
import json
from pathlib import Path

# Vérifier GPU
print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Cellule 3: Copier les scripts
# Copier le contenu de kaggle_finetuning.py et eval.py

# Cellule 4: Entraînement
# Exécuter le script d'entraînement

# Cellule 5: Évaluation
# Exécuter le script d'évaluation

# Cellule 6: Sauvegarde des résultats
# Sauvegarder le modèle et les résultats
```

## 🎯 Conseils pour Optimiser les Résultats

### Hyperparamètres
- **Learning Rate** : Commencer par 2e-5, réduire si instable
- **Batch Size** : Maximiser selon la VRAM disponible
- **Epochs** : 3-5 epochs suffisent généralement

### Données
- **Équilibrage** : Mélanger anglais/français 50/50
- **Taille** : 1000+ exemples par langue recommandé
- **Qualité** : Vérifier l'alignement des labels

### Architecture
- **Dropout** : 0.1 par défaut, augmenter si overfitting
- **Gel** : Garder le backbone gelé pour la stabilité
- **Initialisation** : Utiliser l'initialisation par défaut

---

**Note** : Ce guide assume l'utilisation de Kaggle avec GPU. Pour CPU uniquement, réduire drastiquement la taille du modèle et du dataset. 