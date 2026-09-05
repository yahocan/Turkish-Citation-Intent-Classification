# Turkish Citation Intent Classification

A comparative NLP study for classifying the intent of citations in Turkish academic writing. The project evaluates classical machine-learning baselines, fine-tuned Turkish Transformer models, zero-shot LLMs, and ensemble methods on a five-class citation-intent task.

## Overview

Given a citation context from an academic text, the goal is to predict its citation intent. The project treats this as a five-class sequence-classification problem and uses **Macro F1** as the primary metric because the label distribution is imbalanced.

The work focuses on rigorous comparison rather than a single model run:

- Classical TF-IDF baselines using word and character n-grams
- Fine-tuned Turkish Transformer encoders
- Zero-shot and few-shot LLM baselines
- Voting and stacking ensemble strategies
- Validation-first model selection without using the hidden competition test set

## Models and Approaches

### Classical baselines

- Multinomial Naive Bayes
- Complement Naive Bayes
- Logistic Regression with word-level TF-IDF n-grams
- Linear SVM with character-level TF-IDF n-grams

### Fine-tuned Transformers

- BERTurk
- ModernBERT-TR
- Turkish ELECTRA mC4

The Transformer models use cleaned `citation_context` input. `<CITE>` and `[REF]` are added as special tokens so they remain intact during tokenization. Class-weighted cross-entropy and early stopping are used to address class imbalance and avoid unnecessary training.

### LLM baselines

- GPT-5.6, zero-shot and few-shot
- Nemotron 3 Ultra, zero-shot and few-shot

LLM outputs are constrained to a structured label format for consistent evaluation.

### Ensembles

- Soft voting
- Hard voting
- Logistic Regression stacking
- TabPFN-3 stacking

## Evaluation Protocol

- Training set: 1,866 examples
- Validation set: 330 examples
- Public labeled test set: 550 examples
- Primary metric: Macro F1

The hidden competition/leaderboard test set is not used for hyperparameter tuning, checkpoint selection, or model selection. This avoids leakage into the final evaluation process.

## Results

| Method | Macro F1 on public test set |
|---|---:|
| Complement Naive Bayes | 0.529 |
| Logistic Regression (TF-IDF) | 0.634 |
| Linear SVM (TF-IDF) | 0.633 |
| GPT-5.6 zero-shot | 0.642 |
| Nemotron 3 Ultra zero-shot | 0.635 |
| Best single model: ModernBERT-TR | 0.671 |
| Hard-vote Transformer ensemble | 0.696 |
| **Soft-vote Transformer ensemble** | **0.705** |

The best result comes from averaging the class probabilities of fine-tuned BERTurk, ModernBERT-TR, and Turkish ELECTRA mC4 models. The soft-vote ensemble improves Macro F1 by 0.034 over the best individual Transformer model.

## Key Findings

- Fine-tuned Turkish Transformers outperform the strongest classical TF-IDF baseline.
- Zero-shot LLMs reach roughly the classical-baseline range but remain below fine-tuned Transformers.
- Soft voting is more reliable than learned stacking in this small-validation-data setting.
- Additional context and document-section metadata did not improve final performance; the extra information appeared to introduce noise or encourage dataset-specific shortcuts.
- Rare citation-intent classes remain the main error source, making Macro F1 more informative than accuracy alone.

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_initial.ipynb       # Dataset setup and initial inspection
│   ├── 02_EDA.ipynb           # Exploratory analysis and baseline experiments
│   ├── 03_FE.ipynb            # Preprocessing and feature engineering
│   ├── 04_training.ipynb      # Classical model training and evaluation
│   ├── 04_2_training.ipynb    # Transformer and LLM experiments
│   └── 05_ensemble.ipynb      # Voting and stacking ensembles
├── pyproject.toml
└── uv.lock
```

## Reproducibility

The notebooks are intended to be run in numerical order. A fixed random seed is used where supported. The project was developed with Python, PyTorch, Hugging Face Transformers, scikit-learn, pandas, and NumPy.

> The dataset may have access or licensing restrictions and is therefore not included in this repository.
