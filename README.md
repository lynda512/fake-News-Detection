
# Fake News Detection Using Large Language Models
> This repository contains the implementation of a group project developed for the **Data Science Lab** course, as part of the academic curriculum (The original repository is in the FIMGIT of my university)  and received a final grade of 1.3 .

This repository contains the code and experiments for the project **"Fake News Detection Using Large Language Models"**, which investigates the performance of traditional machine learning models, large language models (LLMs), and hybrid approaches on the **LIAR** dataset for automatic fake news detection.
This project is a part

## Project Overview

The project explores whether LLMs outperform traditional ML algorithms in fake news detection and whether hybrid models that combine LLMs with smaller or traditional models can improve performance and efficiency. The study focuses on multi-class and binary classification of political statements into truthfulness categories (e.g., true, mostly-true, false, pants-fire).  

## Dataset

The project uses the **LIAR** benchmark dataset by Wang, which contains 12.8K short political statements collected from PolitiFact. Each record includes:
- Statement text  
- Truthfulness label (true, mostly-true, half-true, barely-true, false, pants-fire)  
- Speaker metadata (name, job title, party, state)  
- Credibility history counts (e.g., false count, pants-on-fire count)

The dataset is split into:
- Train: 10,296 samples (80.0%)  
- Validation: 1,284 samples (10.0%)  
- Test: 1,267 samples (9.8%)

## Methodology

### Preprocessing and Feature Engineering

Key preprocessing and feature engineering steps include:

- Text cleaning of the *statement* field (lowercasing, URL and symbol removal, stop word removal, lemmatization).  
- Handling missing values and dropping irrelevant columns (e.g., IDs); rare party labels grouped into **Others**.  
- Label encoding of categorical truth labels for supervised learning.  
- Sentiment analysis using TextBlob to derive sentiment scores for each statement.  
- Creation of a **False Ratio** feature using the speaker’s credibility history counts (false and pants-on-fire vs total).  
- TF–IDF vectorization for textual features.

### Models

The following models are implemented and evaluated:

- **Traditional ML models**
  - Logistic Regression (multi-class)  
  - Random Forest (multi-class)

- **Large Language Models (LLMs) – LLaMA 3.2 1B**
  - Causal language modeling with **AutoModelForCausalLM** and supervised fine-tuning (SFTTrainer, prompt-based).  
  - Sequence classification with **AutoModelForSequenceClassification** (multi-class).  
  - Sequence classification with **AutoModelForSequenceClassification** (binary: true vs false).

Hyperparameters include batch size 16, learning rate \(5 \times 10^{-5}\), weight decay 0.3, warmup steps, and early stopping for the sequence-classification setup.

## Results

### Traditional Models

- **Logistic Regression (multi-class)**  
  - Test accuracy: **28%**.

- **Random Forest (multi-class)**  
  - Test accuracy: **34%** (highest among all evaluated models).  
  - Weighted F1-score: **0.31**.

Random Forest shows better overall performance and robustness than logistic regression, despite class imbalance in the LIAR dataset.

### LLaMA 3.2 1B – Multi-class

- **CausalLM + SFT (prompt-based)**  
  - Test accuracy: **25%**.  
  - High precision on some classes but very low recall for minority labels (e.g., pants-fire).  

- **AutoModelForSequenceClassification (multi-class)**  
  - Test accuracy: **24%**.  
  - More uniform precision/recall across classes but still struggles with underrepresented labels.

### LLaMA 3.2 1B – Binary

- **AutoModelForSequenceClassification (binary true/false)**  
  - Test accuracy: **60%**, around 10% above random baseline for two classes.

Overall, conventional models (especially Random Forest) outperform the LLaMA-based approaches on the multi-class LIAR setup in this project.

## Key Findings

- Conventional models remain competitive and can outperform lightweight LLM setups on structured, imbalanced datasets like LIAR.  
- Class imbalance (e.g., underrepresented pants-fire class) strongly degrades performance, particularly recall for minority labels.  
- Reformulating the task as binary classification improves accuracy for LLaMA 3.2 1B but still does not surpass the Random Forest in the multi-class setting in terms of relative performance to chance.

## Future Work

Potential future directions include:

- Designing **hybrid frameworks** where:
  - Traditional models handle the bulk of straightforward cases.  
  - LLMs act as second-stage experts for ambiguous or hard examples.  

- Addressing class imbalance via:
  - Data augmentation (e.g., SMOTE or GAN-based synthesis).  
  - Class-weighted losses and custom sampling strategies.

- Improving LLM performance with domain-specific fine-tuning on political fact-checking corpora and integrating external retrieval.  

- Enhancing interpretability using attention visualization, SHAP/LIME, and human-in-the-loop workflows for sensitive or ambiguous predictions.


## Requirements

Main dependencies used in the project:

- Python 3.x  
- pandas, numpy, scikit-learn  
- TextBlob  
- matplotlib / seaborn (for plots)  
- Hugging Face `transformers`, `datasets`, and `trl` (for SFTTrainer)  
- torch

## How to Run

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Prepare the LIAR dataset (TSV → CSV) and place splits under `data/` as expected by the scripts.

3. Run traditional models:

   ```
   python src/train_traditional.py
   ```

4. Run LLaMA 3.2 1B experiments (adjust model path/checkpoint as needed):

   ```
   python src/train_llama_causallm.py
   python src/train_llama_seqcls.py
   ```

5. Inspect metrics and plots in the `results/` and `notebooks/` directories.

## Citation

If you use this work, please cite the original LIAR dataset paper and any referenced works as appropriate, including Wang (2017) and related literature on fake news detection and LLM-based approaches.
```

