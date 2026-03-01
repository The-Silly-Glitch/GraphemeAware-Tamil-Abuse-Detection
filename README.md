# GraphemeAware-Tamil-Abuse-Detection

Transformer-based detection of abusive Tamil text targeting women using grapheme-aware normalization.

This repository contains the implementation of our system submitted to **DravidianLangTech@ACL 2026**, where our approach secured **3rd rank** in the shared task.

---

## 📌 Overview

Online social media platforms often contain abusive content targeting women. This project addresses the task of detecting abusive Tamil comments using transformer-based models combined with language-aware preprocessing.

We model the task as a **binary text classification problem**:

- **Abusive** (Label: 1)
- **Non-Abusive** (Label: 0)

In addition to standard preprocessing, we introduce a **grapheme-aware normalization technique** designed to preserve Tamil character integrity at the Unicode level before tokenization. This approach helps maintain the semantic meaning of Tamil text while improving model performance.

---

## 🧠 Models Implemented

The repository contains implementations for three state-of-the-art multilingual and language-specific transformers:

- **IndicBERT** - Large-scale BERT model trained on 12 Indian languages
- **MuRIL** (Multilingual and Code-Mixed Language Models) - Designed for Indian language processing
- **TamilBERT** - BERT model specifically pretrained on Tamil corpus

Each model is trained and evaluated under two configurations:

1. **Baseline Preprocessing** - Standard text cleaning without grapheme-level normalization
2. **Grapheme-Aware Preprocessing** - Language-specific normalization preserving Tamil character integrity

**Best Performing Model:** TamilBERT with grapheme-aware normalization (F1-score: 0.8476)

---

## 📂 Repository Structure

```
GraphemeAware-Tamil-Abuse-Detection/
│
├── abusive_text_tamilbert.ipynb          # TamilBERT model training and evaluation
├── abusive_text_indicbert.ipynb          # IndicBERT model training and evaluation
├── abusive_text_murel.ipynb              # MuRIL model training and evaluation
├── Test.ipynb                            # Test predictions generation
└── README.md                             # This file
```

### Notebook Descriptions

#### **abusive_text_tamilbert.ipynb**
Fine-tunes TamilBERT for abusive text classification with grapheme-aware preprocessing. This notebook includes:
- Data loading and exploration
- Grapheme-level text normalization
- Model training with AdamW optimizer
- Validation and evaluation metrics
- Visualization of results

#### **abusive_text_indicbert.ipynb**
Fine-tunes IndicBERT for the same task, providing a comparison with a larger multilingual model.

#### **abusive_text_murel.ipynb**
Fine-tunes MuRIL (code-mixed capable) to evaluate its effectiveness on Tamil abusive text detection.

#### **Test.ipynb**
Generates predictions for the official test dataset using the best-performing model:
- Loads the trained model and tokenizer
- Applies the same grapheme-aware preprocessing
- Outputs predictions in the required format

---

## ⚙️ Training Configuration

All models were trained using consistent hyperparameters to ensure fair comparison:

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 5 |
| Optimizer | AdamW |
| Loss Function | Cross-Entropy Loss |
| Evaluation Metric | F1-score |
| Train-Validation Split | 90%-10% (stratified) |

**Notes:**
- No ensemble methods were used
- No data augmentation was applied
- Stratified split ensures balanced label distribution in train/validation sets

---

## 🔬 Text Preprocessing Pipeline

### Grapheme-Aware Normalization

The key innovation of this work is a **grapheme-level normalization** approach specifically designed for Tamil text:

```python
def grapheme_preprocess(text):
    # 1. HTML Decoding
    text = html.unescape(text)
    
    # 2. Unicode Normalization (NFC)
    text = unicodedata.normalize("NFC", text)
    
    # 3. URL Removal
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # 4. Grapheme Cluster Extraction
    clusters = list(grapheme.graphemes(text))
    
    # 5. Grapheme Rejoin (preserves character integrity)
    text = "".join(clusters)
    
    # 6. Whitespace Normalization
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
```

### Preprocessing Steps

1. **HTML Decoding** - Converts HTML entities to their actual characters
2. **Unicode Normalization** - Applies NFC normalization to ensure consistent character representation
3. **URL Removal** - Eliminates hyperlinks commonly found in social media text
4. **Grapheme Extraction** - Breaks text into grapheme clusters (minimal units with semantic meaning in Tamil)
5. **Whitespace Cleanup** - Removes extra spaces and line breaks

---

## 📊 Results

### Validation Performance (F1-Score)

| Model | Baseline | Grapheme-Aware | Improvement |
|-------|----------|---|---|
| IndicBERT | 0.8201 | 0.8156 | -0.45% |
| MuRIL | 0.7958 | 0.7956 | -0.03% |
| **TamilBERT** | **0.7941** | **0.8476** | **+6.75%** ✓ |

### Key Findings

- **TamilBERT with grapheme-aware preprocessing achieved the best overall performance (F1: 0.8476)**
- Grapheme-level normalization showed significant improvement (+6.75%) specifically for language-specific models (TamilBERT)
- Larger multilingual models (IndicBERT) were less sensitive to grapheme-level preprocessing
- MuRIL, while showing reasonable performance, didn't benefit from the grapheme-aware approach

---

## 🔬 Key Contributions

1. **Grapheme-Level Normalization for Tamil NLP** - Demonstrates that language-specific preprocessing at the grapheme level can meaningfully improve transformer-based Tamil text classification.

2. **Comprehensive Model Comparison** - Systematic evaluation of multilingual vs. language-specific models for Tamil abusive text detection.

3. **Practical Implementation** - Ready-to-use notebooks for fine-tuning multiple models with consistent experimental setup.

Results validate that language-aware preprocessing combined with BERT models specifically trained on Tamil can achieve competitive performance on abusive content detection.

---

## 📝 Requirements

### System Requirements
- Python 3.8 or higher
- GPU recommended (for faster training)
- Google Colab environment (notebooks are designed for Colab)

### Python Dependencies

```
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
accelerate>=0.10.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
grapheme>=0.6.0
```

### Installation

```bash
pip install torch transformers datasets accelerate scikit-learn pandas numpy grapheme
```

### Model Checkpoints

The pretrained model checkpoints for the following are referenced in the notebooks:
- `bert-base-multilingual-uncased` (IndicBERT)
- `deepset/muril-base-cased` (MuRIL)
- `sarathi/tamil-bert` (TamilBERT)

These are automatically downloaded from HuggingFace Model Hub during execution.

---

## 🚀 Usage

### Training a Model

1. **Prepare your data** in CSV format with columns: `Text` and `Class`
   - Example: `trainV2.csv`

2. **Open the appropriate notebook** (e.g., `abusive_text_tamilbert.ipynb`)

3. **Upload your training data** using the file upload cell

4. **Run all cells** to:
   - Install dependencies
   - Load and preprocess data
   - Fine-tune the model
   - Evaluate on validation set

5. **Save the trained model**
   ```python
   model.save_pretrained("/path/to/save/model")
   tokenizer.save_pretrained("/path/to/save/model")
   ```

### Making Predictions

1. **Open `Test.ipynb`**

2. **Update the model path** to point to your saved model

3. **Upload your test data** (CSV file with `Text` column)

4. **Run all cells** to generate predictions

5. **Output**: DataFrame with original text and predicted labels
   - 0 = Non-Abusive
   - 1 = Abusive

---

## 📄 Paper & Citation

**Paper Title:** HNK@DravidianLangTech 2026: Abusive Tamil Text Targeting Women on Social Media

**Task:** DravidianLangTech 2026 - Abuse Detection for Tamil

**Performance:** 3rd Rank in Shared Task

**GitHub Repository:**  
https://github.com/The-Silly-Glitch/GraphemeAware-Tamil-Abuse-Detection

### BibTeX

```bibtex
@inproceedings{vigneshwar2026graphemeaware,
  title={GraphemeAware-Tamil-Abuse-Detection: Transformer-based Detection of Abusive Tamil Text Targeting Women using Grapheme-Aware Normalization},
  author={Vigneshwar R, Hanish and Alaguraj, Nahul and M, Karthikeyan},
  booktitle={Proceedings of ACL 2026: DravidianLangTech Workshop},
  year={2026}
}
```

---

## 📊 Evaluation Metrics

The models are evaluated using:

- **F1-Score** (Primary metric) - Harmonic mean of precision and recall
- **Precision** - Proportion of positive predictions that are correct
- **Recall** - Proportion of actual positives that are correctly identified
- **Accuracy** - Overall correctness of predictions
- **Confusion Matrix** - Breakdown of true positives, false positives, etc.

Performance visualizations (confusion matrices, classification reports) are generated at the end of each training notebook.

---

## 🔍 Experimental Design

### Data Handling
- **Stratified Train-Test Split** ensures balanced label distribution
- **Random Seed = 42** for reproducibility
- **No data leakage** - validation set is completely separate from training

### Model Selection Rationale
- **IndicBERT**: Large multilingual model for broad language coverage
- **MuRIL**: Specialized for code-mixed and Indian language content
- **TamilBERT**: Language-specific model for maximum Tamil context understanding

### Ablation Study
The comparison between baseline and grapheme-aware preprocessing serves as an ablation study, isolating the impact of language-specific normalization.

---

## 🐛 Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError` for grapheme library
- **Solution:** Run the pip install cell: `!pip install -q grapheme`

**Issue:** CUDA out of memory
- **Solution:** Reduce batch size from 16 to 8 or 4 in the training configuration

**Issue:** Model download fails
- **Solution:** Ensure internet connection and sufficient disk space. Models are ~400-500MB each

**Issue:** CSV file not found
- **Solution:** Ensure the CSV file is uploaded before reading. Use the Google Colab file upload cell

---

## 💡 Features & Implementation Details

### Advanced Features
- **Automatic Mixed Precision (AMP)** for faster training
- **Gradient Accumulation** support for large batch training
- **Learning Rate Scheduling** with warmup
- **Early Stopping** to prevent overfitting
- **Model Checkpointing** to save best validation checkpoint

### Code Quality
- Clean, well-documented code
- Consistent variable naming conventions
- Comments explaining complex logic
- Reusable preprocessing functions

---

## 👨‍💻 Authors

- **Hanish Vigneshwar R** - Lead Developer  
- **Nahul Alaguraj** - Co-author  
- **Karthikeyan M** - Co-author  

**Affiliation:** Vellore Institute of Technology, Chennai Campus

---

## 📜 License

This project is provided as-is for research and educational purposes. Please cite appropriately if used in your work.

---

## 🤝 Contributing

While this is a competition submission, we welcome questions and discussions about the approach. Feel free to open issues or contact the authors for clarifications.

---

## 🙏 Acknowledgments

- DravidianLangTech 2026 track organizers
- HuggingFace Transformers library for model implementations
- Grapheme library for grapheme-level text processing
- The research community working on Tamil NLP

---

## 📮 Contact & Support

For questions about the implementation or methodology, please refer to the GitHub repository or contact the authors directly.

**Last Updated:** March 2026
