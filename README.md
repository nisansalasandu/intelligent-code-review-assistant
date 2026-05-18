# <div align="center">

# 

# \# 🔍 Intelligent Code Review Assistant

# 

# \*\*A deep learning–powered tool that detects bugs, vulnerabilities, and code smells in source code using fine-tuned CodeBERT\*\*

# 

# \[!\[Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square\&logo=python\&logoColor=white)](https://python.org)

# \[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square\&logo=pytorch\&logoColor=white)](https://pytorch.org)

# \[!\[HuggingFace](https://img.shields.io/badge/HuggingFace-CodeBERT-FFD21F?style=flat-square\&logo=huggingface\&logoColor=black)](https://huggingface.co/microsoft/codebert-base)

# \[!\[Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square\&logo=streamlit\&logoColor=white)](https://streamlit.io)

# \[!\[License](https://img.shields.io/badge/License-MIT-2E7D32?style=flat-square)](LICENSE)

# \[!\[Internship](https://img.shields.io/badge/Internship-Data%20Science-1B5E20?style=flat-square)]()

# 

# <br/>

# 

# > \*Paste code. Get instant bug detection with token-level explainability.\*

# 

# <br/>

# 

# | Accuracy | F1 Score | Recall | ROC-AUC |

# |:---:|:---:|:---:|:---:|

# | \*\*74.2%\*\* | \*\*0.646\*\* | \*\*0.826\*\* | \*\*0.852\*\* |

# 

# </div>

# 

# \---

# 

# \## 📋 Table of Contents

# 

# \- \[Overview](#-overview)

# \- \[Demo](#-demo)

# \- \[Model Architecture](#-model-architecture)

# \- \[Dataset](#-dataset)

# \- \[Project Structure](#-project-structure)

# \- \[Pipeline](#-pipeline)

# \- \[Installation](#-installation)

# \- \[Usage](#-usage)

# \- \[Notebooks](#-notebooks)

# \- \[Evaluation Results](#-evaluation-results)

# \- \[Explainability](#-explainability)

# \- \[Dashboard Features](#-dashboard-features)

# \- \[Limitations](#-limitations)

# \- \[Future Work](#-future-work)

# 

# \---

# 

# \## 🧠 Overview

# 

# This project builds an end-to-end \*\*automated code review system\*\* that uses deep learning to detect bugs, vulnerabilities, and code smells in source code. It is an individual Data Science internship project completed over 4 weeks.

# 

# The system fine-tunes \*\*\[CodeBERT](https://huggingface.co/microsoft/codebert-base)\*\* — a pre-trained transformer model for programming language understanding — on a combined labelled dataset of 39,656 code samples. The trained model is deployed as a \*\*Streamlit web dashboard\*\* that provides:

# 

# \- ✅ Real-time \*\*binary classification\*\* (clean vs. buggy)

# \- 🎯 \*\*Confidence scores\*\* with adjustable decision threshold

# \- 🔍 \*\*Token-level explainability\*\* via attention weight visualisation

# \- 📊 \*\*Model performance metrics\*\* and training history

# 

# \---

# 

# \## 🎬 Demo

# 

# ```

# streamlit run app/app.py

# ```

# 

# > Screenshot: paste code → instant verdict + highlighted tokens

# 

# <!-- Add your dashboard screenshots below -->

# | Code Review Tab | Metrics Tab |

# |---|---|

# | !\[Code Review](reports/figures/screenshot\_review.png) | !\[Metrics](reports/figures/screenshot\_metrics.png) |

# 

# \---

# 

# \## 🏗️ Model Architecture

# 

# ```

# Input code snippet (string)

# &#x20;       │

# &#x20;       ▼

# CodeBERT Tokenizer  ──  WordPiece, vocab 50,265, max 512 tokens

# &#x20;       │

# &#x20;       ▼

# CodeBERT Encoder  ──  12 transformer layers, 768-dim hidden state

# &#x20; • Bottom 6 layers → FROZEN  (generic syntax features)

# &#x20; • Top 6 layers   → TRAINABLE (task-specific features)

# &#x20;       │

# &#x20;       ▼

# &#x20; \[CLS] token hidden state  (768-dim)

# &#x20;       │

# &#x20;       ▼

# &#x20; Dropout (p = 0.1)

# &#x20;       │

# &#x20;       ▼

# &#x20; Linear (768 → 2)

# &#x20;       │

# &#x20;       ▼

# &#x20; Softmax → P(clean),  P(buggy)

# ```

# 

# | Component | Detail |

# |---|---|

# | Base model | `microsoft/codebert-base` |

# | Encoder layers | 12 (bottom 6 frozen) |

# | Hidden size | 768 |

# | Attention heads | 12 |

# | Max token length | 512 |

# | Dropout | 0.1 |

# | Total parameters | 124,647,170 |

# | Trainable parameters | 43,119,362 (34.6%) |

# 

# \---

# 

# \## 📦 Dataset

# 

# The model was trained on a combined dataset from three sources:

# 

# | Source | Language | Label | Samples |

# |---|---|---|---|

# | \[CodeSearchNet](https://huggingface.co/datasets/code\_search\_net) | Python | Clean (0) | \~14,700 |

# | \[Devign](https://huggingface.co/datasets/google/code\_x\_glue\_cc\_defect\_detection) | C | Mixed (0/1) | \~27,318 |

# | GitHub Bug-Fix PRs | Python | Buggy (1) | \~6 |

# | \*\*Total\*\* | | | \*\*39,656\*\* |

# 

# \*\*Class distribution:\*\* 2.5 : 1 (clean : buggy) → handled with weighted cross-entropy loss

# 

# \*\*Class weights computed from training set:\*\*

# \- Class 0 (clean): `0.6988`

# \- Class 1 (buggy): `1.7573`

# 

# \---

# 

# \## 📁 Project Structure

# 

# ```

# intelligent-code-review-assistant/

# │

# ├── 📓 notebooks/

# │   ├── 01\_data\_collection\_eda.ipynb          # Jupyter (local) — data \& EDA

# │   ├── 02\_preprocessing\_tokenization.ipynb   # Jupyter (local) — tokenisation

# │   ├── 03\_model\_training.ipynb               # Google Colab (T4 GPU) — training

# │   └── 04\_evaluation.ipynb                   # Google Colab (T4 GPU) — evaluation

# │

# ├── 🤖 models/

# │   └── codebert\_classifier/

# │       ├── model.safetensors                 # Trained encoder weights

# │       ├── classifier\_head.pt                # Classification head weights

# │       ├── tokenizer\_config.json             # Tokeniser configuration

# │       └── vocab.json                        # Tokeniser vocabulary

# │

# ├── 🎛️ app/

# │   └── app.py                                # Streamlit dashboard

# │

# ├── 📊 data/

# │   ├── raw/                                  # Downloaded/scraped raw data

# │   │   ├── codesearchnet\_python.csv

# │   │   └── github\_bug\_fixes.csv

# │   └── processed/                            # Tokenised splits + config

# │       ├── train.pt

# │       ├── val.pt

# │       ├── test.pt

# │       ├── class\_weights.npy

# │       └── preprocessing\_config.pkl

# │

# ├── 📈 reports/

# │   ├── training\_results.pkl                  # History, metrics, predictions

# │   ├── classification\_report.csv             # Per-class classification report

# │   └── figures/                              # 20 saved visualisation figures

# │       ├── 01\_label\_distribution.png

# │       ├── 02\_samples\_per\_source.png

# │       ├── 03\_code\_length\_distribution.png

# │       ├── 04\_language\_breakdown.png

# │       ├── 05\_boxplot\_code\_length.png

# │       ├── 06\_correlation\_heatmap.png

# │       ├── 07\_token\_length\_distribution.png

# │       ├── 08\_token\_distribution\_splits.png

# │       ├── 09\_class\_balance\_splits.png

# │       ├── 10\_training\_curves.png

# │       ├── 11\_test\_evaluation.png

# │       ├── 12\_confidence\_distribution.png

# │       ├── 13\_error\_analysis.png

# │       ├── 14\_attention\_true\_positive.png

# │       ├── 15\_attention\_false\_negative.png

# │       ├── 16\_attention\_true\_negative.png

# │       ├── 17\_shap\_bar.png

# │       ├── 18\_shap\_waterfall\_buggy.png

# │       ├── 19\_shap\_waterfall\_clean.png

# │       └── 20\_final\_metrics\_summary.png

# │

# ├── requirements.txt

# ├── .gitignore

# └── README.md

# ```

# 

# \---

# 

# \## 🔄 Pipeline

# 

# ```

# ┌─────────────────────────────────────────────────────────────┐

# │                      4-WEEK PIPELINE                        │

# ├──────────────┬──────────────┬──────────────┬────────────────┤

# │   WEEK 1     │   WEEK 2     │   WEEK 3     │    WEEK 4      │

# │  Notebook 01 │  Notebook 02 │  Notebook 03 │  Notebook 04   │

# │  Jupyter     │  Jupyter     │  Colab (GPU) │  Colab (GPU)   │

# ├──────────────┼──────────────┼──────────────┼────────────────┤

# │ • CodeSearch │ • clean\_code │ • Load .pt   │ • Eval on test │

# │   Net load   │   cleaning   │   datasets   │ • Confusion    │

# │ • Devign     │ • CodeBERT   │ • Fine-tune  │   matrix       │

# │   load       │   tokenise   │   CodeBERT   │ • ROC curve    │

# │ • GitHub PR  │ • 70/15/15   │ • Weighted   │ • Attention    │

# │   scraping   │   split      │   loss       │   viz          │

# │ • EDA plots  │ • Save .pt   │ • Early stop │ • SHAP plots   │

# │              │   tensors    │ • Save model │ • Dashboard    │

# └──────────────┴──────────────┴──────────────┴────────────────┘

# &#x20;        │              │              │               │

# &#x20;        ▼              ▼              ▼               ▼

# &#x20; 39,656 labelled   tokenised     trained model    evaluation

# &#x20;   CSV dataset     .pt tensors   checkpoint       report

# ```

# 

# > \*\*Notebooks 01 \& 02\*\* were run on a \*\*local Jupyter Notebook\*\* for data collection and preprocessing.

# > \*\*Notebooks 03 \& 04\*\* were run on \*\*Google Colab (NVIDIA T4 GPU)\*\* for faster model training and evaluation.

# 

# \---

# 

# \## ⚙️ Installation

# 

# \### Prerequisites

# 

# \- Python 3.10+

# \- Git

# \- 8GB+ RAM (16GB recommended for training)

# \- GPU optional for inference; required for retraining

# 

# \### 1. Clone the repository

# 

# ```bash

# git clone https://github.com/YOUR\_USERNAME/intelligent-code-review-assistant.git

# cd intelligent-code-review-assistant

# ```

# 

# \### 2. Create and activate virtual environment

# 

# ```bash

# \# Windows

# python -m venv venv

# venv\\Scripts\\activate

# 

# \# macOS / Linux

# python -m venv venv

# source venv/bin/activate

# ```

# 

# \### 3. Install dependencies

# 

# ```bash

# pip install --upgrade pip

# pip install -r requirements.txt

# ```

# 

# \### 4. Verify installation

# 

# ```bash

# python -c "import torch, transformers, streamlit; print('✅ All good')"

# ```

# 

# \### requirements.txt

# 

# ```

# torch>=2.0.0

# transformers>=4.40.0

# datasets>=2.0.0

# streamlit>=1.28.0

# scikit-learn>=1.0.0

# shap>=0.45.0

# PyGithub>=2.0.0

# matplotlib>=3.7.0

# seaborn>=0.12.0

# pandas>=2.0.0

# numpy>=1.24.0

# safetensors>=0.4.0

# tqdm>=4.65.0

# huggingface-hub>=0.20.0

# jupyter>=7.0.0

# mlflow>=2.0.0

# ```

# 

# \---

# 

# \## 🚀 Usage

# 

# \### Run the Streamlit dashboard

# 

# ```bash

# streamlit run app/app.py

# ```

# 

# The app opens at `http://localhost:8501` in your browser.

# 

# \### Quick inference (Python API)

# 

# ```python

# import torch

# from transformers import AutoTokenizer

# \# Load tokeniser and model (see app/app.py for full CodeBERTClassifier class)

# 

# tokeniser = AutoTokenizer.from\_pretrained("models/codebert\_classifier")

# 

# code = """

# def calculate\_average(numbers):

# &#x20;   return sum(numbers) / len(numbers)   # bug: crashes if numbers is empty

# """

# 

# \# Tokenise

# enc = tokeniser(code, return\_tensors="pt", max\_length=512,

# &#x20;               padding="max\_length", truncation=True)

# 

# \# Run model

# with torch.no\_grad():

# &#x20;   output = model(\*\*enc)

# &#x20;   probs  = torch.softmax(output\["logits"], dim=-1)

# 

# print(f"P(clean) = {probs\[0]\[0]:.3f}")

# print(f"P(buggy) = {probs\[0]\[1]:.3f}")

# ```

# 

# \---

# 

# \## 📓 Notebooks

# 

# Run the notebooks \*\*in order\*\*. Notebooks 01–02 run locally; 03–04 on Google Colab.

# 

# | # | Notebook | Environment | Input | Output |

# |---|---|---|---|---|

# | 01 | `01\_data\_collection\_eda.ipynb` | Jupyter (local) | HuggingFace Hub, GitHub API | `data/processed/labeled\_dataset\_full.csv` |

# | 02 | `02\_preprocessing\_tokenization.ipynb` | Jupyter (local) | `labeled\_dataset\_full.csv` | `data/processed/train.pt`, `val.pt`, `test.pt` |

# | 03 | `03\_model\_training.ipynb` | Google Colab T4 | `train.pt`, `val.pt` | `models/codebert\_classifier/`, `training\_results.pkl` |

# | 04 | `04\_evaluation.ipynb` | Google Colab T4 | `test.pt`, trained model | Metrics, SHAP plots, attention visualisations |

# 

# \### Running Notebooks 03 \& 04 on Google Colab

# 

# ```python

# \# Step 1: Mount Google Drive in Colab

# from google.colab import drive

# drive.mount('/content/drive')

# 

# \# Step 2: Copy processed data from Drive

# import shutil

# shutil.copytree('/content/drive/MyDrive/code-review/data', 'data')

# 

# \# Step 3: Install dependencies

# !pip install transformers>=4.40.0 datasets torch scikit-learn shap -q

# 

# \# Step 4: Run all cells

# ```

# 

# \### GitHub Token Setup (Notebook 01)

# 

# ```bash

# \# Set as environment variable (never hardcode in notebook)

# export GITHUB\_TOKEN="your\_token\_here"          # macOS/Linux

# set GITHUB\_TOKEN=your\_token\_here               # Windows

# ```

# 

# Get a free token at \[github.com/settings/tokens](https://github.com/settings/tokens) → Generate new token (classic) → scope: `public\_repo` only.

# 

# \---

# 

# \## 📊 Evaluation Results

# 

# \### Test Set Performance (5,949 samples)

# 

# | Metric | Score | Notes |

# |---|---|---|

# | \*\*Accuracy\*\* | 74.23% | Majority-class baseline: 71.6% |

# | \*\*Precision\*\* | 0.5303 | 53% of flagged code is truly buggy |

# | \*\*Recall\*\* | 0.8263 | Catches 82.6% of all real bugs ✅ |

# | \*\*F1 Score\*\* | 0.6460 | Harmonic mean |

# | \*\*ROC-AUC\*\* | 0.8519 | Strong discrimination ability |

# 

# \### Training History (6 epochs)

# 

# | Epoch | Train Loss | Val Loss | Val F1 | Val ROC-AUC |

# |:---:|:---:|:---:|:---:|:---:|

# | 1 | 0.4959 | 0.4448 | 0.6345 | 0.8209 |

# | 2 | 0.4522 | 0.4921 | 0.6222 | 0.8410 |

# | \*\*3 ★\*\* | \*\*0.4227\*\* | \*\*0.4586\*\* | \*\*0.6572\*\* | \*\*0.8552\*\* |

# | 4 | 0.3838 | 0.5310 | 0.6336 | 0.8593 |

# | 5 | 0.3445 | 0.5585 | 0.6328 | 0.8624 |

# | 6 | 0.3082 | 0.7416 | 0.6095 | 0.8623 |

# 

# > ★ Best checkpoint saved at \*\*epoch 3\*\* (early stopping on validation F1, patience=2)

# 

# \### Training Configuration

# 

# | Hyperparameter | Value |

# |---|---|

# | Optimizer | AdamW |

# | Learning rate | 2e-5 |

# | Batch size | 16 |

# | Gradient accumulation | 4 steps |

# | Warmup ratio | 0.1 |

# | Scheduler | Linear warmup + decay |

# | Loss function | Weighted CrossEntropyLoss |

# | Frozen encoder layers | Bottom 6 of 12 |

# | Early stopping patience | 2 epochs |

# 

# \---

# 

# \## 🔍 Explainability

# 

# The dashboard provides two types of explainability:

# 

# \### 1. Attention-Based Token Highlighting

# 

# For each prediction, attention weights from the last CodeBERT transformer layer are extracted, averaged across 12 heads, and projected from the `\[CLS]` token to every input token. High-attention tokens are highlighted:

# 

# \- 🔴 \*\*Red intensity\*\* → token pushed prediction toward \*\*buggy\*\*

# \- 🟢 \*\*Green intensity\*\* → token pushed prediction toward \*\*clean\*\*

# 

# ```

# Method:

# &#x20; attention = last\_layer\_attention\[0]        # shape: (heads, seq, seq)

# &#x20; cls\_attn  = attention.mean(dim=0)\[0, :]   # mean over heads, CLS row

# &#x20; scores    = normalise(cls\_attn, 0, 1)     # scale to \[0, 1]

# ```

# 

# \### 2. SHAP Analysis (Notebook 04)

# 

# SHAP (SHapley Additive exPlanations) values were computed on a sample of test predictions to provide theoretically grounded feature attribution. SHAP satisfies consistency and efficiency axioms that attention weights do not guarantee.

# 

# \---

# 

# \## 🎛️ Dashboard Features

# 

# \### Code Review Tab

# \- 📥 \*\*Code input\*\* — paste any Python or C code snippet

# \- 🚨 \*\*Verdict banner\*\* — red (buggy) / green (clean) with confidence %

# \- 📊 \*\*Confidence bars\*\* — P(clean) and P(buggy) visual bars

# \- 🎨 \*\*Token highlights\*\* — colour-coded attention visualisation

# \- 📋 \*\*Top-15 tokens table\*\* — most-attended tokens with scores

# \- 🔢 \*\*Raw JSON output\*\* — full model output for debugging

# \- 💡 \*\*5 example snippets\*\* — pre-loaded buggy and clean examples

# 

# \### Sidebar Controls

# \- 🎚️ \*\*Decision threshold\*\* — adjustable from 0.10 to 0.90 (default 0.50)

# \- 🔢 \*\*Attention layer\*\* — selectable transformer layer 0–11 for explainability

# 

# \### Model Metrics Tab

# \- 📈 Training curves (loss, F1, precision/recall, ROC-AUC over epochs)

# \- 🟦 Confusion matrix on test set

# \- 📉 Confidence distribution by true label

# \- 📋 Full classification report

# 

# \---

# 

# \## ⚠️ Limitations

# 

# \- \*\*Language bias\*\* — buggy samples are predominantly C (Devign), clean samples are Python (CodeSearchNet). May underperform on Python-specific bugs.

# \- \*\*512-token limit\*\* — functions longer than \~200 lines are truncated; only the first 512 tokens are analysed.

# \- \*\*Binary classification only\*\* — detects presence of bugs but does not categorise bug type or suggest fixes.

# \- \*\*Attention ≠ causation\*\* — attention highlights are approximate explanations, not guaranteed causal attributions (Jain \& Wallace, 2019).

# \- \*\*GitHub scraper yield\*\* — PR label inconsistencies across repositories limited Python bug-fix collection to very few samples.

# 

# \---

# 

# \## 🔭 Future Work

# 

# \- \[ ] Fine-tune \*\*GraphCodeBERT\*\* with data flow graph information for improved structural understanding

# \- \[ ] Train \*\*CodeT5+\*\* seq2seq model to generate fix suggestions alongside detection

# \- \[ ] Multi-class bug categorisation (null dereference, off-by-one, buffer overflow, logic error)

# \- \[ ] \*\*VS Code extension\*\* for in-editor real-time code review

# \- \[ ] \*\*GitHub Actions\*\* CI/CD integration — auto-review on pull requests

# \- \[ ] Replace attention with \*\*Integrated Gradients\*\* for theoretically grounded explainability

# \- \[ ] Expand training data with \*\*BigVul\*\* / \*\*ReVeal\*\* datasets for broader vulnerability coverage

# \- \[ ] Add \*\*Docker\*\* containerisation for consistent deployment

# 

# 

# 

# \---

# 

# \## 📄 License

# 

# This project is licensed under the MIT License — see the \[LICENSE](LICENSE) file for details.

# 

# \---

# 

# <div align="center">

# 

# \*\*Built as an individual Data Science internship project · 4-week timeline\*\*

# 

# \*\[Nisansala Ruwan Pathirana]\*

# 

# </div>

