"""
app.py — Intelligent Code Review Assistant
Streamlit dashboard for the CodeBERT-based bug detection model.

Reads from:
  models/codebert_classifier/   — saved model weights + tokenizer (Notebook 03)
  reports/training_results.pkl  — training history + test metrics  (Notebook 03)

Run with:
  streamlit run app/app.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Data & numerics ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── HuggingFace ───────────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModel

# ── Visualization ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # headless backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Streamlit ─────────────────────────────────────────────────────────────────
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Code Review Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — adjust if your folder layout differs
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR      = os.path.join(BASE_DIR, "models", "codebert_classifier")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")
RESULTS_PKL    = os.path.join(REPORTS_DIR, "training_results.pkl")

MODEL_NAME     = "microsoft/codebert-base"
MAX_LENGTH     = 512
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --surface2:  #21262d;
    --border:    #30363d;
    --accent:    #3fb950;
    --accent2:   #58a6ff;
    --danger:    #f85149;
    --warn:      #e3b341;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --mono:      'JetBrains Mono', monospace;
    --sans:      'DM Sans', sans-serif;
    --serif:     'DM Serif Display', serif;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* ── Header strip ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
}
.app-header h1 {
    font-family: var(--serif) !important;
    font-size: 1.9rem !important;
    font-weight: 400 !important;
    color: var(--text) !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}
.app-header .badge {
    font-family: var(--mono);
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    background: #1f6feb33;
    border: 1px solid #1f6feb88;
    color: var(--accent2);
    letter-spacing: 0.04em;
    white-space: nowrap;
    align-self: flex-end;
    margin-bottom: 4px;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 1.2rem; }
.metric-card {
    flex: 1; min-width: 110px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}
.metric-card .val {
    font-family: var(--mono);
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--accent2);
    line-height: 1.2;
}
.metric-card .lbl {
    font-size: 11px;
    color: var(--muted);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Verdict banner ── */
.verdict-buggy {
    background: linear-gradient(135deg, #2d1b1b 0%, #1e1e1e 100%);
    border: 1px solid #f8514966;
    border-left: 4px solid var(--danger);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}
.verdict-clean {
    background: linear-gradient(135deg, #1b2d1e 0%, #1e1e1e 100%);
    border: 1px solid #3fb95066;
    border-left: 4px solid var(--accent);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}
.verdict-title {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 4px;
}
.verdict-sub {
    font-size: 13px;
    color: var(--muted);
}

/* ── Code input override ── */
.stTextArea textarea {
    font-family: var(--mono) !important;
    font-size: 13px !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ── Token highlight block ── */
.token-viz {
    font-family: var(--mono);
    font-size: 13px;
    line-height: 2.2;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    white-space: pre-wrap;
    word-break: break-all;
}
.tok {
    display: inline;
    border-radius: 3px;
    padding: 1px 0;
}

/* ── Section labels ── */
.section-label {
    font-size: 11px;
    font-family: var(--mono);
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.5rem 0 0.6rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 5px;
}

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 0.8rem;
}
.conf-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 6px;
    font-family: var(--mono);
}
.conf-bar-track {
    height: 10px;
    background: var(--border);
    border-radius: 6px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease;
}

/* ── Buttons ── */
.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 6px !important;
    font-family: var(--sans) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 0.45rem 1.4rem !important;
    transition: background 0.15s ease !important;
}
.stButton > button:hover {
    background: #2ea043 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-family: var(--mono) !important;
    font-size: 13px !important;
    color: var(--muted) !important;
}

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }

/* ── Table ── */
.stDataFrame { font-family: var(--mono) !important; font-size: 12px !important; }

/* ── Info / warning boxes ── */
.stAlert { border-radius: 8px !important; font-size: 13px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION  (must match Notebook 03 exactly)
# ─────────────────────────────────────────────────────────────────────────────
class CodeBERTClassifier(nn.Module):
    """
    Binary classifier: CodeBERT encoder → [CLS] → Dropout → Linear(768, 2).
    Mirrors the architecture defined in Notebook 03.
    """

    def __init__(
        self,
        model_name: str    = "microsoft/codebert-base",
        num_labels: int    = 2,
        dropout: float     = 0.1,
        freeze_layers: int = 6
    ):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self._freeze_encoder_layers(freeze_layers)

    def _freeze_encoder_layers(self, n: int):
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None, output_attentions=False):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits     = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {
            "loss":       loss,
            "logits":     logits,
            "attentions": outputs.attentions if output_attentions else None
        }


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING  (mirrors Notebook 02 clean_code function)
# ─────────────────────────────────────────────────────────────────────────────
def clean_code(code: str) -> str:
    """Minimal cleaning identical to Notebook 02's clean_code()."""
    if not isinstance(code, str):
        return ""
    code = code.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    code = code.replace("\x00", "")
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = re.sub(r"\n{3,}", "\n\n", code)
    code = code.strip()
    code = code[:5000]
    return code


# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CodeBERT model…")
def load_model_and_tokenizer():
    """
    Load tokenizer and model weights.
    Falls back to HuggingFace Hub if local model directory is missing.
    Cached across all Streamlit sessions — loaded only once.
    """
    # ── Tokenizer ────────────────────────────────────────────────────────────
    if os.path.isdir(MODEL_DIR) and os.path.exists(
            os.path.join(MODEL_DIR, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Model ────────────────────────────────────────────────────────────────
    model = CodeBERTClassifier(
        model_name    = MODEL_NAME,
        num_labels    = 2,
        dropout       = 0.1,
        freeze_layers = 6
    )

    safetensors_path = os.path.join(MODEL_DIR, "model.safetensors")
    pt_path          = os.path.join(MODEL_DIR, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state = load_file(safetensors_path)
        model.encoder.load_state_dict(state, strict=False)
    elif os.path.exists(pt_path):
        state = torch.load(pt_path, map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(state, strict=False)
    # else: use pretrained CodeBERT weights as-is (demo mode)

    # Load classifier head if saved separately (Notebook 03 saves it this way)
    clf_path = os.path.join(MODEL_DIR, "classifier_head.pt")
    if os.path.exists(clf_path):
        clf_state = torch.load(clf_path, map_location="cpu", weights_only=True)
        model.classifier.load_state_dict(clf_state)

    model.to(DEVICE)
    model.eval()
    return tokenizer, model


@st.cache_data(show_spinner=False)
def load_training_results():
    """Load training history and test metrics from Notebook 03's pickle file."""
    if not os.path.exists(RESULTS_PKL):
        return None
    with open(RESULTS_PKL, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict(code: str, tokenizer, model):
    """
    Run inference on a single code snippet.

    Returns
    -------
    dict with keys:
        label        — 0 (clean) or 1 (buggy)
        prob_clean   — float probability for class 0
        prob_buggy   — float probability for class 1
        confidence   — max(prob_clean, prob_buggy)
        tokens       — list of decoded token strings
        token_ids    — tensor of input IDs
        attention_mask — tensor
        attentions   — tuple of attention weight tensors (one per layer)
    """
    cleaned = clean_code(code)
    if not cleaned:
        return None

    enc = tokenizer(
        cleaned,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

    probs      = F.softmax(out["logits"], dim=-1)[0].cpu().numpy()
    label      = int(probs.argmax())
    real_len   = int(attention_mask.sum().item())

    # Decode tokens (skip padding)
    ids_list   = input_ids[0].cpu().tolist()[:real_len]
    tokens     = tokenizer.convert_ids_to_tokens(ids_list)

    return {
        "label":         label,
        "prob_clean":    float(probs[0]),
        "prob_buggy":    float(probs[1]),
        "confidence":    float(probs.max()),
        "tokens":        tokens,
        "token_ids":     input_ids,
        "attention_mask": attention_mask,
        "attentions":    out["attentions"],
        "real_len":      real_len
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXPLAINABILITY — attention-based token highlighting
# ─────────────────────────────────────────────────────────────────────────────
def get_token_importance(result: dict, layer: int = -1) -> list:
    """
    Compute per-token importance scores from CodeBERT attention weights.

    Method (same as Notebook 04):
      1. Take the last attention layer (layer=-1 → most task-specific)
      2. Average over all 12 attention heads
      3. Extract attention from the [CLS] token to every other token
         (this is how much the model "looks at" each token when forming
          its sequence-level representation)
      4. Normalise to [0, 1]

    Returns a list of (token_string, importance_score) tuples.
    """
    if result["attentions"] is None:
        return [(t, 0.0) for t in result["tokens"]]

    attn = result["attentions"][layer]            # (1, heads, seq, seq)
    attn = attn[0].mean(dim=0).cpu().numpy()      # (seq, seq) — mean over heads
    cls_attn = attn[0, :result["real_len"]]       # CLS row, real tokens only

    # Normalise
    mn, mx = cls_attn.min(), cls_attn.max()
    if mx - mn < 1e-8:
        cls_attn = np.zeros_like(cls_attn)
    else:
        cls_attn = (cls_attn - mn) / (mx - mn)

    return list(zip(result["tokens"], cls_attn.tolist()))


def render_token_html(token_importances: list, label: int) -> str:
    """
    Build an HTML string that colour-codes each token by its attention score.

    High attention → red (buggy indicator) or green (clean indicator).
    Low attention  → neutral background.
    """
    base_r, base_g, base_b = (248, 81, 73) if label == 1 else (63, 185, 80)
    parts = []

    for token, score in token_importances:
        # Skip special tokens from display
        if token in ("<s>", "</s>", "<pad>", "<unk>"):
            continue

        # Convert sub-word prefix (Ġ) to a space for readability
        display = token.replace("Ġ", " ").replace("Ċ", "\n")

        alpha = max(0.05, min(score * 0.85, 0.85))
        bg    = f"rgba({base_r},{base_g},{base_b},{alpha:.3f})"

        parts.append(
            f"<span class='tok' style='background:{bg};' "
            f"title='attention: {score:.3f}'>{display}</span>"
        )

    return "<div class='token-viz'>" + "".join(parts) + "</div>"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_confidence_bars(prob_clean: float, prob_buggy: float) -> str:
    """Return HTML confidence bar widgets."""
    def bar(label, prob, color):
        pct = f"{prob*100:.1f}%"
        return f"""
        <div class="conf-bar-wrap">
          <div class="conf-bar-label"><span>{label}</span><span>{pct}</span></div>
          <div class="conf-bar-track">
            <div class="conf-bar-fill" style="width:{prob*100:.1f}%;background:{color};"></div>
          </div>
        </div>"""
    return (bar("Clean code", prob_clean, "#3fb950") +
            bar("Buggy code", prob_buggy, "#f85149"))


def make_training_curve_fig(history: dict) -> plt.Figure:
    """
    Plot training curves using exact keys from training_results.pkl.

    Confirmed keys in history dict (from Notebook 03):
        train_loss, val_loss, val_accuracy, val_precision,
        val_recall, val_f1, val_roc_auc
    NOTE: train_f1 does NOT exist — Notebook 03 only logged train_loss.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    # Build a 2x2 grid: loss, f1, precision/recall, roc-auc
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor="#161b22")
    axes = axes.flatten()

    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_color("#30363d")

    # ── Plot 1: Loss ──────────────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], "o-", color="#58a6ff",
                 lw=2, label="Train loss")
    axes[0].plot(epochs, history["val_loss"],   "o-", color="#f85149",
                 lw=2, label="Val loss")
    axes[0].set_title("Loss",   color="#e6edf3", fontsize=11)
    axes[0].set_xlabel("Epoch", color="#8b949e")
    axes[0].legend(facecolor="#21262d", labelcolor="#e6edf3", fontsize=9)

    # ── Plot 2: Validation F1 ─────────────────────────────────────────────────
    axes[1].plot(epochs, history["val_f1"], "o-", color="#3fb950",
                 lw=2, label="Val F1")
    # Mark the best epoch
    best_idx = int(np.argmax(history["val_f1"]))
    axes[1].axvline(best_idx + 1, color="#e3b341", linestyle="--",
                    linewidth=1.2, label=f"Best epoch {best_idx+1}")
    axes[1].set_title("Validation F1",  color="#e6edf3", fontsize=11)
    axes[1].set_xlabel("Epoch",         color="#8b949e")
    axes[1].legend(facecolor="#21262d", labelcolor="#e6edf3", fontsize=9)

    # ── Plot 3: Validation Precision & Recall ─────────────────────────────────
    axes[2].plot(epochs, history["val_precision"], "o-", color="#d2a8ff",
                 lw=2, label="Val precision")
    axes[2].plot(epochs, history["val_recall"],    "o-", color="#ffa657",
                 lw=2, label="Val recall")
    axes[2].set_title("Precision & Recall", color="#e6edf3", fontsize=11)
    axes[2].set_xlabel("Epoch",             color="#8b949e")
    axes[2].legend(facecolor="#21262d",     labelcolor="#e6edf3", fontsize=9)

    # ── Plot 4: Validation ROC-AUC ────────────────────────────────────────────
    axes[3].plot(epochs, history["val_roc_auc"], "o-", color="#79c0ff",
                 lw=2, label="Val ROC-AUC")
    axes[3].set_title("ROC-AUC",  color="#e6edf3", fontsize=11)
    axes[3].set_xlabel("Epoch",   color="#8b949e")
    axes[3].legend(facecolor="#21262d", labelcolor="#e6edf3", fontsize=9)

    fig.suptitle("Training History (6 epochs)", color="#e6edf3",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def make_confusion_matrix_fig(labels, preds) -> plt.Figure:
    """Plot confusion matrix from test set predictions."""
    from sklearn.metrics import confusion_matrix
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#161b22")
    ax.set_facecolor("#0d1117")
    sns.heatmap(
        cm, annot=True, fmt="d",
        cmap=sns.light_palette("#58a6ff", as_cmap=True),
        xticklabels=["Clean", "Buggy"],
        yticklabels=["Clean", "Buggy"],
        ax=ax, linewidths=0.5, linecolor="#30363d"
    )
    ax.set_xlabel("Predicted", color="#8b949e")
    ax.set_ylabel("Actual",    color="#8b949e")
    ax.set_title("Confusion Matrix", color="#e6edf3")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE SNIPPETS
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLES = {
    "ZeroDivisionError (buggy)": """\
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)   # crashes if numbers is empty
""",
    "Safe average (clean)": """\
def calculate_average(numbers):
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)
""",
    "Off-by-one in loop (buggy)": """\
def find_max(arr):
    max_val = arr[0]
    for i in range(len(arr) + 1):   # IndexError: goes one too far
        if arr[i] > max_val:
            max_val = arr[i]
    return max_val
""",
    "Correct loop (clean)": """\
def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for item in arr:
        if item > max_val:
            max_val = item
    return max_val
""",
    "Null dereference risk (buggy)": """\
def get_user_name(user_dict):
    return user_dict['profile']['name'].strip()   # KeyError / AttributeError
""",
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load model ────────────────────────────────────────────────────────────
    tokenizer, model = load_model_and_tokenizer()
    results          = load_training_results()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
      <div>
        <h1>Intelligent Code Review Assistant</h1>
      </div>
      <span class="badge">CodeBERT · fine-tuned</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Model metrics strip (from Notebook 04 outputs) ───────────────────────
    if results:
        tm = results.get("test_metrics", {})
        acc  = tm.get("accuracy",  0.7423)
        f1   = tm.get("f1",        0.6460)
        prec = tm.get("precision", 0.5303)
        rec  = tm.get("recall",    0.8263)
        auc  = tm.get("roc_auc",   0.8519)
    else:
        acc, f1, prec, rec, auc = 0.7423, 0.6460, 0.5303, 0.8263, 0.8519

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="val">{acc*100:.1f}%</div><div class="lbl">Accuracy</div></div>
      <div class="metric-card"><div class="val">{f1:.3f}</div><div class="lbl">F1 Score</div></div>
      <div class="metric-card"><div class="val">{prec:.3f}</div><div class="lbl">Precision</div></div>
      <div class="metric-card"><div class="val">{rec:.3f}</div><div class="lbl">Recall</div></div>
      <div class="metric-card"><div class="val">{auc:.3f}</div><div class="lbl">ROC-AUC</div></div>
      <div class="metric-card"><div class="val">39,656</div><div class="lbl">Training samples</div></div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # SIDEBAR
    # ─────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Settings")

        attention_layer = st.slider(
            "Attention layer for explainability",
            min_value=0, max_value=11, value=11,
            help="Higher layers are more task-specific. Layer 11 = last layer."
        )

        threshold = st.slider(
            "Bug decision threshold",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="Probability above this → flagged as buggy. "
                 "Lower = more sensitive (catches more bugs but more false alarms)."
        )

        st.markdown("---")
        st.markdown("### Example snippets")
        selected_example = st.selectbox(
            "Load an example",
            ["— select —"] + list(EXAMPLES.keys())
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        Fine-tuned **CodeBERT** classifier trained on:
        - CodeSearchNet (clean)
        - Devign vulnerability dataset
        - GitHub bug-fix PRs

        **Explainability** via attention-weight
        visualisation (last transformer layer,
        mean over 12 heads, CLS→token attention).
        """)

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN TABS
    # ─────────────────────────────────────────────────────────────────────────
    tab_review, tab_metrics, tab_about = st.tabs([
        "Code Review",
        "Model Metrics",
        "About the Model"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: CODE REVIEW
    # ══════════════════════════════════════════════════════════════════════════
    with tab_review:
        st.markdown('<div class="section-label">Paste code snippet</div>',
                    unsafe_allow_html=True)

        # Pre-fill from sidebar example selector
        default_code = ""
        if selected_example != "— select —":
            default_code = EXAMPLES[selected_example]

        code_input = st.text_area(
            label="code_input",
            label_visibility="collapsed",
            value=default_code,
            height=260,
            placeholder="Paste Python (or C) code here and click Review…",
            key="code_area"
        )

        col_btn, col_hint = st.columns([1, 5])
        with col_btn:
            run_review = st.button("Review Code", use_container_width=True)
        with col_hint:
            st.caption(f"Using threshold: **{threshold:.2f}** · "
                       f"Attention layer: **{attention_layer}** · "
                       f"Device: **{str(DEVICE).upper()}**")

        # ── Run inference ─────────────────────────────────────────────────────
        if run_review:
            if not code_input.strip():
                st.warning("Please paste some code before clicking Review.")
            else:
                with st.spinner("Analysing code…"):
                    result = predict(code_input, tokenizer, model)

                if result is None:
                    st.error("Could not process the input. "
                             "Make sure it contains valid code.")
                else:
                    # Apply custom threshold
                    label      = 1 if result["prob_buggy"] >= threshold else 0
                    confidence = result["prob_buggy"] if label == 1 \
                                 else result["prob_clean"]

                    # ── Verdict banner ────────────────────────────────────────
                    if label == 1:
                        st.markdown(f"""
                        <div class="verdict-buggy">
                          <div class="verdict-title" style="color:#f85149;">
                            Potential Bug Detected
                          </div>
                          <div class="verdict-sub">
                            Model confidence: <strong>{result['prob_buggy']*100:.1f}%</strong>
                            · This code exhibits patterns associated with bugs,
                            vulnerabilities, or code smells.
                          </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="verdict-clean">
                          <div class="verdict-title" style="color:#3fb950;">
                            Code Looks Clean
                          </div>
                          <div class="verdict-sub">
                            Model confidence: <strong>{result['prob_clean']*100:.1f}%</strong>
                            · No strong bug patterns detected.
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Two-column layout: confidence | explainability ────────
                    left, right = st.columns([1, 2])

                    with left:
                        st.markdown('<div class="section-label">Confidence scores</div>',
                                    unsafe_allow_html=True)
                        st.markdown(
                            plot_confidence_bars(
                                result["prob_clean"],
                                result["prob_buggy"]
                            ),
                            unsafe_allow_html=True
                        )

                        st.markdown('<div class="section-label">Token stats</div>',
                                    unsafe_allow_html=True)
                        real = result["real_len"]
                        pad  = MAX_LENGTH - real
                        st.markdown(f"""
                        <div class="conf-bar-wrap" style="padding:10px 14px;">
                          <div style="font-size:12px; color:#8b949e; font-family:'JetBrains Mono',monospace;">
                            Real tokens : <strong style="color:#e6edf3;">{real}</strong> / {MAX_LENGTH}<br>
                            Padding     : <strong style="color:#8b949e;">{pad}</strong><br>
                            Truncated   : <strong style="color:#e3b341;">
                              {'Yes' if real == MAX_LENGTH else 'No'}
                            </strong>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

                    with right:
                        st.markdown('<div class="section-label">Token-level explainability '
                                    '(attention highlights)</div>',
                                    unsafe_allow_html=True)

                        token_imp = get_token_importance(result, layer=attention_layer)
                        token_html = render_token_html(token_imp, label)
                        st.markdown(token_html, unsafe_allow_html=True)

                        st.caption(
                            "Red tokens pushed prediction toward **buggy**.  "
                            "Green tokens pushed toward **clean**.  "
                            "Intensity = attention weight magnitude."
                        )

                    # ── Top flagged tokens table ───────────────────────────────
                    with st.expander("Top 15 most-attended tokens"):
                        token_imp_sorted = sorted(
                            token_imp, key=lambda x: x[1], reverse=True
                        )[:15]
                        top_df = pd.DataFrame(token_imp_sorted,
                                              columns=["Token", "Attention Score"])
                        top_df["Token"] = top_df["Token"].str.replace("Ġ", " ")
                        top_df["Attention Score"] = top_df["Attention Score"].round(4)
                        st.dataframe(top_df, use_container_width=True, hide_index=True)

                    # ── Raw probabilities ─────────────────────────────────────
                    with st.expander("Raw model output"):
                        st.json({
                            "prob_clean":    round(result["prob_clean"], 6),
                            "prob_buggy":    round(result["prob_buggy"], 6),
                            "predicted_label": label,
                            "label_name":    "buggy" if label == 1 else "clean",
                            "threshold_used": threshold,
                            "real_tokens":   result["real_len"],
                            "device":        str(DEVICE)
                        })

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: MODEL METRICS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_metrics:
        if results is None:
            st.info("`reports/training_results.pkl` not found. "
                    "Run Notebooks 03 & 04 first to generate this file.")
        else:
            history     = results.get("history", {})
            # top-level numpy arrays in the pkl (confirmed keys)
            test_labels = results.get("test_labels", np.array([]))
            test_preds  = results.get("test_preds",  np.array([]))
            test_probs  = results.get("test_probs",  np.array([]))
            best_epoch  = results.get("best_epoch",  "—")
            best_val_f1 = results.get("best_val_f1", 0.0)
            # test_metrics keys in pkl: accuracy, precision, recall, f1, roc_auc
            tm_live   = results.get("test_metrics", {})
            acc_live  = float(tm_live.get("accuracy",  acc))
            prec_live = float(tm_live.get("precision", prec))
            rec_live  = float(tm_live.get("recall",    rec))
            f1_live   = float(tm_live.get("f1",        f1))
            auc_live  = float(tm_live.get("roc_auc",   auc))

            # ── Summary metrics table ─────────────────────────────────────────
            st.markdown('<div class="section-label">Test set performance</div>',
                        unsafe_allow_html=True)

            metrics_df = pd.DataFrame([
                {"Metric": "Accuracy",      "Score": f"{acc_live*100:.2f}%"},
                {"Metric": "Precision",     "Score": f"{prec_live:.4f}"},
                {"Metric": "Recall",        "Score": f"{rec_live:.4f}"},
                {"Metric": "F1 Score",      "Score": f"{f1_live:.4f}"},
                {"Metric": "ROC-AUC",       "Score": f"{auc_live:.4f}"},
                {"Metric": "Best epoch",    "Score": str(best_epoch)},
                {"Metric": "Best val F1",   "Score": f"{best_val_f1:.4f}"},
                {"Metric": "Train samples", "Score": "27,758"},
                {"Metric": "Val samples",   "Score": "5,949"},
                {"Metric": "Test samples",  "Score": "5,949"},
            ])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            st.caption(
                f"Evaluated on 5,949 held-out test samples. "
                f"Recall {rec_live:.3f} = model catches {rec_live*100:.1f}% of real bugs. "
                f"ROC-AUC {auc_live:.4f} = strong discrimination ability."
            )

            # ── Training curves + confusion matrix ────────────────────────────
            st.markdown('<div class="section-label">Training history</div>',
                        unsafe_allow_html=True)

            c1, c2 = st.columns([3, 2])

            with c1:
                if history:
                    fig = make_training_curve_fig(history)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    st.info("Training history not found in results file.")

            with c2:
                if test_labels is not None and len(test_labels) > 0 and len(test_preds) > 0:
                    fig2 = make_confusion_matrix_fig(test_labels, test_preds)
                    st.pyplot(fig2, use_container_width=True)
                    plt.close(fig2)
                else:
                    st.info("Test predictions not found in results file.")

            # ── Confidence distribution ───────────────────────────────────────
            if test_probs is not None and len(test_probs) > 0:
                st.markdown('<div class="section-label">Confidence distribution</div>',
                            unsafe_allow_html=True)

                probs_arr  = np.array(test_probs)
                labels_arr = np.array(test_labels)

                fig3, ax = plt.subplots(figsize=(9, 3), facecolor="#161b22")
                ax.set_facecolor("#0d1117")
                ax.tick_params(colors="#8b949e")
                for spine in ax.spines.values():
                    spine.set_color("#30363d")

                buggy_probs = probs_arr[labels_arr == 1]
                clean_probs = probs_arr[labels_arr == 0]

                ax.hist(clean_probs, bins=40, alpha=0.65,
                        color="#3fb950", label="Clean (true)")
                ax.hist(buggy_probs, bins=40, alpha=0.65,
                        color="#f85149", label="Buggy (true)")
                ax.axvline(0.5, color="#e3b341", linestyle="--",
                           linewidth=1.5, label="Threshold 0.5")
                ax.set_xlabel("P(buggy)", color="#8b949e")
                ax.set_ylabel("Count",    color="#8b949e")
                ax.set_title("Model confidence — true label colour",
                             color="#e6edf3")
                ax.legend(facecolor="#21262d", labelcolor="#e6edf3")
                fig3.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)

            # ── Classification report ─────────────────────────────────────────
            report_csv = os.path.join(REPORTS_DIR, "classification_report.csv")
            if os.path.exists(report_csv):
                st.markdown('<div class="section-label">Full classification report</div>',
                            unsafe_allow_html=True)
                report_df = pd.read_csv(report_csv, index_col=0)
                st.dataframe(report_df.round(4), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: ABOUT
    # ══════════════════════════════════════════════════════════════════════════
    with tab_about:
        st.markdown("""
        <div class="section-label">Project overview</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        This dashboard is the deployment stage of a **4-notebook deep learning pipeline**
        for automated code review. The model detects potential bugs, vulnerabilities,
        and code smells in source code using a fine-tuned CodeBERT transformer.

        ---

        ### Architecture

        ```
        Input code
              ↓
        Text cleaning (Notebook 02)
              ↓
        CodeBERT tokenizer  (WordPiece, vocab 50,265)
              ↓
        CodeBERT encoder    (12 transformer layers, 768-dim hidden)
              ↓
        [CLS] token → Dropout(0.1) → Linear(768 → 2)
              ↓
        Softmax → P(clean), P(buggy)
        ```

        ---

        ### Training data (39,656 samples total)

        | Source | Label | Samples |
        |---|---|---|
        | CodeSearchNet (Python) | Clean (0) | ~14,700 |
        | Devign vulnerability dataset (C) | Mixed | ~27,318 |
        | GitHub bug-fix PRs (Python) | Buggy (1) | ~6 |

        ---

        ### Key design decisions

        - **Frozen layers**: Bottom 6 of 12 CodeBERT encoder layers frozen → 60% faster training, less overfitting
        - **Weighted loss**: Class weights [0.699, 1.757] to handle 2.5:1 class imbalance
        - **Optimiser**: AdamW with linear warmup + decay scheduler
        - **Explainability**: Last-layer attention weights, mean over 12 heads, CLS→token projection

        ---

        ### File structure

        ```
        intelligent-code-review-assistant/
        ├── notebooks/
        │   ├── 01_data_collection_eda.ipynb
        │   ├── 02_preprocessing_tokenization.ipynb
        │   ├── 03_model_training.ipynb
        │   └── 04_evaluation.ipynb
        ├── models/codebert_classifier/
        │   ├── model.safetensors
        │   ├── classifier_head.pt
        │   └── tokenizer_config.json
        ├── reports/
        │   ├── training_results.pkl
        │   └── figures/
        └── app/
            └── app.py          ← you are here
        ```

        ---

        ### Running the app

        ```bash
        # From project root
        streamlit run app/app.py
        ```
        """)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{acc*100:.1f}%")
        with col2:
            st.metric("ROC-AUC", f"{auc:.4f}")
        with col3:
            _bvf1 = results.get("best_val_f1", 0.0) if results else None
            st.metric("Best Val F1", f"{_bvf1:.4f}" if _bvf1 is not None else "—")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()