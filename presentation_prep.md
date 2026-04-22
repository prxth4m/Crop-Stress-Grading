# 🎓 Complete Presentation Prep — Crop Stress Grading Project

---

# PART 1: THE NARRATIVE (What to Say)

Build your presentation in this exact order. Each section flows into the next.

---

## 1. Problem Statement — Start Here

> **Opening line:** *"Food security is one of the biggest challenges of our time. Crop stress — caused by drought, pests, disease, or nutrient deficiency — destroys harvests, but the damage is often invisible to the human eye until it's too late."*

### The Real-World Problem
- Crops under environmental stress go through **distinct physiological stages** before visible symptoms appear
- By the time a farmer *sees* yellowing or wilting, significant yield loss has already occurred
- Ground-level scouting is **expensive, slow, and covers tiny areas**
- We need a way to detect stress **early, remotely, and at scale**

### Why Satellites?
- The **EO-1 (Earth Observing-1)** satellite carries the **Hyperion** hyperspectral sensor
- It captures reflectance across **242 spectral bands** (400–2500 nm) — visible light through shortwave infrared
- This is essentially a complete **"spectral fingerprint"** of the Earth's surface at each pixel
- Different crop conditions cause measurably different absorption patterns at specific wavelengths

### Why Deep Learning?
- Traditional vegetation indices (like NDVI) use only **2–3 bands** and lose most spectral information
- Deep learning can process all **131+ usable bands simultaneously** to detect subtle patterns invisible to simple indices

### The Gap We're Filling
- Most existing models are **black boxes** — they predict a stress grade but can't explain *why*
- Agricultural experts won't trust a model that just says "stress level 4" without justification
- **Our research question:** *Can we classify crop stress stages from hyperspectral data AND explain which spectral bands drove the decision?*

---

## 2. Our Solution — What We Built

> *"We built an end-to-end Explainable Deep Learning pipeline that does three things: preprocesses raw satellite data, classifies crop stress using four different architectures, and then opens up the black box using three XAI techniques."*

### The Three Pillars

| Pillar | What | Why |
|--------|------|-----|
| **Preprocessing** | Savitzky-Golay filtering, derivative spectroscopy, vegetation indices, SMOTE | Clean noisy satellite data, extract meaningful features, handle class imbalance |
| **Classification** | 4 architectures: CNN, ResNet, Transformer, Hybrid | Compare local vs global spectral pattern recognition |
| **Explainability** | Grad-CAM, GradientSHAP, Attention Weights | Map model decisions back to physically meaningful spectral bands |

### Pipeline Flow (Top to Bottom)
```
Raw EO-1 CSV (6,988 rows × 242 bands)
    ↓ Drop noisy bands (>50% NAs) → 131 bands remain
    ↓ Savitzky-Golay smoothing → remove sensor noise
    ↓ 1st derivative spectroscopy → highlight absorption edges
    ↓ Compute MLVI + H_VSI vegetation indices
    ↓ StandardScaler normalisation
    ↓ Stratified 70/15/15 split
    ↓ SMOTE augmentation (training set only)
    ↓ → 264 features per sample
    ↓
    ↓ Train 4 models (AdamW + Cosine Annealing + Early Stopping)
    ↓
    ↓ Run XAI (Grad-CAM + GradientSHAP + Attention)
    ↓
    ↓ Generate comparison plots + heatmaps
    ↓
    → Results: Best model = Hybrid at 89.9% accuracy
```

---

## 3. The Data Pipeline — What We Did and Why

### 3.1 Raw Data
- **Source:** EO-1 Hyperion hyperspectral sensor
- **Format:** CSV with columns `X<wavelength>` (e.g., `X427`, `X680`, `X854`, `X2133`)
- **Target:** `Stage` — categorical label for crop stress stage (6 classes)
- **Raw size:** ~6,988 rows

### 3.2 Quality Filtering
- Drop spectral bands with **>50% missing values** (atmospheric interference, sensor calibration gaps)
- **198 bands → 131 usable bands**
- Drop remaining rows with any NAs → **6,933 clean rows**

### 3.3 Savitzky-Golay Filtering
> *Referenced from: signal processing literature, widely used in remote sensing for spectral smoothing*

- Fits a **polynomial (degree 2)** over a sliding window of **11 bands**
- Unlike a moving average, it **preserves spectral shape** — peaks and valleys stay intact
- Removes high-frequency sensor noise while keeping meaningful absorption features

### 3.4 Derivative Spectroscopy (1st Derivative)
> *Referenced from: analytical chemistry and remote sensing literature*

- Computes `np.gradient(X_smoothed, axis=1)` — rate of change of reflectance across bands
- Highlights **inflection points** where reflectance changes slope sharply
- These correspond to **absorption edges** (e.g., the "red edge" at ~700 nm)
- Also removes **additive baseline shifts** from atmospheric scattering

### 3.5 Vegetation Indices
Two custom indices computed from specific wavelengths:

**MLVI** = (NIR − SWIR1) / SWIR2
- Uses bands at ~854nm, ~1649nm, ~2133nm
- Captures **moisture content and canopy structure** changes

**H_VSI** = (NIR − SWIR1) / (NIR + SWIR1 + SWIR2)
- Normalised ratio — removes illumination variation
- More robust across different satellite passes

### 3.6 Final Feature Vector
```
131 smoothed bands + 131 derivative bands + MLVI + H_VSI = 264 features
```

### 3.7 Normalisation
- **StandardScaler** (Z-score): mean=0, std=1 for each feature
- Fitted only on training data to prevent data leakage

### 3.8 Splitting & SMOTE
- **Stratified split:** 70% train / 15% val / 15% test (preserves class distribution)
- **SMOTE** applied **only to training set** — creates synthetic minority samples by interpolating between existing ones in feature space
- Val and test remain untouched (reflect real-world distribution)

---

## 4. The Four Model Architectures

> *"We compare four architectures to answer: does capturing local spectral patterns, global spectral patterns, or both give the best results?"*

### 4.1 Paper CNN — The Baseline (26,982 parameters)
> *Referenced from: standard 1D CNN approaches for spectral classification*

```
Input (264,1) → Conv1d(1→16, k=3) → ReLU → Conv1d(16→32, k=3) → ReLU
    → MaxPool(2) → Dropout(0.3) → Flatten → Linear → 6 classes
```

- Two convolutional layers with **kernel size 3** — each band "sees" only its 2 nearest neighbours
- Captures **local spectral patterns** (adjacent band correlations, absorption features)
- **Limitation:** Cannot capture relationships between distant bands (e.g., 680nm and 2133nm)

### 4.2 ResNet — Deeper with Skip Connections (109,782 parameters)
> *Referenced from: He et al., "Deep Residual Learning for Image Recognition" (2015) — adapted to 1D*

```
Input → Conv1d(1→16, k=7, stride=2) → ResBlock(16→32) → ResBlock(32→64, s=2)
    → ResBlock(64→128, s=2) → AdaptiveAvgPool → Dropout → Linear → 6 classes
```

**Each ResBlock:**
```
x → Conv → BatchNorm → ReLU → Conv → BatchNorm → (+x) → ReLU
                                                     ↑
                                            skip connection
```

- **Skip connections** solve the **vanishing gradient problem** — gradients flow directly to earlier layers
- Enables deeper networks without degradation
- Each block learns "what to change" rather than "what to output from scratch"
- **AdaptiveAvgPool** makes it input-length agnostic

### 4.3 Transformer — Global Attention (4,492,550 parameters)
> *Referenced from: Vaswani et al., "Attention Is All You Need" (2017) — adapted from NLP/Vision to 1D spectral sequences*

```
Input (264,1) → Linear(1→64) → +Positional Encoding
    → TransformerEncoder(3 layers, 4 heads, ff=256) → Flatten → MLP → 6 classes
```

**Key Concepts:**
- **Linear embedding:** Projects each band from dimension 1 to 64 (richer representation)
- **Positional encoding:** Sinusoidal (sin/cos) — injects band position because Transformers have no inherent notion of order
- **Self-attention:** `Attention(Q,K,V) = softmax(QK^T / √d) × V` — every band attends to every other band
- **4 attention heads** = 4 different relationship patterns learned in parallel
- **Global receptive field from layer 1** — no locality constraint

### 4.4 Hybrid — CNN + Transformer Combined (2,286,406 parameters)
> *Our novel architecture combining the strengths of both approaches*

```
Input → Conv1d(1→32, k=3) + BN + ReLU → Conv1d(32→64, k=3) + BN + ReLU
    → (+skip via 1×1 Conv) → MaxPool(2) → [264 → 132 tokens]
    → TransformerEncoder(2 layers, 4 heads) → Flatten → MLP → 6 classes
```

**Why it's the best:**
1. **CNN stage** extracts local spectral features (absorption edges, adjacent band patterns)
2. **Skip connection** (1×1 conv projection) preserves raw spectral information alongside CNN features
3. **MaxPool** halves sequence length — Transformer processes 132 enriched tokens instead of 264 raw bands
4. **Transformer stage** captures global relationships between these locally-processed features
5. Result: **local feature extraction + global context = best of both worlds**

**Critical insight:** The Hybrid achieves **higher accuracy (89.9%) with fewer parameters (2.3M) than the pure Transformer (4.5M)** because the CNN pre-processing gives the Transformer richer input tokens.

---

## 5. Training Strategy

### All four models share this recipe:

| Component | Choice | Why |
|-----------|--------|-----|
| **Optimizer** | AdamW (lr=1e-3, weight_decay=1e-4) | Proper L2 regularisation decoupled from adaptive rates |
| **Loss** | CrossEntropy + Label Smoothing (ε=0.1) | Soft targets prevent overconfidence |
| **Scheduler** | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) | Periodic restarts escape local minima |
| **Gradient clipping** | max_norm=1.0 | Prevents exploding gradients in Transformer layers |
| **Early stopping** | Patience=15 on validation MCC | Stops when model stops improving; saves best weights |
| **Epochs** | Up to 60 | Early stopping typically triggers around 30-45 |
| **Batch size** | 64 | Balance between gradient noise and computation |

### Why MCC (Matthews Correlation Coefficient)?
- Accuracy is misleading with imbalanced classes
- MCC accounts for all four confusion matrix quadrants (TP, TN, FP, FN)
- Range: -1 (perfectly wrong) to +1 (perfect)
- Our best MCC = 0.874 → genuinely learning all classes

---

## 6. Explainability (XAI) — The Key Differentiator

> *"This is what separates our project from a generic classification exercise."*

### 6.1 Grad-CAM
> *Referenced from: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017)*

- Works on CNN layers (Paper CNN, ResNet, Hybrid)
- Hooks into the **last Conv1d layer**, captures forward activations and backward gradients
- Computes: `cam = ReLU(Σ(gradient_weights × activations))`
- Upsamples to original 264-band length
- Shows which spectral regions **positively contributed** to the prediction

### 6.2 GradientSHAP
> *Referenced from: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (SHAP, 2017) — implemented via Facebook's Captum library*

- Works on **all four models** (model-agnostic)
- Based on **Shapley values** from game theory — fairly attributes output to each input feature
- Uses a **zero baseline** (no signal) and computes expected gradients along the path from baseline to input
- Provides **per-band, per-class attribution scores**
- We take absolute values (both positive and negative attributions indicate importance)

### 6.3 Attention Weights
> *Intrinsic to Transformer architecture*

- Extracts self-attention matrices from Transformer/Hybrid encoder layers
- Monkey-patches the forward pass to capture `need_weights=True`
- Averages across heads and layers → takes **diagonal** (each band's self-attention score)
- Shows what the model "naturally focused on" — no extra computation needed

### The Convergence Argument
> *"The strongest evidence that our model learned real physics is that three completely different XAI techniques — gradient-based (Grad-CAM), perturbation-based (SHAP), and architecture-intrinsic (Attention) — all converge on the same spectral bands. These bands correspond to known agrophysical phenomena: chlorophyll absorption at ~680nm, the red-edge at ~700-750nm, and SWIR moisture bands at ~1600-2200nm."*

---

## 7. Results

| Model | Accuracy | F1 Score | MCC | Parameters |
|-------|----------|----------|-----|------------|
| Paper CNN | 85.38% | 85.41% | 0.819 | 26,982 |
| ResNet | 86.63% | 86.62% | 0.834 | 109,782 |
| Transformer | 87.98% | 87.96% | 0.851 | 4,492,550 |
| **Hybrid** | **89.90%** | **89.92%** | **0.874** | **2,286,406** |

### Interpreting the Progression
- **CNN → ResNet (+1.25%):** Skip connections enable deeper feature hierarchies
- **ResNet → Transformer (+1.35%):** Global attention captures long-range spectral dependencies
- **Transformer → Hybrid (+1.92%):** CNN pre-processing gives richer input to Transformer
- **Hybrid is more efficient:** Higher accuracy than Transformer with **half the parameters**

---

## 8. Research Paper References & What We Used From Them

| Reference | What We Used |
|-----------|-------------|
| **He et al. (2015)** — Deep Residual Learning | Skip connection design in ResNet and Hybrid models |
| **Vaswani et al. (2017)** — Attention Is All You Need | Self-attention mechanism, positional encoding, multi-head attention |
| **Selvaraju et al. (2017)** — Grad-CAM | Gradient-weighted class activation mapping for CNN explainability |
| **Lundberg & Lee (2017)** — SHAP | Shapley value-based feature attribution (via GradientSHAP) |
| **Savitzky & Golay (1964)** | Polynomial smoothing filter for spectral noise reduction |
| **Chawla et al. (2002)** — SMOTE | Synthetic minority oversampling for class imbalance |
| **Loshchilov & Hutter (2017)** — AdamW | Decoupled weight decay optimisation |
| **Loshchilov & Hutter (2017)** — SGDR | Cosine annealing with warm restarts scheduler |
| **Remote sensing literature** | Derivative spectroscopy, vegetation indices (MLVI, H_VSI) |
| **Facebook Research — Captum** | Library used for GradientSHAP implementation |

---

# PART 2: LIVE DEMONSTRATION (What to Do)

---

## Pre-Demo Setup (Do This BEFORE the Presentation)

Make sure everything is ready:

```powershell
# Navigate to project
cd d:\crop-stress-grading-eo-1-main

# Activate virtual environment
.\venv\Scripts\activate

# Verify Python and key packages work
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import pandas, sklearn, captum; print('All packages OK')"
```

> [!IMPORTANT]
> Models are **already trained** in `checkpoints/Full_Run/`. You do NOT need to retrain during the demo. The demo shows the pipeline stages and then discusses the pre-generated results.

---

## Demo Step 1: Show the Project Structure

```powershell
# Show the clean project structure
Get-ChildItem -Recurse -Name -Exclude venv,__pycache__,.git | Where-Object { $_ -notmatch "venv|__pycache__|\.git" } | Select-Object -First 35
```

**What to say:** Walk through the folder structure — data pipeline, models, explainability, training, scripts. Show it matches the flowchart.

---

## Demo Step 2: Data Analysis (Quick EDA)

```powershell
python -m scripts.analyze_data
```

**Expected output:**
```
Split sizes
Train: ~13860 | Val: ~1040 | Test: ~1040
Total: ~15940
Train/Val/Test ratio: ~87% / ~7% / ~7%
(Train is larger due to SMOTE augmentation)

Features
Number of features: 264

Class Distribution (Train - post SMOTE)  ← balanced by SMOTE
Class Distribution (Val - original)       ← original proportions
Class Distribution (Test - original)      ← original proportions
```

**What to say:** *"Notice the training set is much larger because SMOTE generated synthetic samples. But validation and test sets preserve the original distribution — we never augment evaluation data."*

---

## Demo Step 3: Show the Code — Preprocessing Pipeline

Open `data_pipeline/preprocessing.py` and walk through:

1. **Lines 56-63:** Loading raw CSV, skipping metadata rows
2. **Lines 73-81:** Selecting spectral columns (`X*`), dropping >50% NA columns
3. **Lines 93-100:** Savitzky-Golay filtering — explain the polynomial smoothing
4. **Lines 102-103:** 1st derivative — explain inflection point detection
5. **Lines 105-113:** MLVI and H_VSI computation — explain the vegetation indices
6. **Lines 108-114:** Feature concatenation → 264 total features
7. **Lines 117-118:** StandardScaler normalisation
8. **Lines 130-131:** Stratified train/val/test split
9. **Lines 138-144:** SMOTE augmentation on training set only

---

## Demo Step 4: Show the Code — Model Architectures

Open each model file and **highlight the key design element:**

### `models/paper_cnn.py`
- **Lines 8-17:** Two Conv1d layers → MaxPool → Dropout
- **Key point:** Simple, local receptive field only

### `models/resnet.py`
- **Lines 5-26:** ResBlock with the **skip connection** at line 24: `out += self.shortcut(x)`
- **Lines 29-56:** Three ResBlocks going deeper with stride-2 downsampling
- **Key point:** Skip connections prevent vanishing gradients

### `models/transformer.py`
- **Lines 7-18:** Sinusoidal positional encoding
- **Lines 22-47:** Linear embedding → Positional Encoding → 3-layer Transformer Encoder
- **Key point:** Every band attends to every other band (global context)

### `models/hybrid.py`
- **Lines 8-18:** Two Conv blocks + skip projection (line 18)
- **Lines 23-27:** Transformer encoder on top of CNN features
- **Line 44:** Skip connection: `x = x + residual`
- **Key point:** Local CNN features → MaxPool → Global Transformer attention

---

## Demo Step 5: Show the Code — Training Strategy

Open `training/trainer.py`:
- **Line 21:** Gradient clipping: `clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **Lines 65-69:** History tracking (train_loss, val_loss, val_acc, val_mcc)
- **Lines 94-98:** Early stopping on MCC — save best weights only

Open `scripts/train.py`:
- **Line 82:** CrossEntropyLoss with label smoothing and class weights
- **Line 83:** AdamW optimizer
- **Lines 84-86:** CosineAnnealingWarmRestarts scheduler

---

## Demo Step 6: Show Pre-Generated Results

### 6a. Model Metrics (JSON)

```powershell
python -c "import json;d=json.load(open('_results/raw/Full_Run/hybrid.json'));m=d['test_metrics'];print('Accuracy:',round(m['accuracy'],4));print('F1:',round(m['f1_score'],4));print('MCC:',round(m['mcc'],4))"
```

**Expected:**
```
Accuracy: 0.899
F1: 0.8992
MCC: 0.8745
```

### 6b. Open the Comparison Plots

Open these files in the image viewer and discuss each:

```
_results/plots/Full_Run/comparison/test_metrics_comparison.png   ← bar chart: 4 models compared
_results/plots/Full_Run/comparison/loss_comparison.png           ← training dynamics
_results/plots/Full_Run/comparison/val_metrics_comparison.png    ← accuracy/MCC over epochs
_results/plots/Full_Run/per_model/hybrid_confusion_matrix.png   ← where Hybrid gets confused
_results/plots/Full_Run/per_model/hybrid_loss_curve.png         ← train vs val loss
```

**What to say for confusion matrix:** *"Most misclassifications happen between adjacent stress stages — which makes biological sense because the spectral difference between early and mid-stress is subtle."*

**What to say for loss curve:** *"Notice the periodic jumps — that's the cosine annealing scheduler restarting the learning rate. Each time it lets the model explore new parameter regions."*

---

## Demo Step 7: Generate XAI Maps (Live!)

This is the **most impressive live demo moment.** Run this command:

```powershell
python -m scripts.explain --exp Full_Run --model hybrid --method all
```

**What to say while it runs (takes ~30-60 seconds):**
> *"This loads our best Hybrid model checkpoint and runs three XAI techniques simultaneously. Grad-CAM traces gradients back through the CNN layers. GradientSHAP computes Shapley values against a zero baseline. And we extract the raw attention weights from the Transformer component."*

**After it finishes, open the output files:**

```
_results/explain/Full_Run/hybrid/top_bands.png           ← TOP 20 MOST IMPORTANT BANDS
_results/explain/Full_Run/hybrid/gradcam_mean.png        ← CNN saliency per class
_results/explain/Full_Run/hybrid/gradientshap_mean.png   ← SHAP attribution curves
_results/explain/Full_Run/hybrid/attention_mean.png      ← Transformer attention focus
_results/explain/Full_Run/hybrid/gradientshap_heatmap.png ← class × band heatmap
```

**Key talking point for `top_bands.png`:**
> *"These are the 20 spectral bands the model considers most important. They cluster around known agrophysical wavelengths — chlorophyll absorption near 680nm, the red-edge at 700-750nm, and SWIR moisture bands. The model independently discovered the same physics that remote sensing scientists have known for decades. This proves it's learning real signal, not dataset artifacts."*

---

## Demo Step 8: Regenerate Plots (Optional, Quick)

```powershell
python -m scripts.plot_results --exp Full_Run
```

**Expected:** `Plots saved to _results/plots/Full_Run/`

This regenerates all comparison charts from the saved JSON metrics.

---

## Demo Step 9: (If Time Permits) Train a Single Model Live

Only do this if the professor wants to see actual training:

```powershell
# Train just the Paper CNN (fastest, ~2-3 minutes)
python -m scripts.train --model paper_cnn --exp LiveDemo --epochs 10
```

**What to say:**
> *"I'm training the simplest model for 10 epochs so you can see the training loop in action. Watch the loss decrease and the accuracy climb. In our full experiment, we trained all four models for up to 60 epochs each."*

---

## Quick Summary for Closing

> *"To summarise: we built an end-to-end pipeline that takes raw EO-1 hyperspectral satellite data, preprocesses it using Savitzky-Golay filtering and derivative spectroscopy, trains four deep learning architectures — CNN, ResNet, Transformer, and a novel Hybrid model — achieving 89.9% accuracy on 6-class crop stress classification. Most importantly, three independent XAI techniques all converge on the same physically meaningful spectral bands, proving our model learns genuine agrophysical signal. This makes the predictions trustworthy for real-world precision agriculture."*

---

## Emergency Q&A (Quick Answers)

| Question | Answer |
|----------|--------|
| Why not Random Forest/SVM? | Can't exploit the sequential ordering of spectral bands like CNN/Transformer can |
| Why 1D convolutions, not 2D? | Input is a single spectral sequence, not an image — no spatial height dimension |
| Why SMOTE, not just duplicating? | SMOTE creates **new** synthetic samples via interpolation, preventing memorisation |
| Why MCC, not accuracy? | MCC handles class imbalance; accuracy can be misleading |
| Why label smoothing? | Prevents overconfidence; soft targets improve generalisation |
| Why AdamW, not Adam? | Proper decoupled weight decay; standard for Transformer training |
| Why gradient clipping? | Transformer attention layers prone to gradient explosions early in training |
| Why the Hybrid beats Transformer? | CNN pre-processes local features → richer tokens for Transformer, with fewer parameters |
| What are the limitations? | Single time-point only; EO-1 specific; could add LIME/Integrated Gradients |
| Future work? | Multi-temporal sequences; cross-sensor generalisation; edge deployment |
