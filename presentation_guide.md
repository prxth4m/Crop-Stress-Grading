# 🌾 Crop Stress Grading — Presentation & Demonstration Guide

**Project:** Explainable Deep Learning for Crop Stress Grading using EO-1 Hyperspectral Satellite Data  
**Models trained:** Paper CNN · ResNet1D · Transformer · Hybrid  
**Goal of this guide:** Give you everything you need to explain the project confidently — what problem is being solved, how each piece works, and how to run a live demo.

---

## 📋 Table of Contents

1. [The Problem — Why This Matters](#1-the-problem--why-this-matters)
2. [The Data — What We're Working With](#2-the-data--what-were-working-with)
3. [Data Pipeline — How Raw Satellite Data Becomes Model Input](#3-data-pipeline--how-raw-satellite-data-becomes-model-input)
4. [The Four Model Architectures](#4-the-four-model-architectures)
5. [Training Strategy — How We Train Responsibly](#5-training-strategy--how-we-train-responsibly)
6. [Explainability (XAI) — Making the Black Box Transparent](#6-explainability-xai--making-the-black-box-transparent)
7. [Results — What the Numbers Mean](#7-results--what-the-numbers-mean)
8. [Live Demonstration Commands](#8-live-demonstration-commands)
9. [Frequently Asked Questions (Professor Q&A)](#9-frequently-asked-questions-professor-qa)

---

## 1. The Problem — Why This Matters

### What is Crop Stress?
Crops under environmental stress (drought, disease, nutrient deficiency, pest damage) go through distinct physiological stages. If detected **early**, farmers can intervene before yield loss occurs. If missed, entire harvests are lost.

### Why Satellites?
Ground-level scouting is expensive, slow, and covers tiny areas. The **EO-1 (Earth Observing-1) satellite** carries a **Hyperion hyperspectral sensor** that captures reflectance across **242 spectral bands** (400–2500 nm) from space — essentially a full "spectral fingerprint" of the Earth's surface at each pixel.

### Why Deep Learning?
Traditional vegetation indices (e.g., NDVI) use only 2–3 bands and lose most of the spectral information. Deep learning can process all 131+ usable bands **simultaneously** to detect subtle patterns invisible to simple indices.

### The Research Question
> *Can we accurately classify the stress stage of a crop using EO-1 hyperspectral data, and can we explain which spectral bands drove that decision — so practitioners can trust the model?*

This is the core challenge: **accuracy alone isn't enough** — agricultural experts need to know *why* the model decided what it did.

---

## 2. The Data — What We're Working With

### Source
- **Instrument:** EO-1 Hyperion hyperspectral sensor
- **Input columns:** Spectral reflectance bands named `X<wavelength_nm>` (e.g., `X427`, `X680`, `X854`, `X2133`)
- **Target column:** `Stage` — categorical label indicating the **stress stage** of the crop at that observation
- **Raw observations:** ~6,988 rows before cleaning → **6,933 clean rows** retained

### Why Hyperspectral?
Normal RGB cameras see 3 bands. Hyperion sees **198+ usable bands** across the visible and infrared spectrum. Different crop conditions cause measurably different absorption patterns:

| Spectral Region | Wavelength | Why It Matters |
|---|---|---|
| Visible (Red) | ~680 nm | Chlorophyll absorption — plant health |
| Near Infrared (NIR) | ~854 nm | Cell structure — water content proxy |
| Short-Wave IR 1 (SWIR1) | ~1649 nm | Water stress detection |
| Short-Wave IR 2 (SWIR2) | ~2133 nm | Lignin, cellulose — structural stress |

---

## 3. Data Pipeline — How Raw Satellite Data Becomes Model Input

**Script:** `scripts/prepare_data.py` → calls `data_pipeline/preprocessing.py`

The pipeline runs in a single command and outputs clean, balanced, normalised train/val/test splits.

```
python -m scripts.prepare_data
```

### Step-by-Step Breakdown

#### Step 1: Load & Select Target
```python
df = pd.read_csv(input_path, skiprows=9, na_values=['NA', 'na', ''])
```
The raw CSV has 9 metadata header rows that are skipped. The `Stage` column becomes our target `y`.

#### Step 2: Drop Low-Quality Bands
```
Total spectral columns found: 198
Dropped 67 spectral columns (>50% NAs)
Remaining spectral columns: 131
```
Any spectral band with more than 50% missing values across all observations is dropped. These are bands affected by sensor calibration gaps or atmospheric interference. **131 bands remain.**

#### Step 3: Drop Incomplete Rows
```
Dropped 55 rows with remaining NAs. Keeping 6933 rows.
```
After band filtering, any row still containing a missing value in any band is removed.

#### Step 4: Savitzky-Golay Filtering (Noise Reduction)
```python
X_smoothed = savgol_filter(X_raw.values, window_length=11, polyorder=2, axis=1)
```
- **What it does:** Fits a polynomial of degree 2 over a sliding window of 11 bands, replacing each point with the fitted polynomial value.
- **Why:** Raw satellite spectral measurements contain sensor noise. This filter preserves spectral peaks and troughs (which carry meaning) while removing high-frequency noise. Think of it as "spectral blurring" that keeps important features intact.

#### Step 5: Derivative Spectroscopy
```python
X_deriv1 = np.gradient(X_smoothed, axis=1)
```
- **What it does:** Computes the first-order numerical derivative (rate of change) of reflectance across bands.
- **Why:** The 1st derivative of a spectral curve highlights **inflection points** — the precise wavelengths where reflectance transitions sharply. These transitions correspond to specific absorption features of chlorophyll, water, etc. It removes additive baseline shifts caused by atmospheric scattering (different days, different sun angles).

#### Step 6: Vegetation Indices — MLVI and H_VSI
Two custom multi-band vegetation indices are computed from specific wavelengths:

**MLVI (Multi-band Landsat Vegetation Index):**
```python
mlvi = (nir - swir1) / (swir2 + 1e-8)
```
- Uses NIR (~854nm), SWIR1 (~1649nm), SWIR2 (~2133nm)
- Sensitive to **moisture and canopy structure changes** under stress

**H_VSI (Hyperspectral Vegetation Stress Index):**
```python
h_vsi = (nir - swir1) / (nir + swir1 + swir2 + 1e-8)
```
- Normalised version — ratio removes illumination variation
- More robust to atmospheric differences between satellite passes

These 2 scalar values are **appended** to each sample's feature vector.

#### Step 7: Feature Concatenation
```
Final feature vector per sample = [131 smoothed bands] + [131 derivative bands] + [MLVI] + [H_VSI]
                                 = 264 features total
```

#### Step 8: Z-score Normalisation
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)
```
Each of the 264 features is scaled to zero mean, unit variance. Fitted **only on training data** — prevents data leakage.

#### Step 9: Stratified Train/Val/Test Split
```
Train: 70% | Val: 15% | Test: 15%
```
Stratified by class label — ensures each stress stage class is proportionally represented in all splits.

#### Step 10: SMOTE (Class Imbalance Correction)
```python
sm = SMOTE(random_state=42)
X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
```
- **What it does:** Synthetic Minority Over-sampling Technique. For underrepresented stress classes, it generates synthetic samples by interpolating between existing minority-class samples in feature space.
- **Why:** If one stress stage has far fewer samples, the model will learn to bias toward the majority class. SMOTE ensures all classes have sufficient representation **only in the training set** (val and test are kept original to reflect real-world distribution).

**Outputs written to:**
- `data/splits/train.csv` — SMOTE-augmented training set
- `data/splits/val.csv` — original validation set
- `data/splits/test.csv` — original test set

---

## 4. The Four Model Architectures

All four models operate on the **same 1D spectral sequence** of 264 features treated as a time-series of spectral bands. This is the key insight: instead of treating the features as an unordered tabular vector, we treat its sequential band ordering as **spatial structure**, which allows CNNs and Transformers to extract local and global patterns.

### 4.1 Paper CNN (`models/paper_cnn.py`)
**Architecture: 1D Convolutional Neural Network — The Baseline**

```
Input (batch, 264, 1)
    → transpose → (batch, 1, 264)
    → Conv1d(1→16, k=3) + ReLU
    → Conv1d(16→32, k=3) + ReLU
    → MaxPool1d(k=2) → (batch, 32, 132)
    → Dropout(0.3)
    → Flatten → Linear(32×132 → num_classes)
```

**What it does:** Uses two convolutional layers to extract **local spectral patterns** — absorption features spanning a few adjacent bands (like the chlorophyll red-edge around 680–700nm). MaxPooling reduces dimensionality and provides some translation invariance. Dropout prevents overfitting.

**Strengths:** Fast, simple, captures local band correlations well.  
**Limitations:** Each band can only "see" bands within its kernel window (size 3). Long-range spectral relationships (e.g., relationship between a 680nm feature and a 2133nm feature) are not captured.

**Result: 83.85% accuracy** — strong baseline.

---

### 4.2 ResNet1D (`models/resnet.py`)
**Architecture: 1D Residual Network — Deeper with Skip Connections**

```
Input (batch, 1, 264)   [after transpose]
    → Initial Conv1d(1→16, k=7, stride=2)  → (batch, 16, 132)
    → ResBlock1D(16→32)
    → ResBlock1D(32→64, stride=2)          → (batch, 64, 66)
    → ResBlock1D(64→128, stride=2)         → (batch, 128, 33)
    → AdaptiveAvgPool1d(1)                 → (batch, 128, 1)
    → Dropout + Linear(128 → num_classes)
```

**Each ResBlock:**
```
Input x
    → Conv1d → BatchNorm → ReLU → Conv1d → BatchNorm
    → + shortcut(x)    ← skip connection
    → ReLU
```

**What it does:** The **skip connection** is the key innovation. By adding the original input `x` directly to the output of two convolutions, gradients can flow directly to earlier layers without vanishing. This allows **deeper networks** (more layers) without the degradation problem. Each residual block learns "what to change" about the input rather than "what the output should be from scratch."

**Strengths:** Deeper, can learn more complex non-linear spectral combinations without losing gradient signal. BatchNorm stabilizes training.  
**Why stride-2 blocks?** They act as learned pooling, progressively reducing sequence length while increasing channel depth.

**Result: 85.96% accuracy** — improvement over flat CNN.

---

### 4.3 Transformer (`models/transformer.py`)
**Architecture: Spectral Self-Attention — Global Band Relationships**

```
Input (batch, 264, 1)
    → Linear embedding: (batch, 264, 64)   [each band → 64-dim vector]
    → + Positional Encoding
    → TransformerEncoder (3 layers, 4 attention heads, dim_ff=256)
    → Flatten: (batch, 264×64)
    → Linear(264×64 → 256) → ReLU → Dropout(0.3)
    → Linear(256 → 64) → ReLU
    → Linear(64 → num_classes)
```

**Positional Encoding:**
```python
pe[0, :, 0::2] = sin(position × div_term)
pe[0, :, 1::2] = cos(position × div_term)
```
Since the Transformer processes all bands in parallel (no convolution order), sinusoidal positional encodings inject the **band position** into each band's embedding so the model knows band 5 is different from band 125.

**Self-Attention (what makes it powerful):**  
For each band, self-attention computes how much to "attend to" every other band:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```
- Every band gets to query every other band
- The model can directly learn: "when band 680nm is in absorption, band 854nm tends to respond this way"
- **No locality constraint** — long-range interactions across the full 264-band spectrum are captured in a single attention step

**4 heads** = 4 different "attention patterns" in parallel, each specializing in different relationships.

**Strengths:** Global receptive field from the first layer. Can model complex long-range spectral dependencies.  
**Limitations:** Requires more data to learn, harder to interpret than CNN.

**Result: 88.17% accuracy** — best single-mechanism model.

---

### 4.4 Hybrid (`models/hybrid.py`)
**Architecture: CNN + Transformer Combined — Local + Global**

```
Input (batch, 264, 1)
    → transpose → (batch, 1, 264)
    
    [CNN Stage — Local Feature Extraction]
    residual = Conv1d(1→64, k=1)     ← projection shortcut
    → Conv1d(1→32, k=3) + BN + ReLU
    → Conv1d(32→64, k=3) + BN + ReLU
    → + residual                      ← skip connection
    → MaxPool1d(2) → (batch, 64, 132)
    
    [Transformer Stage — Global Context]
    → transpose → (batch, 132, 64)
    → TransformerEncoder (2 layers, 4 heads, dim_ff=256)
    
    [Classifier Head]
    → Flatten → Linear(64×132 → 256) → ReLU → Dropout(0.3)
    → Linear(256 → 64) → ReLU
    → Linear(64 → num_classes)
```

**What it does:**
1. **CNN stage** first extracts **local spectral features** (adjacent band patterns, absorption edges) — like a feature extractor that "processes the raw signal"
2. **MaxPool** reduces sequence length by half — the Transformer will operate on 132 semantically rich tokens rather than 264 raw band values
3. **Transformer stage** then applies **self-attention over these CNN features** — modelling long-range relationships between spectral regions that were already locally processed
4. **Skip connection in CNN** (shortcut from input → after both conv blocks) prevents vanishing gradients in the local stage

**Why is this best?**  
CNN provides a richer, more structured input to the Transformer. The Transformer doesn't need to "discover" local band structure from scratch — the CNN already did that. They are complementary:
- CNN: "what's happening locally around each wavelength?"
- Transformer: "how do these local features across different spectral regions relate to each other?"

**Result: 90.48% accuracy** — best of all four models.

---

## 5. Training Strategy — How We Train Responsibly

**Script:** `scripts/train.py`  
**Trainer engine:** `training/trainer.py`

```bash
python -m scripts.train --model all --exp ProfessorDemo
```

### Optimizer: AdamW
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```
- **Adam** with decoupled **weight decay** (L2 regularisation that doesn't affect the adaptive learning rate)
- Prevents overfitting by penalising large weight magnitudes

### Loss Function: Cross-Entropy with Label Smoothing
```python
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```
- **Class weights:** Computed inversely proportional to class frequency, passed to the loss to explicitly up-weight underrepresented stress stages
- **Label smoothing (0.1):** Instead of training toward hard one-hot labels (0 or 1), targets become soft: `0.9` for the correct class, `0.1/num_classes` distributed among others. Prevents the model from becoming overconfident. Improves generalisation.

### Learning Rate Scheduler: Cosine Annealing with Warm Restarts
```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
```
- LR follows a cosine curve: starts at `1e-3`, decays to `1e-6`, then **restarts** with a new cycle
- Cycle length doubles each restart: T_0=10 → 10, 20, 40 epochs...
- **Why this works:** Restarts allow the model to escape local minima. The decaying LR also acts as implicit regularisation.

### Gradient Clipping
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
Transformer models are prone to **exploding gradients** during early training. Clipping forces all gradient norms to a maximum of 1.0, stabilising training.

### Early Stopping (patience=15)
```python
best_mcc = -1.0
if metrics['mcc'] > best_mcc:
    torch.save(model.state_dict(), save_path)   # save best weights
    wait = 0
else:
    wait += 1
    if wait >= 15: break
```
The model checkpoint is saved **only when validation MCC improves**. If MCC doesn't improve for 15 consecutive epochs, training stops. The best weights are always reloaded for evaluation.

**Why MCC (Matthews Correlation Coefficient)?**
- Accuracy can be misleading when class distribution is imbalanced (even after SMOTE, val/test are original)
- MCC is a correlation measure between true and predicted labels that accounts for all quadrants of the confusion matrix
- MCC = 1 means perfect prediction; MCC = 0 means random; MCC = -1 means perfectly inverse

### Output Files
```
checkpoints/<exp>/<model_name>.pt    ← saved model weights
_results/raw/<exp>/<model_name>.json ← history + metrics
```

---

## 6. Explainability (XAI) — Making the Black Box Transparent

**Script:** `scripts/explain.py`

```bash
python -m scripts.explain --exp ProfessorDemo --model hybrid --method all
```

The XAI module loads each trained checkpoint and generates visualisations showing **which spectral bands the model relied on** to make its predictions.

### 6.1 Grad-CAM (Gradient-weighted Class Activation Mapping)
**Available on:** Paper CNN, ResNet1D, Hybrid (all conv-based models)

**How it works:**
```python
# Compute gradient of the target class score with respect to final conv layer
out[0, target_class].backward()
alpha = gradients['g'].mean(dim=-1, keepdim=True)   # global average pooling of gradients
cam = (alpha * activations['a']).sum(dim=1)          # weighted sum of feature maps
cam = relu(cam)                                       # keep only positive contributions
```

**In plain English:**
1. Run a forward pass, get the model's class prediction
2. Compute how much changing each feature in the **last convolutional layer** would change the target class score (that's the gradient)
3. Average these gradients across all channels (gives a weight per channel)
4. Multiply the weight of each channel by the **activation** of that channel
5. Sum across all channels → one saliency value per spectral band position
6. Apply ReLU (only keep bands that positively contributed)
7. Upsample back to the original 264-band length

**Output:** A saliency curve showing which bands had the highest activation weighted importance for each stress class.

**What to look for:** If the model is physically reasonable, high-saliency bands should cluster near biologically meaningful wavelengths (green peak ~550nm, red edge ~700nm, NIR plateau ~800nm, water absorption ~1450nm, 1940nm).

### 6.2 GradientShap
**Available on:** All four models (model-agnostic)

```python
explainer = GradientShap(model)
baseline = torch.zeros_like(X).to(device)    # zero = no signal
attrs = explainer.attribute(X, baseline, target=Y)
```

**How it works:**
- Based on SHAP (SHapley Additive exPlanations) — a game-theory concept for fairly attributing model output to each input feature
- GradientShap approximates SHAP values by computing the **expected gradient** along a path from a baseline (all zeros = "background noise") to the actual input
- The attribution for each band = how much changing that band from baseline to its real value contributed to the prediction

**Advantages over Grad-CAM:**
- Works on any model architecture (no need for conv layers)
- Provides **signed attributions** — can tell if a band pushed the prediction toward or away from the target class
- More theoretically grounded (satisfies SHAP axioms)

**Outputs generated:**
- `gradientshap_mean.png` — Per-class attribution curves across all 264 bands
- `gradientshap_heatmap.png` — Class × Band heatmap (viridis colourmap)
- `top_bands.png` — Bar chart of top 20 bands by mean attribution magnitude

### 6.3 Attention Weights
**Available on:** Transformer, Hybrid

```python
# Hooks capture attention weights from all transformer encoder layers
result = attention_all_classes(model, test_loader, num_classes, device, n_samples)
```

**How it works:**
- The Transformer's self-attention computes `softmax(QK^T / √d_k)` — the attention weight matrix
- Each row shows how much a given band "attended to" every other band
- By averaging these weights across all heads and layers, we get: for each position in the sequence, how much attention weight was distributed there across all queries
- Averaged over all samples → mean attention profile per class

**What it reveals:** Which parts of the spectral sequence the Transformer naturally "focused on" across different stress stages. This is the model's own internal view of important features — not backpropagated, but forward-pass derived.

### Summary of XAI Methods

| Method | Models | Mechanistic Basis | Granularity |
|--------|--------|---|---|
| Grad-CAM | CNN, ResNet, Hybrid | Gradient × Activation at last conv layer | Band-level saliency |
| GradientShap | All | Expected gradient wrt baseline (SHAP approximation) | Band-level signed attribution |
| Attention Weights | Transformer, Hybrid | Self-attention distribution across the sequence | Band-level attention score |

Running `--method all` generates all applicable methods for a given model and saves everything to `_results/explain/<exp>/<model_name>/`.

---

## 7. Results — What the Numbers Mean

```bash
python -m scripts.plot_results --exp ProfessorDemo
```

### Accuracy

| Model | Accuracy (%) | F1 Score (%) |
|---|---|---|
| Paper CNN | 83.85 | 83.88 |
| ResNet1D | 85.96 | 86.00 |
| Transformer | 88.17 | 88.19 |
| **Hybrid** | **90.48** | **90.48** |

### Interpreting the Progression
- **CNN → ResNet:** Skip connections solve the vanishing gradient problem, enabling a deeper 3-block architecture. More complex feature hierarchies → +2.11% accuracy.
- **ResNet → Transformer:** Global receptive field via self-attention captures long-range spectral relationships inaccessible to convolutions → +2.21%.
- **Transformer → Hybrid:** CNN pre-processes local features before they enter the Transformer. Richer input tokens → +2.31%.

### Generated Plots (in `_results/plots/<exp>/`)
```
per_model/
    <model>_loss_curve.png       ← Train vs Val loss over epochs
    <model>_confusion_matrix.png ← True vs Predicted class heatmap

comparison/
    loss_comparison.png          ← All 4 model loss curves together
    val_metrics_comparison.png   ← Accuracy & MCC over epochs
    test_metrics_comparison.png  ← Final accuracy, F1, MCC bar charts
```

**Key things to discuss from plots:**
- **Loss curves:** Are both train and val loss decreasing together? Any sign of overfitting (val loss increasing while train decreases)?
- **Confusion matrix:** Which stress stages are most confused? Adjacent stages (early vs. mid) are typically harder to separate.
- **MCC comparison:** Shows how well each model handles class imbalance in the original test set.

---

## 8. Live Demonstration Commands

> ⚠️ **Models are already trained.** Do NOT re-run training during the demo. Use the pre-trained checkpoints from `checkpoints/ProfessorDemo/`.

### Step 1: Activate Environment
```powershell
cd d:\crop-stress-grading-eo-1-main
.\venv\Scripts\activate
```

### Step 2: (If needed) Prepare Data
```powershell
python -m scripts.prepare_data
```
**What to explain while it runs:**
> "This processes the raw EO-1 hyperspectral CSV — dropping noisy bands, applying Savitzky-Golay smoothing, computing derivative spectroscopy, adding our two vegetation indices MLVI and H_VSI, normalising, then splitting into train, val, and test sets, and finally oversampling with SMOTE to handle class imbalance."

Expected output:
```
Using target column: 'Stage'
Total spectral columns found: 198
Dropped 67 spectral columns (>50% NAs)
Remaining spectral columns: 131
Dropped 55 rows with remaining NAs. Keeping 6933 rows.
Applying Savitzky-Golay filtering with window length 11
Computing Derivative Spectroscopy
Calculating MLVI and H_VSI
Saved processed data to data\processed\processed_data.csv
Data augmented with SMOTE
Saved splits to data\splits/
```

### Step 3: Generate Explainability Maps
```powershell
python -m scripts.explain --exp ProfessorDemo --model hybrid --method all
```
**What to explain:**
> "This loads the best saved Hybrid model checkpoint and runs three XAI techniques: Grad-CAM traces gradients back to the last convolutional layer to find which spectral footprint the CNN found most important. GradientShap computes SHAP values via the expected gradient path from a zero baseline. Attention weights extract the attention matrices from the Transformer layers. All outputs are saved as images."

Expected output location: `_results/explain/ProfessorDemo/hybrid/`

### Step 4: Generate Performance Plots
```powershell
python -m scripts.plot_results --exp ProfessorDemo
```
**What to explain:**
> "This reads the JSON metrics files saved during training and generates publication-quality comparison plots — loss curves, confusion matrices, and a final test set comparison across all four models."

Expected output: `_results/plots/ProfessorDemo/comparison/`

### Step 5: Walkthrough the Output Files
Open and discuss:
```
_results/
├── plots/ProfessorDemo/
│   ├── comparison/
│   │   ├── loss_comparison.png          ← side-by-side training dynamics
│   │   ├── val_metrics_comparison.png   ← MCC and accuracy over training
│   │   └── test_metrics_comparison.png  ← final model ranking
│   └── per_model/
│       ├── hybrid_loss_curve.png
│       └── hybrid_confusion_matrix.png
└── explain/ProfessorDemo/hybrid/
    ├── gradcam_mean.png             ← which bands CNN attended to
    ├── gradcam_heatmap.png          ← per-class Grad-CAM heatmap
    ├── gradientshap_mean.png        ← SHAP attribution curves
    ├── gradientshap_heatmap.png     ← per-class SHAP heatmap
    ├── top_bands.png                ← top 20 most important bands
    ├── attention_mean.png           ← Transformer attention curves
    └── attention_heatmap.png        ← per-class attention heatmap
```

---

## 9. Frequently Asked Questions (Professor Q&A)

**Q: Why not use traditional ML models like Random Forest or SVM?**  
A: Traditional ML works on flat feature vectors and cannot exploit the sequential, ordered nature of spectral bands. A 1D CNN captures local absorption features; a Transformer captures global dependencies across the full spectrum. Traditional models also struggle with high-dimensional correlated features (264 features with complex inter-band relationships).

**Q: Why 1D convolutions and not 2D?**  
A: The input is a single spectral sequence per observation — not an image. There is no spatial height dimension; the "width" is the band index. 1D convolutions slide along the spectral axis, capturing local spectral patterns.

**Q: Why treat the spectral sequence like a time-series even though it's spatial?**  
A: The ordering of bands by wavelength creates a meaningful 1D structure — adjacent bands in wavelength space are physically related (overlapping absorption features). Treating it as a sequence allows the model to learn "spectral locality" the same way a temporal model learns temporal locality.

**Q: How is SMOTE different from just duplicating samples?**  
A: SMOTE generates **new synthetic samples** by interpolating between existing minority-class samples in feature space: `new_sample = x_i + λ × (x_j - x_i)` where `x_i`, `x_j` are two minority samples and `λ ∈ [0,1]`. This creates diverse new data rather than exact copies, preventing the model from memorising specific samples.

**Q: Why use MCC as the primary early stopping metric instead of accuracy?**  
A: On an imbalanced test set, a model that predicts only the majority class can still achieve high accuracy. MCC is bounded in [-1, 1] and accounts for all four values of the confusion matrix (TP, TN, FP, FN). It's only high when the model performs well across all classes, making it more informative for multi-class imbalanced problems.

**Q: Why label smoothing?**  
A: Neural networks trained with hard one-hot labels become overconfident. A model that outputs probability 0.999 for one class cannot reliably distinguish between similar stress stages. Label smoothing forces the model to distribute a small probability mass (0.1/num_classes) to all incorrect classes, reducing overconfidence and improving calibration.

**Q: What does the confusion matrix tell us?**  
A: Each row is the true class, each column is the predicted class. Diagonal values are correct predictions. Off-diagonal values are misclassifications. High values near the diagonal (adjacent rows/columns) mean the model confuses adjacent stress stages — which is expected since early and mid-stress may have similar spectral signatures.

**Q: If the Hybrid model is best, why train the other three?**  
A: Scientific rigour — we can only claim the Hybrid is better by comparison. Each architecture represents a design decision (local features only vs. global features only vs. combined). The progression from CNN → ResNet → Transformer → Hybrid tells a coherent story about how each architectural innovation contributes to performance gain.

**Q: How do we know the XAI outputs are physically meaningful?**  
A: We look for high attributions near known spectral absorption features: chlorophyll at ~680nm (red), the red-edge at ~700–725nm, the NIR plateau at ~800–900nm, water absorption bands at ~970nm and ~1450nm, and SWIR stress-sensitive features at ~1649nm and ~2133nm. If the model's top bands cluster around these wavelengths, we have empirical evidence that the model learned physically real patterns rather than dataset artifacts.

**Q: What would you do differently if you had more time?**  
A: Pre-training on large unlabelled hyperspectral datasets (self-supervised), trying Vision Transformers with 2D spectral-spatial patches if image-level data were available, incorporating temporal sequences (multiple satellite passes over the same crop), and performing cross-crop generalisation testing.

---

*Guide prepared for: Crop Stress Grading using EO-1 Hyperspectral Data*  
*Models: Paper CNN · ResNet1D · Transformer · Hybrid*  
*XAI Methods: Grad-CAM · GradientShap · Attention Weights*
