# 🎯 CROP STRESS GRADING — FINAL PRESENTATION PITCH

> **Share this with your team. Read it tonight. You'll be ready tomorrow.**

---

## PART 1: WHAT IS OUR PROJECT?

**Title:** Explainable Deep Learning for Crop Stress Grading using EO-1 Hyperspectral Satellite Data

**One-liner to tell the professor:**
> "We built a system that looks at satellite images of farmland and automatically tells you how stressed the crops are — from healthy to extreme stress — using deep learning. And we can also explain *why* the model made each decision."

**In simple terms:**
- Satellites take pictures of farms, but not normal photos — they capture light in **242 different wavelengths** (like having 242 different color channels instead of just Red, Green, Blue)
- Each wavelength tells something different about the plant's health
- We feed this data into deep learning models that learn to classify crop stress into **6 levels** (Healthy → Extreme Stress)
- We also added **explainability** — so we can show *which* wavelengths the model looked at, proving it learned real plant science

---

## PART 2: WHAT IS CROP STRESS?

Crops get stressed by drought, disease, pests, or nutrient deficiency. When stressed, their internal chemistry changes:
- **Chlorophyll breaks down** → less light absorption at ~680nm
- **Water content drops** → changes reflectance at ~1450nm and ~1649nm
- **Cell structure degrades** → changes at ~854nm (NIR region)

These changes are **invisible to the naked eye** in early stages, but a hyperspectral satellite can detect them because it sees wavelengths humans can't.

**Why it matters:** If you catch stress early, farmers can intervene (irrigate, spray, fertilize). If you miss it, entire harvests are lost. India alone loses ~15-25% of crop yield to undetected stress annually.

**The 6 Stress Stages in our dataset:**

| Stage | Meaning |
|-------|---------|
| 0 | Healthy |
| 1 | Early Stress |
| 2 | Mild Stress |
| 3 | Moderate Stress |
| 4 | Severe Stress |
| 5 | Extreme Stress |

---

## PART 3: THE BASE PAPER — WHAT THEY DID

**Paper:** *"MLVI-CNN: A Hyperspectral Stress Detection Framework Using Machine Learning-Optimized Indices and Deep Learning for Precision Agriculture"*
**Authors:** Poornima S & A. Shirly Edward (SRM Institute)
**Published:** Frontiers in Plant Science, September 2025

### Their Approach (5 steps):
1. Took **EO-1 Hyperion satellite data** (242 spectral bands)
2. Applied **Savitzky-Golay filtering** to reduce noise
3. Used **RFE (Recursive Feature Elimination)** to pick top 10 important bands
4. Created 2 new vegetation indices from those bands:
   - **MLVI** = (NIR − SWIR1) / SWIR2
   - **H_VSI** = (NIR − SWIR1) / (NIR + SWIR1 + SWIR2)
5. Fed **only these 2 numbers** into a simple **2-layer CNN** → classified 6 stress levels

### Their Results:

| Model | Accuracy | MCC |
|-------|----------|-----|
| LDA | 77.40% | 0.528 |
| SVM | 78.97% | 0.570 |
| **1D CNN (best)** | **83.40%** | **0.659** |

### Their Limitations (THIS IS WHAT WE IMPROVED):
1. **Only 2 input features** — threw away 260+ spectral bands worth of information
2. **Simple 2-layer CNN** — too shallow to learn complex patterns
3. **No explainability** — can't prove the model learned real plant physics
4. **No comparison** with modern architectures (no ResNet, no Transformer)
5. **Low MCC (0.659)** — struggles with minority stress classes

---

## PART 4: OUR PROPOSED SOLUTION — WHAT WE IMPROVED

### 🔹 Improvement 1: More Features (2 → 264)

**What they did:** Compressed everything into just MLVI and H_VSI = **2 features**

**What we did:**
- Kept **all 131 usable bands** (dropped 67 noisy ones)
- Applied **Savitzky-Golay smoothing** (same as base paper)
- Added **1st derivative spectroscopy** — highlights where reflectance changes sharply (absorption edges)
- **Still kept MLVI + H_VSI** (didn't throw away their contribution)

```
131 smoothed bands + 131 derivative bands + MLVI + H_VSI = 264 features
```

**Why this matters (simple explanation):**
> "Imagine diagnosing a patient. The base paper checked only blood pressure and heart rate (2 numbers). We run a full body scan — blood work, X-ray, ECG, everything (264 measurements). More data = better diagnosis."

---

### 🔹 Improvement 2: Better Model Architectures (4 models instead of 1)

We built **4 models**, each more powerful than the last:

#### Model 1: Paper CNN (Replicating the base paper)
- Same 2-layer CNN architecture, but now with **264 features** instead of 2
- Result: **85.38% accuracy** — already beats the base paper (83.40%)!
- This proves: **more features alone = better results**

#### Model 2: ResNet (Deeper with Skip Connections)
- 6 convolutional layers with **skip connections** (shortcuts that help gradients flow)
- Result: **86.63% accuracy**
- This proves: **deeper networks = better feature learning**

#### Model 3: Transformer (Global Attention)
- Every band can "look at" every other band simultaneously
- Learns long-range relationships (e.g., how chlorophyll at 680nm relates to water at 1450nm)
- Result: **87.98% accuracy**
- This proves: **global attention = captures relationships CNNs miss**

#### Model 4: Hybrid — OUR BEST MODEL (CNN + Transformer)
- CNN first extracts local patterns → feeds enriched features to Transformer
- Transformer then captures global relationships between those features
- **Best of both worlds** with half the parameters of pure Transformer
- Result: **89.90% accuracy, MCC 0.874**

**The story to tell the professor:**
> "We started by replicating the base paper's CNN but with all 264 features — that alone beat them. Then we progressively added architectural innovations: depth (ResNet), global attention (Transformer), and finally combined both (Hybrid). Each step improved results, proving both more features AND better architectures matter."

---

### 🔹 Improvement 3: Explainability (XAI) — Base Paper Had NONE

We used **3 different XAI techniques** to prove our model learned real physics:

| Method | What it does (simple) | Available on |
|--------|----------------------|--------------|
| **Grad-CAM** | Highlights which bands the CNN focused on | CNN, ResNet, Hybrid |
| **GradientSHAP** | Fairly scores each band's contribution (game theory) | All 4 models |
| **Attention Weights** | Shows what the Transformer naturally focused on | Transformer, Hybrid |

**The killer argument:**
> "Three completely different methods — gradient-based, game-theory-based, and attention-based — all point to the **same spectral bands**. These bands match known plant science: chlorophyll at ~680nm, red-edge at ~700nm, water at ~1450nm, SWIR stress bands at ~1649nm. This proves our model learned real physics, not random noise."

---

## PART 5: RESULTS — BASE PAPER vs OURS

| Model | Accuracy | F1 Score | MCC | Source |
|-------|----------|----------|-----|--------|
| LDA | 77.40% | — | 0.528 | Base paper |
| SVM | 78.97% | — | 0.570 | Base paper |
| 1D CNN | 83.40% | 82.95% | 0.659 | Base paper |
| **Paper CNN (ours)** | **85.38%** | **85.41%** | **0.819** | Ours |
| **ResNet (ours)** | **86.63%** | **86.62%** | **0.834** | Ours |
| **Transformer (ours)** | **87.98%** | **87.96%** | **0.851** | Ours |
| **Hybrid (ours)** | **89.90%** | **89.92%** | **0.874** | Ours |

### Key Numbers to Remember:
- **+6.5% accuracy** improvement (83.40% → 89.90%)
- **+0.215 MCC** improvement (0.659 → 0.874) — huge for handling minority classes
- Even our **simplest model** (Paper CNN) beats the base paper

---

## PART 6: PLOTS & HEATMAPS — WHAT TO SHOW AND SAY

> All plots are pre-generated. Just open the image and explain using the script below.

---

### 📊 Plot 1: Test Metrics Comparison Bar Chart
**📁 File:** `d:\crop-stress-grading-eo-1-main\_results\plots\Full_Run\comparison\test_metrics_comparison.png`

**What it shows:** Side-by-side bars comparing Accuracy, F1, and MCC for all 4 models.

**Say this:**
> "This chart compares all four models on the test set. You can see the clear progression — Paper CNN is our baseline, ResNet adds depth (+1.25%), Transformer adds global attention (+1.35%), and Hybrid combines both for the best result (+1.92%). MCC also improves consistently, which is important because MCC is more reliable than accuracy for imbalanced data."

---

### 📊 Plot 2: Loss Comparison (All Models)
**📁 File:** `d:\crop-stress-grading-eo-1-main\_results\plots\Full_Run\comparison\loss_comparison.png`

**What it shows:** Training and validation loss curves for all 4 models over training epochs.

**Say this:**
> "All models converge properly. The Hybrid converges faster and reaches lower loss. The periodic waves you see are from Cosine Annealing — the learning rate resets periodically to help the model escape local minima. The small gap between training and validation loss means we're NOT overfitting."

---

### 📊 Plot 3: Validation Metrics Over Epochs
**📁 File:** `d:\crop-stress-grading-eo-1-main\_results\plots\Full_Run\comparison\val_metrics_comparison.png`

**What it shows:** How accuracy and MCC evolved during training for each model.

**Say this:**
> "The Hybrid reaches higher MCC earlier and maintains it. We use MCC as our early stopping criterion — we save the best model weights when MCC peaks, not accuracy, because MCC better reflects true multi-class performance."

---

### 📊 Plot 4: Hybrid Confusion Matrix
**📁 File:** `d:\crop-stress-grading-eo-1-main\_results\plots\Full_Run\per_model\hybrid_confusion_matrix.png`

**What it shows:** 6×6 grid — rows = true class, columns = predicted class. Diagonal = correct.

**Say this:**
> "This is our best model's confusion matrix. The strong diagonal means most predictions are correct. Most errors happen between adjacent stress stages — like confusing mild with moderate stress — which makes biological sense because neighboring stages have very similar spectral signatures. Even Class 3 with only 34 test samples is mostly correctly classified, thanks to SMOTE balancing."

---

### 📊 Plot 5: Hybrid Loss Curve
**📁 File:** `d:\crop-stress-grading-eo-1-main\_results\plots\Full_Run\per_model\hybrid_loss_curve.png`

**What it shows:** Train vs validation loss for the Hybrid model specifically.

**Say this:**
> "Train and validation loss track closely — confirming no overfitting. The periodic bumps are cosine annealing warm restarts. Early stopping triggered when MCC didn't improve for 15 consecutive epochs."

---

### 🔬 Plot 6: Top 20 Most Important Bands (XAI)
**📁 File:** `d:\crop-stress-grading-eo-1-main\_results\explain\Full_Run\hybrid\top_bands.png`

**Say this:**
> "This is one of the most important charts. GradientSHAP identified the 20 bands our model relies on most. They cluster around known plant-science wavelengths — chlorophyll at ~680nm, red-edge at ~700nm, SWIR moisture bands. The base paper manually selected bands using RFE. Our model independently discovered the same regions through learning."

---

### 🔬 Plot 7: Grad-CAM Heatmap
**📁 File:** `d:\crop-stress-grading-eo-1-main\_results\explain\Full_Run\hybrid\gradcam_mean.png`

**Say this:**
> "Grad-CAM shows which spectral regions the CNN part of our Hybrid focused on. Different stress stages activate different regions — proving the model learned class-specific spectral patterns, not one generic pattern."

---

### 🔬 Plot 8: GradientSHAP Heatmap
**📁 File:** `d:\crop-stress-grading-eo-1-main\_results\explain\Full_Run\hybrid\gradientshap_heatmap.png`

**Say this:**
> "This is a class-by-band attribution matrix. Each row is a stress class, each column is a spectral band. Brighter = more important. Healthy vegetation activates NIR bands, stressed crops activate SWIR bands — exactly what plant science predicts."

---

### 🔬 Plot 9: Attention Weights
**📁 File:** `d:\crop-stress-grading-eo-1-main\_results\explain\Full_Run\hybrid\attention_mean.png`

**Say this:**
> "These are the Transformer's own attention weights — what it naturally focused on, no extra computation needed. The peaks align with the same regions found by Grad-CAM and SHAP — three independent confirmations."

---

### 🔬 Plot 10 & 11 (Bonus — show if time permits):
- **Grad-CAM class heatmap:** `_results\explain\Full_Run\hybrid\gradcam_heatmap.png`
- **GradientSHAP mean curves:** `_results\explain\Full_Run\hybrid\gradientshap_mean.png`
- **Attention heatmap:** `_results\explain\Full_Run\hybrid\attention_heatmap.png`

---

## PART 7: LIVE DEMO COMMANDS

> ⚠️ Models are already trained. You do NOT need to retrain. Just run the commands below.

### Setup (run once):
```powershell
cd d:\crop-stress-grading-eo-1-main
.\venv\Scripts\activate
```

### Command 1: Data Preprocessing
```powershell
python -m scripts.prepare_data
```
**While it runs, say:**
> "This processes raw satellite data — drops noisy bands, applies Savitzky-Golay smoothing, computes derivatives, adds MLVI and H_VSI indices, normalizes everything, splits into train/val/test, and applies SMOTE to balance classes."

### Command 2: Train All Models (ONLY if professor asks — takes time!)
```powershell
python -m scripts.train --model all --exp LiveDemo
```
**Say:** "This trains all 4 models sequentially. Each uses AdamW optimizer, cosine annealing, label smoothing, and early stopping on MCC."

### Command 3: Generate XAI Explanations
```powershell
python -m scripts.explain --exp Full_Run --model hybrid --method all
```
**Say:** "This loads the trained Hybrid model and runs all three XAI methods — Grad-CAM, GradientSHAP, and Attention weight extraction."

### Command 4: Generate Comparison Plots
```powershell
python -m scripts.plot_results --exp Full_Run
```
**Say:** "This reads the saved training metrics and generates all comparison charts and confusion matrices."

### Quick Single-Model Commands (if professor wants to see just one):
```powershell
# Train only Hybrid:
python -m scripts.train --model hybrid --exp QuickTest

# Explain only Hybrid:
python -m scripts.explain --exp Full_Run --model hybrid --method gradientshap

# Plot only:
python -m scripts.plot_results --exp Full_Run
```

---

## PART 8: Q&A CHEAT SHEET

**Q: What's the main difference from the base paper?**
> They used 2 features. We use 264. They used one simple CNN. We tested 4 architectures and added explainability.

**Q: Why not just use more features with the same CNN?**
> We did — that's our Paper CNN (85.38%). But a 2-layer CNN with kernel size 3 only sees 2 neighboring bands at a time. For 264 features, you need deeper or attention-based models.

**Q: Why is Hybrid better than pure Transformer?**
> The CNN pre-processes local features first, so the Transformer only learns global relationships between enriched tokens. It's also half the parameters (2.3M vs 4.5M).

**Q: How do you handle class imbalance?**
> Two ways: (1) SMOTE generates synthetic minority samples in training, (2) class-weighted loss function penalizes majority-class errors less.

**Q: Why MCC and not just accuracy?**
> Accuracy can be high even if the model ignores minority classes. MCC considers all parts of the confusion matrix and is only high when ALL classes are predicted well.

**Q: How do you prove the model learned real science?**
> Three different XAI methods all point to the same bands — and those bands match known chlorophyll, water, and cellulose absorption wavelengths from plant biology textbooks.

**Q: What are the limitations?**
> Single time-point data (no temporal sequences), EO-1 specific (not tested on other satellites), Class 3 has very few samples (34).

**Q: Future work?**
> Multi-temporal analysis, cross-sensor transfer learning, ensemble methods, and self-supervised pre-training on unlabelled hyperspectral data.

---

## PART 9: GLOSSARY — KEY TERMS EXPLAINED SIMPLY

### 📗 Data & Preprocessing Terms

| Term | Simple Explanation |
|------|-------------------|
| **EO-1 Hyperion** | A NASA satellite with a sensor that captures 242 wavelengths of light (not just RGB). Retired in 2017, but its data is still used for research. |
| **Hyperspectral Data** | Like a regular photo but with 200+ "color channels" instead of 3. Each channel captures a specific wavelength of light. |
| **Spectral Band** | One wavelength channel. Band X680 = reflectance at 680 nanometers. |
| **Savitzky-Golay Filter** | A smoothing technique that fits a polynomial curve over a sliding window. Removes sensor noise while preserving real peaks and valleys in the data. |
| **Derivative Spectroscopy** | Computing the rate of change of reflectance across wavelengths. Highlights where the signal changes sharply — these are absorption edges from plant chemicals. |
| **MLVI** | Multi-band Landsat Vegetation Index = (NIR − SWIR1) / SWIR2. A formula using 3 specific bands to detect moisture/canopy stress. Created by the base paper. |
| **H_VSI** | Hyperspectral Vegetation Stress Index = (NIR − SWIR1) / (NIR + SWIR1 + SWIR2). Normalized version of MLVI — more robust to lighting differences. |
| **Z-score Normalization** | Scales each feature to mean=0, standard deviation=1. Ensures all features are on the same scale so no single feature dominates the model. |
| **SMOTE** | **Synthetic Minority Over-sampling Technique.** If Class 3 has only 100 samples but Class 0 has 2000, SMOTE creates new synthetic Class 3 samples by interpolating between existing ones: `new = sample_A + random × (sample_B − sample_A)`. Applied ONLY to training data. |
| **Stratified Split** | Dividing data into train/val/test so each split has the same proportion of each class. Prevents all samples of a rare class ending up in one split. |
| **RFE** | **Recursive Feature Elimination.** Removes the least important feature, retrains, repeats. The base paper used this to select 10 bands. We skip this and keep all bands. |

### 📘 Model Architecture Terms

| Term | Simple Explanation |
|------|-------------------|
| **1D CNN** | Convolutional Neural Network that slides a small filter along a 1D sequence (our spectral bands). Learns local patterns — like "what happens between adjacent wavelengths." |
| **Kernel Size (k=3)** | The width of the sliding filter. k=3 means each filter sees 3 adjacent bands at a time. |
| **MaxPool** | Shrinks the data by keeping only the max value in each window. Reduces size and makes the model focus on strongest signals. |
| **Skip Connection** | A shortcut that adds the original input directly to the output: `output = F(x) + x`. Solves the vanishing gradient problem in deep networks. Used in ResNet and Hybrid. |
| **ResNet (Residual Network)** | A deep CNN that uses skip connections. Can go many layers deep without losing gradient signal. |
| **Transformer** | A model where every input element (band) can "attend to" every other element simultaneously. Captures long-range relationships that CNNs miss. |
| **Self-Attention** | The mechanism inside Transformers: `Attention = softmax(QK^T / √d) × V`. Every band asks "which other bands are relevant to me?" and gets a weighted combination of all bands. |
| **Multi-Head Attention** | Running attention 4 times in parallel with different weights. Each "head" learns a different type of relationship between bands. |
| **Positional Encoding** | Since Transformers process all bands in parallel (no order), we add sine/cosine signals to tell the model "this is band 5, this is band 125." |
| **Hybrid Model** | Our best model. CNN extracts local features first → feeds them to a Transformer that captures global relationships. Best of both worlds. |
| **Batch Normalization (BN)** | Normalizes each layer's output to mean=0, std=1 during training. Stabilizes and speeds up training. |
| **Dropout** | Randomly turns off neurons during training (e.g., 30% dropout). Forces the network to not rely on any single neuron — reduces overfitting. |

### 📙 Training Terms

| Term | Simple Explanation |
|------|-------------------|
| **AdamW** | An optimizer (adjusts model weights). Adam = adaptive learning rates per parameter. The "W" adds proper weight decay (L2 regularization) to prevent large weights. |
| **Learning Rate (lr)** | How big of a step the model takes when updating weights. Too high = unstable. Too low = slow. Ours starts at 0.001. |
| **Cosine Annealing** | The learning rate follows a cosine curve — starts high, drops smoothly to near zero, then **restarts**. The restart helps escape local minima. |
| **Label Smoothing** | Instead of training with hard labels (1 for correct, 0 for wrong), we use soft labels (0.9 for correct, 0.017 for each wrong class). Prevents overconfidence. |
| **Class Weights** | In the loss function, rare classes get higher weight so mistakes on rare classes are penalized more. Helps with class imbalance. |
| **Early Stopping** | If the model doesn't improve (MCC) for 15 straight epochs, training stops. Prevents overfitting and wasted compute. |
| **Gradient Clipping** | Limits gradient values to max 1.0. Prevents exploding gradients that can crash Transformer training. |
| **Cross-Entropy Loss** | The standard loss function for classification. Measures how different the model's predicted probabilities are from the true labels. |
| **Epoch** | One complete pass through the entire training dataset. |
| **Overfitting** | When the model memorizes the training data but performs poorly on new data. We prevent this with dropout, weight decay, early stopping, and label smoothing. |

### 📕 Evaluation Metrics

| Term | Simple Explanation |
|------|-------------------|
| **Accuracy** | % of predictions that are correct. Simple but misleading with imbalanced classes. |
| **F1 Score** | Harmonic mean of Precision and Recall. Balances "of all predictions for class X, how many were correct?" with "of all true class X samples, how many did we find?" |
| **MCC** | **Matthews Correlation Coefficient.** Ranges from -1 (all wrong) to +1 (all right). Uses ALL four confusion matrix values (TP, TN, FP, FN). Only way to get high MCC is to predict ALL classes well. Our best: 0.874. |
| **Confusion Matrix** | A grid showing predicted vs actual classes. Diagonal = correct. Off-diagonal = errors. |

### 📓 Explainability (XAI) Terms

| Term | Simple Explanation |
|------|-------------------|
| **XAI** | **Explainable AI.** Techniques that show *why* a model made a decision. Critical for trust in agriculture/medicine. |
| **Grad-CAM** | **Gradient-weighted Class Activation Mapping.** Traces gradients back to the last CNN layer to see which spectral regions contributed most to the prediction. Like a "heatmap of importance." |
| **GradientSHAP** | Combines SHAP theory (game theory) with gradients. Fairly distributes "credit" for each prediction among all 264 input bands. Shows which bands pushed the model toward or away from each class. |
| **SHAP Values** | Based on Shapley values from game theory. Imagine 264 "players" (bands) cooperating to make a prediction. SHAP fairly distributes the total prediction score among all players based on their contribution. |
| **Attention Weights** | Built into Transformers. Shows how much each band "looked at" other bands when making a decision. Extracted directly — no extra computation needed. |
| **Captum** | Facebook/Meta's PyTorch library for model interpretability. We use it to compute GradientSHAP values. |

### 📒 Domain-Specific Terms

| Term | Simple Explanation |
|------|-------------------|
| **NIR** | **Near Infrared** (~800-900nm). Healthy plants strongly reflect NIR light due to cell structure. Stressed plants reflect less. |
| **SWIR** | **Short-Wave Infrared** (~1400-2500nm). Sensitive to water content and plant structure. Stress increases SWIR reflectance. |
| **Red Edge** | The sharp increase in reflectance between ~680nm (red) and ~750nm (NIR). Shifts when plants are stressed — a key diagnostic feature. |
| **Chlorophyll Absorption** | Chlorophyll absorbs light at ~680nm. Less chlorophyll (stressed plant) = less absorption = higher reflectance at 680nm. |
| **Vegetation Index** | A mathematical formula combining specific wavelengths to highlight plant health. NDVI is the most famous: (NIR−Red)/(NIR+Red). |

---

## PART 10: PRESENTATION FLOW — RECOMMENDED ORDER

| # | What to Say/Show | Time |
|---|-----------------|------|
| 1 | **"Our project is..."** — one-liner from Part 1 | 1 min |
| 2 | **What is Crop Stress** — why it matters, why satellites, why DL | 2 min |
| 3 | **Base Paper** — what they did, their results (83.40%), their limitations | 3 min |
| 4 | **Improvement 1** — 264 features vs 2 (the "full body scan" analogy) | 2 min |
| 5 | **Improvement 2** — 4 architectures, the progression story | 4 min |
| 6 | **Improvement 3** — XAI, why it matters, 3 methods | 2 min |
| 7 | **Results table** — base paper vs ours, highlight +6.5% and +0.215 MCC | 1 min |
| 8 | **Show plots:** test_metrics → loss → confusion matrix | 3 min |
| 9 | **Show XAI:** top_bands → gradcam → shap heatmap → attention | 3 min |
| 10 | **Convergence argument** — "3 methods, same bands, matches real science" | 1 min |
| 11 | **Conclusion + Future Work** | 1 min |
| 12 | **Q&A** | 5 min |
| | **TOTAL** | **~28 min** |

---

## QUICK FILE REFERENCE

### Comparison Plots
```
d:\crop-stress-grading-eo-1-main\_results\plots\Full_Run\comparison\
├── test_metrics_comparison.png   ← THE key chart (show first)
├── loss_comparison.png           ← training dynamics
└── val_metrics_comparison.png    ← accuracy/MCC over epochs
```

### Per-Model Plots
```
d:\crop-stress-grading-eo-1-main\_results\plots\Full_Run\per_model\
├── hybrid_confusion_matrix.png   ← show this one
├── hybrid_loss_curve.png         ← show this one
├── paper_cnn_confusion_matrix.png
├── resnet_confusion_matrix.png
└── transformer_confusion_matrix.png
```

### XAI Heatmaps (ALL IMPORTANT)
```
d:\crop-stress-grading-eo-1-main\_results\explain\Full_Run\hybrid\
├── top_bands.png                 ← MOST IMPORTANT (show first)
├── gradcam_mean.png              ← CNN saliency per class
├── gradcam_heatmap.png           ← class × band Grad-CAM
├── gradientshap_mean.png         ← SHAP attribution curves
├── gradientshap_heatmap.png      ← class × band SHAP
├── attention_mean.png            ← Transformer attention focus
└── attention_heatmap.png         ← class × band attention
```

---

> **Good luck tomorrow! 🚀 Read Parts 1-6 for the story, Part 9 for terminology, and Part 8 for live demo commands.**
