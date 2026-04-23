# 🎓 FINAL PRESENTATION GUIDE — Easy Version

> Read this top to bottom. It's your talk track. Say what's written here in your own words.

---

# STEP 1: THE PROBLEM (Start here)

**Say this:**

> "Crops get stressed from drought, heat, pests, or poor soil. By the time you can see the damage with your eyes, it's already too late — the yield is lost. We need a way to detect stress early, before it becomes visible."

**Then explain:**

- Farmers can't walk through thousands of acres checking each plant
- Satellites can scan huge areas from space
- The **EO-1 satellite** has a special sensor called **Hyperion** that doesn't just see 3 colors (RGB) — it sees **242 different wavelengths** of light (visible + infrared)
- Think of it like this: **a normal camera sees 3 colors, this satellite sees 242 colors** — it can detect things invisible to human eyes
- Different stress conditions change how plants reflect light at specific wavelengths
- For example:
  - **~680nm (red light):** chlorophyll absorption — tells you if the plant is photosynthesizing
  - **~1450nm (infrared):** water absorption — tells you if the plant is dehydrated
  - **~2133nm (shortwave infrared):** cell structure — tells you if the leaves are physically breaking down

---

# STEP 2: WHAT THE BASE PAPER DID

**Say this:**

> "We based our project on a research paper called MLVI-CNN. Let me explain what they did and where we improved."

**The base paper (MLVI-CNN) in simple terms:**

1. They took the satellite data (242 bands of light measurements)
2. They cleaned the noise using a **smoothing filter** (Savitzky-Golay — basically like blurring a photo slightly to remove grain)
3. They used a technique called **RFE** (Recursive Feature Elimination) to find the 10 most useful wavelengths out of 242
4. From those 10 bands, they created **2 custom formulas** (indices):
   - **MLVI** = (NIR − SWIR1) / SWIR2 → measures moisture and structure
   - **H_VSI** = (NIR − SWIR1) / (NIR + SWIR1 + SWIR2) → same idea but normalized
5. They fed **just these 2 numbers** into a **simple 2-layer CNN**
6. They got **83.40% accuracy** classifying 6 stress levels

**The problem with their approach:**

- Imagine you have a 242-page report about a patient, but you only read 2 sentences and make a diagnosis. That's what they did — **they threw away 240+ bands of useful data**
- Their CNN was too simple — just 2 layers. It can't learn complex patterns
- They had **no way to explain** why the model made a particular prediction

---

# STEP 3: WHAT WE IMPROVED (The 3 Big Things)

---

## 🔹 Improvement 1: We used ALL the data (2 features → 264 features)

**Say this:**

> "Instead of boiling down 242 bands into just 2 numbers, we kept all the spectral information."

**What we did:**
- Started with 242 bands → dropped the noisy ones (67 had too many missing values) → **131 clean bands**
- Applied the same smoothing filter as the base paper
- Added something extra: **derivative spectroscopy** — this calculates the *rate of change* between neighboring bands
  - Simple analogy: if the original data is like measuring temperature, the derivative is like measuring *how fast the temperature is changing*. Sharp changes tell you something important is happening at that wavelength
- We also still computed their MLVI and H_VSI (we didn't throw away their work)
- Final input: **131 smooth bands + 131 derivative bands + MLVI + H_VSI = 264 features**

**Why this matters (explain simply):**
> "With 2 features, the model is basically comparing 2 numbers. With 264 features, it sees the complete spectral fingerprint of the crop and can discover patterns on its own — patterns that even humans might miss."

---

## 🔹 Improvement 2: We tested 4 models, not just 1

**Say this:**

> "The base paper only used one simple model. We built 4 models, each one more powerful than the last, to see which works best."

### Model 1: Paper CNN (same as base paper, but with 264 features)
- **What it is:** 2 convolutional layers — same design as the base paper
- **Think of it like:** a magnifying glass that can only look at 3 bands at a time (nearby bands only)
- **Result: 85.38%** — already beats the base paper (83.40%) just because we gave it more data
- **Parameters: ~27K** (small model)

### Model 2: ResNet (deeper with shortcuts)
- **What it is:** 6 convolutional layers arranged in 3 blocks (2 layers per block)
- **The key idea — skip connections:** imagine you're passing a message through 6 people. By the time it reaches the last person, the message gets distorted. Skip connections let you pass a copy of the original message directly, so nothing gets lost
- In technical terms: solves the **vanishing gradient problem** — without skip connections, deep networks actually get worse, not better
- **Result: 86.63%**
- **Parameters: ~110K**

### Model 3: Transformer (looks at everything at once)
- **What it is:** Instead of sliding a small window across the bands (like CNN does), the Transformer looks at **all 264 bands simultaneously** and figures out which bands are related to each other
- **Think of it like:** CNN reads a book word by word. Transformer reads the entire page at once and connects ideas that are far apart
- **Self-attention** = each band asks "which other bands should I pay attention to?"
- 4 attention heads = 4 different ways of looking at the relationships, running in parallel
- **Result: 87.98%**
- **Parameters: ~4.5M** (big model)

### Model 4: Hybrid — OUR MODEL (CNN + Transformer combined)
- **What it is:** First, CNN extracts local features from nearby bands. Then, Transformer connects those features globally
- **Think of it like:** CNN is a summary writer — it reads each paragraph and writes a summary. Transformer is an editor — it reads all the summaries and connects the big ideas across the whole document
- **Skip connection** in the CNN stage preserves the original information too
- MaxPool cuts the sequence in half → Transformer works with 132 enriched tokens instead of 264 raw ones → faster and more focused
- **Result: 89.90%** ← best of all four
- **Parameters: ~2.3M** (half the size of pure Transformer, but better accuracy!)

**The punchline:**
> "Our Hybrid model beats the base paper by +6.5% accuracy. And it's smarter and more efficient than a pure Transformer — better accuracy with half the parameters."

---

## 🔹 Improvement 3: We made the model explainable (XAI)

**Say this:**

> "The base paper just reported accuracy. But if a farmer or scientist asks 'why did you classify this as stressed?' — they had no answer. We built explainability into our project."

**We used 3 different techniques:**

### Grad-CAM (works on CNN models)
- Looks at the last CNN layer and asks: "which spectral regions lit up the most when making the prediction?"
- Like a heatmap — bright areas = important wavelengths

### GradientSHAP (works on ALL models)
- Based on game theory (Shapley values)
- Imagine 264 players (bands) on a team. SHAP calculates how much each player contributed to the final score (prediction)
- Each band gets a score — higher score = more important for the prediction

### Attention Weights (built into Transformer/Hybrid)
- The Transformer already calculates attention scores — we just extract them
- Shows which bands the model "naturally looks at" — no extra work needed

**The key finding:**
> "All 3 techniques — which work completely differently — point to the same spectral bands as most important. And those bands match real physics: chlorophyll at ~680nm, water absorption at ~1450nm, and leaf structure at ~2133nm. This proves our model learned real crop science, not just random patterns in the data."

---

# STEP 4: THE RESULTS (Show the Numbers)

## Results Comparison Table

| Who | Model | Accuracy | MCC |
|-----|-------|----------|-----|
| Base paper | LDA | 77.40% | 0.528 |
| Base paper | SVM | 78.97% | 0.570 |
| Base paper | **1D CNN** | **83.40%** | **0.659** |
| **Us** | Paper CNN | 85.38% | 0.819 |
| **Us** | ResNet | 86.63% | 0.834 |
| **Us** | Transformer | 87.98% | 0.851 |
| **Us** | **Hybrid** | **89.90%** | **0.874** |

**What to highlight:**
- Our worst model (85.38%) is already better than their best (83.40%)
- Our best model (Hybrid, 89.90%) is **+6.5%** better
- MCC went from 0.659 to 0.874 — this means our model is much better at handling all 6 classes, not just the easy ones
- **MCC** is like a more honest version of accuracy — it doesn't get fooled by class imbalance

---

# STEP 5: SHOW THE PLOTS (Open these files and explain)

---

### 📊 Plot 1 — The Money Chart
**File:** `_results/plots/Full_Run/comparison/test_metrics_comparison.png`

> "This bar chart shows accuracy, F1, and MCC for all 4 models side by side. You can see the clear progression — each model is better than the previous one. The Hybrid (green bar) is the tallest everywhere."

---

### 📊 Plot 2 — Training Loss
**File:** `_results/plots/Full_Run/comparison/loss_comparison.png`

> "This shows how the error (loss) decreased during training. All models improve over time. The wavy pattern is from our learning rate scheduler — it periodically bumps the learning rate up to help the model explore better solutions. The important thing: train and validation loss are close together = no overfitting."

---

### 📊 Plot 3 — Accuracy Over Training
**File:** `_results/plots/Full_Run/comparison/val_metrics_comparison.png`

> "This tracks accuracy and MCC during training. The Hybrid reaches the highest point and stays there. We saved the model at the point where MCC was highest."

---

### 📊 Plot 4 — Confusion Matrix (Hybrid)
**File:** `_results/plots/Full_Run/per_model/hybrid_confusion_matrix.png`

> "This 6×6 grid shows what the model predicted vs what was actually true. The diagonal = correct answers. Most mistakes happen between adjacent stress levels — like confusing 'mild' with 'moderate' stress — which makes sense because they look very similar even spectrally."

---

### 📊 Plot 5 — Hybrid Loss Curve
**File:** `_results/plots/Full_Run/per_model/hybrid_loss_curve.png`

> "Train loss and validation loss track closely — no overfitting. The model learned well."

---

### 📊 Plot 6 — Top 20 Important Bands (XAI)
**File:** `_results/explain/Full_Run/hybrid/top_bands.png`

> "This is the most important XAI plot. It shows the 20 spectral bands our model relies on most. These match real science — chlorophyll, water, and leaf structure wavelengths. The model figured out the same things that crop scientists already know, completely on its own."

---

### 📊 Plot 7, 8, 9 — Heatmaps (Grad-CAM, SHAP, Attention)
**Files:**
- `_results/explain/Full_Run/hybrid/gradcam_mean.png`
- `_results/explain/Full_Run/hybrid/gradientshap_heatmap.png`
- `_results/explain/Full_Run/hybrid/attention_mean.png`

> "These three heatmaps come from three completely different XAI methods, but they all highlight the same spectral regions. This cross-confirmation is strong evidence that our model is trustworthy."

---

# STEP 6: WHAT WE USED FROM THE BASE PAPER vs. WHAT'S NEW

| Thing | From Base Paper? | What We Changed |
|-------|-----------------|-----------------|
| EO-1 Hyperion dataset | ✅ Same | — |
| Savitzky-Golay smoothing | ✅ Same | — |
| MLVI and H_VSI formulas | ✅ Same (we kept them) | We use them as 2 of 264 features, not the only 2 |
| Z-score normalization | ✅ Same concept | We use StandardScaler from sklearn |
| 70/15/15 data split | ✅ Same | — |
| 2-layer CNN architecture | ✅ Replicated as "Paper CNN" | We added 3 more architectures |
| Derivative spectroscopy | ❌ New | We added 131 derivative features |
| ResNet (skip connections) | ❌ New | 6 layers, 3 blocks |
| Transformer (attention) | ❌ New | Global band-to-band attention |
| Hybrid (CNN + Transformer) | ❌ New | Our main model |
| SMOTE for class imbalance | ❌ New | Creates synthetic minority samples |
| Label smoothing | ❌ New | Prevents overconfident predictions |
| AdamW optimizer | ❌ New | Better than plain Adam |
| Cosine annealing scheduler | ❌ New | Helps escape local minima |
| Gradient clipping | ❌ New | Stabilizes Transformer training |
| Grad-CAM explainability | ❌ New | Base paper had no XAI |
| GradientSHAP | ❌ New | Base paper had no XAI |
| Attention visualization | ❌ New | Base paper had no XAI |

---

# IF PROFESSOR ASKS HARD QUESTIONS

**"Why not just use Random Forest or SVM?"**
> Those treat the 264 features as an unordered bag of numbers. Our models treat them as an ordered sequence (like a signal), which lets them learn patterns between neighboring wavelengths.

**"Why is the Hybrid better than Transformer if Transformer is newer?"**
> Transformer alone has to figure out both local AND global patterns from scratch. Hybrid lets CNN handle local patterns first, then Transformer only needs to handle global connections. It's like giving someone a summary before asking them to analyze the full document.

**"How do you know you're not overfitting?"**
> Three checks: (1) train and val loss are close in the loss curves, (2) early stopping on MCC prevents overtraining, (3) we evaluate on a test set the model never sees during training.

**"What's SMOTE?"**
> Some stress classes have very few samples (Class 3 has only 34). SMOTE creates fake but realistic new samples by blending existing ones. Like mixing two similar data points to create a new one in between. We only do this on training data — test data stays real.

**"Why does accuracy change each time you run it?"**
> Neural networks start with random weights and use random mini-batches. Different starting points → slightly different results. The ranking always stays the same though: Hybrid > Transformer > ResNet > Paper CNN. That's why we show pre-generated results.

**"What could be improved?"**
> Use multiple satellite images over time (temporal sequences), test on different crops and satellites, and try ensemble methods (combining multiple models together).

---

# QUICK CHEAT SHEET (Stick this on your phone)

- Base paper: **2 features, 2-layer CNN, 83.40%**
- Us: **264 features, Hybrid (CNN+Transformer), 89.90%**
- Improvement: **+6.5% accuracy, +0.215 MCC**
- 3 improvements: **more features, better models, XAI explainability**
- Key XAI finding: **3 different techniques → same bands → real physics**
- MLVI = (NIR − SWIR1) / SWIR2
- H_VSI = (NIR − SWIR1) / (NIR + SWIR1 + SWIR2)
- 6 stress classes: Healthy → Extreme Stress
- Skip connection = shortcut that prevents gradient vanishing
- Attention = every band looks at every other band
