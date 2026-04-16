# FINDINGS

### ResNet-50 | ViT-B/16 | Caltech-101 | Blur · Noise · Rotation

---

## Experimental Constraints (Read First)

Before interpreting any finding, note the following conditions that bound all conclusions:

- Backbones are **frozen**. Only the classification head is trained.
- All robustness behavior reflects **ImageNet-pretrained feature spaces**, not
  end-to-end architectural properties.
- Caltech-101 is a **small, clean, canonical-viewpoint dataset**. Results may
  not generalize to in-the-wild distributions.
- All accuracy values are **single-run point estimates** with no confidence
  intervals. Treat trends, not exact numbers.
- Rotation padding introduces **zero-filled boundary patches** that neither
  model was trained to handle.

---

## Quantitative Results

| Model     | Clean Baseline | Blur (Sev 5) | Noise (Sev 5) | Rotation (Sev 5) |
| --------- | -------------- | ------------ | ------------- | ---------------- |
| ResNet-50 | ~95.05%        | ~88%         | ~65%          | ~87%             |
| ViT-B/16  | ~95.85%        | ~88%         | ~84%          | ~80%             |

### Per-Corruption Summary

**Blur**

- Both models: minimal degradation across all severity levels
- Neither model relies heavily on high-frequency detail for classification
- Grad-CAM and attention maps remain broadly stable
- **Finding: Blur is not a meaningful differentiator between these two architectures
  under the frozen transfer learning setup**

**Noise**

- ResNet-50: sharp accuracy drop, significant at Sev 3+, severe at Sev 5
- ViT-B/16: gradual decline, maintains substantially higher accuracy at Sev 5
- **Finding: CNN is clearly more sensitive to additive Gaussian noise**
- Caveat: ViT's noise robustness likely inherits from large-scale pretraining
  procedures, not architecture alone

**Rotation**

- ResNet-50: moderate degradation, relatively stable compared to ViT
- ViT-B/16: steeper drop, especially at Sev 4–5 (±40–50°)
- **Finding: ViT is more sensitive to geometric transformation than CNN**
- Caveat: ViT fixed positional embeddings are a plausible structural reason,
  but padding artifacts are a confound

---

## Interpretability Findings

### ResNet-50 — Grad-CAM

#### Blur (Images 1, 4 in paper)

| Severity | Observed Behavior                                |
| -------- | ------------------------------------------------ |
| Clean    | Dense activation centered on object (lens, body) |
| Sev 1–2  | Activation broadens slightly, remains on object  |
| Sev 3–4  | Spatial drift begins — peak shifts within object |
| Sev 5    | Diffuse but still broadly object-aligned         |

- **Claim:** Broadly stable with measurable spatial drift at higher severities
- Clean localization quality varies significantly by image class

#### Noise (Images 2, 5 in paper)

| Severity | Observed Behavior                                 |
| -------- | ------------------------------------------------- |
| Clean    | Concentrated on discriminative object region      |
| Sev 1–2  | Activation begins spreading, slight fragmentation |
| Sev 3    | Clear fragmentation, multiple competing regions   |
| Sev 4–5  | Severe mislocalization — background dominates     |

- **This is the strongest and cleanest finding**
- CNN activation goes from coherent → fragmented → background-dominant
- Consistent across camera and other object categories
- **Mechanism:** Localized convolutional feature detectors are disrupted by
  high-frequency pixel perturbations, causing the gradient signal used by
  Grad-CAM to respond to noise artifacts rather than semantic features

#### Rotation (Image 3 in paper)

| Severity | Observed Behavior                                        |
| -------- | -------------------------------------------------------- |
| Sev 1    | Attention pivots — re-localizes to different object part |
| Sev 2    | Partial recovery on original region                      |
| Sev 3–5  | Progressive fragmentation across object parts            |

- More nuanced than described in paper
- CNN doesn't simply degrade — it **re-localizes to different object sub-regions**
  as orientation changes
- Suggests CNN uses orientation-specific local feature detectors

#### Class-Specific Anomaly: Faces Category (Images 7–9)

- ResNet-50 focuses on **chest/torso region**, not the face, even on clean input
- Activation remains on torso through blur severities
- Under noise, attention eventually migrates toward chin/beard area
- **Interpretation:** Model uses clothing texture or torso appearance as a
  classification shortcut for this class
- **Implication:** Grad-CAM "degradation" in Faces is compounding a
  pre-existing localization failure, not a pure corruption effect

---

### ViT-B/16 — Attention Maps

#### Critical Baseline Issue

> For the camera and faces categories, ViT attention is **poorly localized
> on clean images**. High-attention patches appear in background regions
> before any corruption is applied. This means corruption-induced
> mislocalization cannot be cleanly separated from pre-existing
> mislocalization.

#### Blur (Images 4, 10 in paper)

| Severity | Observed Behavior                                       |
| -------- | ------------------------------------------------------- |
| Clean    | Attention scattered — mostly off-object (corners, bg)   |
| Sev 1–5  | Pattern largely unchanged, slight increase in diffusion |

- Camera category: dominant attention spot is near bottom-left, not on lens
- Faces category: background structural elements (lab equipment) attract attention
- **Finding: ViT does not meaningfully degrade under blur because it was not
  well-localized to begin with for these classes**
- Cannot attribute stability to robustness; it may reflect a flat localization
  baseline

#### Noise (Images 5, 11 in paper)

| Severity | Observed Behavior                                        |
| -------- | -------------------------------------------------------- |
| Clean    | Sparse, few hotspots — partially off-object              |
| Sev 1–2  | Hotspots shift slightly, some migrate to noise artifacts |
| Sev 3–4  | Increased scatter, background patches gain attention     |
| Sev 5    | Heavily fragmented, no semantic anchor                   |

- Unlike CNN, attention fragmentation is **patch-discrete** — scattered dots,
  not continuous diffuse regions
- At Sev 5 (faces category), attention migrates to forehead and background
  equipment, losing the torso focus seen at lower severities
- **Structural difference from CNN confirmed:** ViT fragments into discrete
  patches rather than a continuous diffuse map

#### Rotation (Images 6, 12 in paper)

| Severity | Observed Behavior                                    |
| -------- | ---------------------------------------------------- |
| Sev 1    | Minor attention shift, partially on object           |
| Sev 2–3  | Attention moves toward image edges/corners           |
| Sev 4–5  | Dominant attention near rotated boundary, off-object |

- Boundary-region attention at high severities is consistent with
  **positional embedding misalignment** — rotated content no longer matches
  the spatial context the embeddings encode
- Zero-padding at borders may additionally attract attention as out-of-
  distribution patches
- **Two confounds exist:** architectural sensitivity to rotation + padding artifacts
- Paper only identifies the former; the latter is unaddressed

#### Class-Specific Anomaly: Faces Category (Images 10–12)

- Faces: ViT attention anchors on background lab equipment structures,
  not the face
- Rotation causes attention to drift further into background and image borders
- **Finding:** ViT classification of faces in this dataset likely relies on
  background context rather than facial features — a severe shortcut
- Mirrors the CNN torso-shortcut but the ViT version is arguably worse:
  background context rather than at least part of the subject

---

## Failure Mode Comparison

| Dimension              | ResNet-50 (CNN)                                  | ViT-B/16                                            |
| ---------------------- | ------------------------------------------------ | --------------------------------------------------- |
| Blur                   | Broad stability, mild drift                      | Flat (poor baseline localization)                   |
| Noise                  | Fragmentation → mislocalization                  | Patch scatter, sparse structure preserved           |
| Rotation               | Re-localizes to sub-regions, then fragments      | Boundary/corner attraction, positional misalignment |
| Failure character      | **Spatial diffusion + fragmentation**            | **Patch mislocalization**                           |
| Clean baseline quality | Varies by class (good for camera, bad for faces) | Consistently worse than CNN across tested classes   |
| Corruption sensitivity | Noise >> Rotation > Blur                         | Rotation >> Noise > Blur                            |

### The Core Distinction

> CNNs tend to degrade through spatial diffusion and fragmentation of
> activation under corruption. ViTs retain patch-discrete structure but
> exhibit mislocalization both before and after corruption, making it
> difficult to isolate the corruption contribution. The distinction holds
> directionally but is not absolute.

---

## Recommendations for Future Work

- **Unfreeze backbones** and compare end-to-end trained models under the
  same corruption suite
- **Add corruption augmentation** during training and evaluate delta in
  robustness
- **Select evaluation classes** where both models show clean-image
  localization quality, to isolate corruption effects from shortcut effects
- **Fix rotation padding** — use reflect or replicate padding instead of
  zero-fill to eliminate boundary artifacts
- **Multiple runs with confidence intervals** — report mean ± std over at
  least 3 seeds
- **Expand to ImageNet-C** for comparison with established benchmarks
  (Hendrycks & Dietterich 2019)
- **Investigate the shortcut classes** (Faces, potentially others) as a
  separate study — the background-context and clothing-texture shortcuts
  are interesting findings in their own right
