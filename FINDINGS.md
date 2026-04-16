# FINDINGS

## Objective

Compare robustness and internal behavior of:

- CNN: ResNet-50
- Vision Transformer: ViT-B/16

under controlled corruptions: **blur, noise, rotation**.

## Quantitative Results (Robustness)

- **Blur**
  - Both models remain highly stable
  - Minimal accuracy drop
  - ⇒ Reliance on **coarse structural information**

- **Noise**
  - CNN: **sharp degradation**
  - ViT: **moderate degradation**
  - ⇒ CNN is **highly sensitive to high-frequency perturbations**

- **Rotation**
  - CNN: moderate degradation
  - ViT: **stronger degradation**
  - ⇒ ViT shows **weaker geometric invariance**

## CNN Behavior (ResNet-50)

**Representation:**

- Dense, continuous **spatial activations** (via Grad-CAM)

**Observed behavior:**

- **Blur**
  - Activations remain aligned with object
  - Slight smoothing, no mislocalization

- **Noise**
  - Activations become **fragmented**
  - Shift toward background / artifacts
  - Lose spatial coherence

**Failure mode:**

> Progressive breakdown of spatial localization due to instability in localized feature representations.

---

## Vision Transformer Behavior (ViT-B/16)

**Representation:**

- Sparse, patch-based **self-attention**

**Observed behavior:**

- **Noise**
  - Attention initially object-aligned
  - Becomes **scattered across discrete patches**
  - Includes non-object regions

- **Rotation**
  - Attention fails to follow object orientation
  - Becomes **spatially inconsistent**

**Failure mode:**

> Structured but **semantically misaligned attention**, rather than spatial diffusion.

---

## Comparative Insight

| Aspect              | CNN (ResNet-50)                 | ViT (ViT-B/16)          |
| ------------------- | ------------------------------- | ----------------------- |
| Representation      | Dense spatial activations       | Sparse attention        |
| Degradation pattern | Progressive                     | Structured but unstable |
| Noise robustness    | Poor                            | Better                  |
| Rotation robustness | Better                          | Poor                    |
| Failure mechanism   | Fragmentation + mislocalization | Mislocalization         |

---

## Core Conclusion

> CNNs fail by losing coherent spatial localization, while Vision Transformers fail by maintaining structured attention that becomes semantically misaligned.

---

## Interpretation

- CNNs rely on **localized discriminative features**
  → vulnerable when features are corrupted (noise)

- ViTs rely on **global patch relationships**
  → more stable under noise, but weaker under geometric transformations

Robustness differences are not only performance-based but they also arise from **fundamentally different internal failure mechanisms**:

- CNN → **spatial breakdown (fragmentation + misalignment)**
- ViT → **semantic breakdown (mislocalized attention)**

---

## Code

All implementation and experiments:
👉 https://github.com/M-DEV-1/vision-under-corruption
