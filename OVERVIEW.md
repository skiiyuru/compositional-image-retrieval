# Project Overview

## Goal

Build a system that, given a reference image and a set of positive (`+`) and negative (`−`) text constraints, retrieves images that preserve the reference's identity while satisfying the constraints. We extend the CLAY framework (Lim et al., 2026) which builds on GDE (Berasi et al., 2025), addressing CLAY's limitation of naïve embedding stacking when handling multiple conditions.

## The four threads

We split the work into four experimental threads. Each is independently runnable; later threads compose with earlier ones.

### Thread 1 — Zero-shot baseline

- **What:** Naïve Euclidean arithmetic on CLIP embeddings: `q = v_ref + Σt⁺ − Σt⁻`, then cosine similarity for retrieval.
- **Why:** Establishes the lower bound everyone must beat. Forces us to nail down the evaluation pipeline before specialising.
- **Reference:** Radford et al. 2021 (CLIP); also the brief's described baseline.
- **Where to start:** Load CLIP ViT-B/32, extract embeddings for the CelebA test split, implement Recall@K / Precision@K against the provided ground-truth JSON.

### Thread 2 — GDE-style geodesic composition

- **What:** Replace text-derived attribute directions with **image-derived** ones, computed on the unit hypersphere using the logarithm map. Add denoising via CLIP's image-to-text probabilities.
- **Why:** GDE's results show massive gains (Table 1 in their paper) when respecting the manifold geometry vs naïve Euclidean composition. Training-free.
- **Reference:** Berasi et al. 2025 (GDE), Sec. 3.
- **Where to start:** Implement `log_map` and `exp_map` functions; compute intrinsic mean of CelebA embeddings; derive per-attribute primitive directions.

### Thread 3 — LoRA on CLIP's text tower

- **What:** Inject low-rank adapters into the Q/K/V matrices of CLIP's text encoder. Freeze everything else. Train on a contrastive retrieval objective.
- **Why:** Specialises text representations for attribute-style phrasing without bloating parameters. Tests the training-based paradigm.
- **Reference:** Hu et al. 2022 (LoRA); Zhang et al. 2022 (NOAH) for the prompt-tuning context.
- **Where to start:** Adapt the LoRA implementation from the NLU project. Wrap CLIP's `nn.Linear` layers in `LoRALinear`.

### Thread 4 — CLAY sign-aware subspaces

- **What:** Build separate textual subspaces for positive (`P⁺`) and negative (`P⁻`) constraints using CLAY's manifold-aware tangent-space SVD. Two scoring variants:
  - **B2 (null-space):** project database embeddings _away_ from `P⁻`, then score against `P⁺`
  - **B1 (additive):** `score = sim(q, P⁺v) − λ·sim(q, P⁻v)`
- **Why:** Directly extends CLAY's mechanism to handle CLAY's own limitation: naïve stacking of multiple conditions. Training-free.
- **Reference:** Lim et al. 2026 (CLAY), Sec. 3.2.
- **Where to start:** Implement manifold-aware SVD on a set of text prompt embeddings; precompute `P⁺` and `P⁻` for each query.

## How the threads compose

Threads 2 and 4 share the **manifold machinery** (log/exp maps, intrinsic mean, tangent-space ops). They should be built on a common utilities module — agreed on early to avoid duplicated or incompatible implementations.

The full progression for the report:

```
T1: Euclidean arithmetic                     (baseline)
T2: GDE (image-derived, geodesic)            (+ manifold)
T2 + T4: GDE + sign-aware subspaces          (+ sign handling)
T2 + T4 + T3: + LoRA on text tower           (+ adaptation)
```

Each step adds one mechanism, so any performance gain can be attributed cleanly.

## Team split

| Person | Owns                      | Key skill                     |
| ------ | ------------------------- | ----------------------------- |
| A      | Thread 2 (GDE)            |
| Steve  | Thread 3 (LoRA)           |
| C      | Thread 4 (CLAY subspaces) | careful reading of CLAY paper |

Persons A and C must coordinate on the shared manifold utilities module from week 1.

## Phases

- **Week 1 (Steve):** Repo setup, evaluation harness, Thread 1 baseline.
- **Weeks 2–3 (parallel):** Each person works their thread.
- **Week 4 (integration):** Combine threads, run full comparison chain.
- **Final days (assembly):** Inline everything into the deliverable notebook, write narrative, run on Colab, freeze outputs, submit.

## Mandatory constraints (from the brief)

- Model: CLIP ViT-B/32 from HuggingFace
- Deliverable: single self-contained Jupyter notebook on Google Colab
- Implement everything from scratch (no PEFT, no third-party fusion libraries)
- Evaluate on the mandatory query list in `celeba_evaluation.json`
- Report Recall@K and Precision@K at K=1, 5, 10
