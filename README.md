# Dynamic and Hybrid Conditioning for Compositional Image Retrieval

![cover](image.png)

Team: Stephen Nganga, Anas Tansim and Zishan Ahmed

Given a reference face and a set of signed attribute constraints (e.g. `-Smiling, +Eyeglasses, +Wearing_Hat`), retrieve faces that keep the reference's identity while satisfying the constraints. All work lives in [index.ipynb](index.ipynb).

- **Data:** CelebA (test split as the retrieval database, train split for LoRA training)
- **Model:** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`), image embeddings frozen
- **Benchmark:** 13 queries from `celeba_evaluation.json`, from single-attribute edits to three-attribute compositions
- **Metrics:** Recall@K and Precision@K at K = 1, 5, 10, macro-averaged over queries

## Methods

| # | Method | Idea | Training |
| --- | --- | --- | --- |
| 1 | **Baseline** | Naïve Euclidean arithmetic: `q = v_ref + Σt⁺ − Σt⁻`, then cosine similarity. | None |
| 2 | **LoRA** | Rank-4 adapters on the Q/K/V projections of CLIP's text encoder (147k params, 0.10% of the model). Trained with InfoNCE plus an L2 penalty that keeps adapted attribute embeddings close to the frozen ones. | Yes |
| 3 | **GDE** | Training-free geodesic composition on the unit hypersphere. Attribute directions come from images (intrinsic mean of images with vs. without the attribute), and the edit is applied in tangent space at the global intrinsic mean, then mapped back with the exponential map. | None |
| 4 | **LoRA + GDE** | Same geodesic composition as GDE, but the attribute directions come from the LoRA-adapted text embeddings. | Uses LoRA weights |

**LoRA training data:** for each of the 40 CelebA attributes we pick reference images that have the attribute. Positives also have it and differ by at most 2 of the other 39 attributes. Hard negatives lack it and differ by at most 5. We built two sets, with 50 and 100 references per attribute, split them 80/20 into train and validation, and trained with Adam (lr 1e-4) and early stopping (patience 3).

**GDE edit strength:** we swept α ∈ {0.05, 0.1, 0.2, 0.35, 0.5}, and α = 0.5 was best for both GDE and LoRA + GDE.

## Results

Macro-averaged over the 13 benchmark queries (%):

| Method | R@1 | R@5 | R@10 | P@1 | P@5 | P@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 2.81 | 8.52 | 12.65 | 2.81 | 2.05 | 1.69 |
| LoRA (50 ex/attr) | 4.49 | 13.35 | 20.29 | 4.49 | 3.36 | 2.86 |
| LoRA (100 ex/attr) | **4.82** | 13.80 | **21.18** | **4.82** | 3.48 | 3.03 |
| GDE (α = 0.5) | 3.33 | 10.86 | 16.23 | 3.33 | 2.64 | 2.21 |
| LoRA-100 + GDE (α = 0.5) | 4.78 | **14.29** | 21.03 | 4.78 | **3.76** | **3.18** |

### Key findings

1. **LoRA gave the biggest improvement.** Adapting only the text encoder fixes CLIP's weak attribute encodings: `+Male` goes from 2.6% to 21.9% R@10, and `-Smiling, +Eyeglasses, +Wearing_Hat` from 13.9% to 44.3%.
2. **Geometry alone helps.** GDE lifts R@10 from 12.65% to 16.23% with no training, and it is strongest on `+Male` and `-Young`.
3. **LoRA and GDE complement each other on compositional queries.** LoRA + GDE gets the best result on `-Smiling, +Eyeglasses, +Wearing_Hat` (46.8% R@10), but it does worse than plain LoRA on simple single-attribute queries.
4. **Negative-only queries are still hard.** `-Male, -Mustache` stays below 8% R@10 for every method, because LoRA only ever trains on `ref + attribute`.

### Limitations and future work

- Add negative (`ref − attribute`) queries to the LoRA training data.
- Replace the single global α with a learned α per attribute or per query, or use flat arithmetic for single-attribute queries and geodesic composition for multi-attribute ones.
- Train on more data. The 100-example model was still improving when early stopping kicked in.
- Evaluate on other datasets (OVAD, AwA2) to test generalisation.

## Running the notebook

The expensive steps (embedding CelebA, building the training examples, LoRA training, the α sweeps and the full evaluations) are commented out in the notebook. Their outputs are downloaded from the Hugging Face dataset [`skiiyuru/conditional-image-retrieval`](https://huggingface.co/datasets/skiiyuru/conditional-image-retrieval) into `src/`, so the notebook runs end-to-end quickly. To reproduce from scratch, uncomment those cells.

CelebA itself must be in `data/celeba/` in the layout `torchvision.datasets.CelebA` expects.

### Environment

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync                        # installs pinned deps; torch picks your CUDA build via torch-backend = "auto"
source .venv/bin/activate      # optional, uv run does this for you
```

### Code quality

```bash
uv run pre-commit install      # ruff lint + format on every commit (.py and .ipynb)
uv run ruff check . --fix
uv run ruff format .
```

## References

- Radford et al., 2021. *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
- Hu et al., 2022. *LoRA: Low-Rank Adaptation of Large Language Models*
- Berasi et al., 2025. *Not Only Text: Exploring Compositionality of Visual Representations in Vision-Language Models* (GDE)
