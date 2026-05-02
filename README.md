# Compositional Image Retrieval

Read `OVERVIEW.md` for a detailed brief regarding the project

## Directory structure

```
repo/
├── OVERVIEW.md              # the task overview doc
├── README.md                # this file
├── data/                    # gitignored; CelebA images, eval JSON
├── src/                     # development code (one file per thread)
│   ├── manifold.py          # SHARED: log/exp maps, intrinsic mean (A + C)
│   ├── eval.py              # SHARED: Recall@K, Precision@K, GT loading
│   ├── baseline.py          # Thread 1
│   ├── gde.py               # Thread 2 (Person A)
│   ├── lora.py              # Thread 3 (Person B)
│   └── clay.py              # Thread 4 (Person C)
├── notebooks/
│   ├── steve.ipynb          # one notebook per person, never merged
│   ├── a.ipynb
│   ├── b.ipynb
│   └── final.ipynb          # assembled at the end
└── results/                 # saved metrics, checkpoints
```

## Branching strategy

- `main` — stable, working code only. Merges happen via PR.
- One branch per person (`thread:gde`, `thread:lora`, `thread:clay`) — long-lived, used during development.
- Shared files (`manifold.py`, `eval.py`) → coordinate before editing; small focused PRs.

## Final notebook assembly

Because the deliverable must be self-contained:

1. Develop in `src/*.py` (clean diffs, testable)
2. Near the end, create `notebooks/final.ipynb`
3. **Inline** the code from `src/*.py` into notebook cells, organised by thread
4. Add markdown narrative around each section
5. Connect Colab to the repo; run end-to-end; freeze outputs

The "inlining" step is mostly mechanical copy-paste, but it forces a final cleanup of dead code — treat it as a feature, not a chore.

## Coordination points

- **Week 1:** A and C agree on the `manifold.py` API (function signatures for `log_map`, `exp_map`, `intrinsic_mean`).
- **End of week 1:** Thread 1 baseline + eval harness merged to `main`. Everyone uses the same evaluation code from this point on.
- **Week 4:** Combined comparison chain runs end-to-end. Lock the codebase 3 days before submission.

## Environment Setup

**Prerequisites:** [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

```bash
# Clone and enter the repo
git clone ...

# Create the venv and install all dependencies
uv sync

# Activate (optional — uv run does this automatically)
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml` + `uv.lock` and installs the exact pinned versions. PyTorch is resolved automatically for your local CUDA version via `torch-backend = "auto"` — no manual index selection needed.

### Running code

```bash
# Run a script
uv run python src/baseline.py

# Start Jupyter
uv run jupyter notebook
```

Using `uv run` ensures the correct venv is always used, even without activating it.

### Adding a dependency

```bash
uv add <package>          # adds to pyproject.toml and updates uv.lock
git add pyproject.toml uv.lock
git commit -m "add <package>"
```

Never edit `uv.lock` by hand — it's auto-generated.

### Troubleshooting

- **`uv sync` fails on torch:** Make sure your NVIDIA driver is up to date. Run `nvidia-smi` to check your driver version.
- **Different torch version than expected:** This is normal — `torch-backend = "auto"` picks the best wheel for your GPU/driver combo.
- **Colab:** The final notebook pins stable torch directly in a `!pip install` cell. The uv setup is for local dev only.

## Code Quality

Install the pre-commit hooks once after cloning:

```bash
uv run pre-commit install
```

Ruff runs automatically on every commit (lint + format, for `.py` and `.ipynb` files). To run manually:

```bash
uv run ruff check . --fix
uv run ruff format .
```

To update hook versions:

```bash
uv run pre-commit autoupdate
git add .pre-commit-config.yaml && git commit -m "bump pre-commit hooks"
```
