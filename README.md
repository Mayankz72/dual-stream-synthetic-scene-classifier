# Dual-Stream Synthetic Scene Classifier

A dual-stream CNN (RGB + edge features) for binary classification of synthetic, ray-traced 3D scenes — trained **entirely from scratch**, no pre-trained weights.

Built for the **[IITH Deep Learning 2026 Hackathon](https://www.kaggle.com/competitions/iith-deep-learning-2026-hackathon)** (CS5480: Deep Learning, IIT Hyderabad).

**Public leaderboard: 0.8092** — up from a 0.7534 single-stream baseline.

<p align="center">
  <img src="assets/architecture.png" alt="Dual-stream architecture diagram" width="820" />
</p>

## Problem

Scenes contain cubes, spheres, and cylinders varying in material (matte/specular), size, color, and position, with no canonical orientation. Each image gets a binary label whose generating rule is **latent** — never disclosed, and inferred purely from the training data. Pre-trained weights are disallowed.

| Split | Images |
|---|---|
| Train — class 0 (`train/0/`) | 9,000 |
| Train — class 1 (`train/1/`) | 9,000 |
| Test — `test/` (labels hidden) | 5,010 |

A model can hit ~100% validation accuracy by latching onto any attribute correlated with the label in training — color, material reflectance, lighting — without learning the true rule. That train/leaderboard gap is the core challenge; see [Results](#results).

## Approach

**Two streams, so no single shortcut dominates:**
- **RGB** — raw pixels, heavy augmentation (`ColorJitter`, `RandomGrayscale`, `GaussianBlur`, `RandomErasing`) + MixUp.
- **Edge** — [`EdgeTransform`](src/scene_classifier/edge_transform.py): bilateral filter (suppresses specular highlights) → Canny + Sobel + Laplacian, stacked into a 3-channel, material-invariant geometry map.

Both streams share **ShapeNet** ([`model.py`](src/scene_classifier/model.py)): 4 stages of `ResBlock → SEBlock → strided conv`, global average pooling, dropout MLP head. Trained with AdamW, cosine LR, and SWA.

**At inference:**
1. **BN test-time adaptation** — reset and re-estimate BatchNorm stats on the unlabeled test set.
2. **4-way deterministic TTA** — average logits over identity / H-flip / V-flip / both-flip.
3. **Two-round self-training** — Round 1 scores the test set; images where *both* streams predict `sigmoid < 0.15` become negative-only pseudo-labels; Round 2 retrains both streams from scratch on the expanded set.
4. **Ensemble** — `0.5 * rgb_logit + 0.5 * edge_logit`, thresholded at 0.

## Results

<p align="center">
  <img src="assets/ablation_chart.png" alt="Leaderboard score progression across iterative improvements" width="700" />
</p>

| Approach | Key additions | Public LB |
|---|---|---|
| A1 | Baseline RGB CNN | 0.7534 |
| A2 | + SE blocks + stronger color augmentation | ≈0.764 |
| A3 | + Edge stream + dual-stream ensemble | ≈0.769 |
| A4 | + SWA + BN test-time adaptation | ≈0.787 |
| **Final** | + Pseudo-labeling + 4-way TTA | **0.8092** |

Both streams reach ≈100% validation accuracy but only 0.8092 on the leaderboard — a ≈19-point gap indicating the models partly rely on spurious, training-distribution-specific correlations rather than the true rule. The edge stream and self-training loop narrow this gap but don't close it. Full analysis in the [report](reports/team_48_report.pdf).

## Repository Structure

```
src/scene_classifier/    Installable package (the refactored pipeline)
├── model.py               ResBlock, SEBlock, ShapeNet backbone
├── edge_transform.py      EdgeTransform (bilateral filter + Canny/Sobel/Laplacian)
├── data.py                Dataset classes + path loading
├── augment.py             Train/eval transforms, MixUp
├── train.py               train_one_model (SWA, cosine LR) + adapt_bn
├── inference.py           predict_with_tta, pseudo-label selection
└── pipeline.py            End-to-end orchestration + CLI entry point
tests/                    Model/transform smoke tests (pytest)
notebooks/                Original Kaggle submission notebook (unmodified)
reports/                  Full written report
submissions/              Final submission.csv
assets/                   README diagrams
```

`src/scene_classifier` is a from-scratch refactor of `notebooks/team_48_original_submission.ipynb` into a modular, tested package — same logic, the notebook stays untouched as the historical record of what actually ran on Kaggle.

## Usage

```bash
pip install -r requirements.txt
pip install -e .
```

Expected dataset layout:

```
dataset_root/
├── train/{0,1}/*.png
└── test/*.png
```

```bash
python -m scene_classifier.pipeline --data-dir /path/to/dataset_root --output submission.csv
pytest tests/   # no dataset needed — shape/contract checks only
```

## Key Design Decisions

| Decision | Why |
|---|---|
| No pre-trained weights | Competition rule; ImageNet priors don't transfer well to this synthetic domain anyway |
| Edge stream | Material-invariant geometry signal; suppresses specular artifacts |
| SWA + BN test-time adaptation | Flatter minima + re-calibrated BN stats on test distribution — ≈1.6pp combined gain |
| Negative-only pseudo-labels | A false positive would inflate the positive rate and hurt accuracy directly; only dual-model-agreed, high-confidence negatives are used |
| 4-way TTA, not 8-way with rotations | Rotations gave edge maps ambiguous spatial relationships; flips alone were simpler and stronger |
| Equal-weight (0.5/0.5) ensemble | RGB and edge streams make largely uncorrelated errors; no held-out set was reserved to tune weights |

## Team

CS5480: Deep Learning, IIT Hyderabad — Team 48:

- Aayush Ranjan — RGB model pipeline, augmentation, MixUp
- Ankit Kumar Sinha — Ensemble & inference pipeline, TTA, BN adaptation, pseudo-labeling
- Mayank Mishra — ShapeNet architecture, edge feature pipeline, dataset utilities

## License

[MIT](LICENSE)
