# Implementation notes: what is faithful to ConvNeXt V2, and what is not

Reference: Woo et al., *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked
Autoencoders*, CVPR 2023 — [arXiv:2301.00808](https://arxiv.org/abs/2301.00808).

Everything here is implemented from scratch in PyTorch. No `timm`, no pre-trained
weights, no reference implementation copied. This document states precisely where
the implementation matches the paper and where it does not, because the gap
matters for interpreting the results.

## Architecture — faithful

| Component | Paper | This repo | Where |
|---|---|---|---|
| Depthwise 7×7 conv, inverted bottleneck | ✅ | ✅ | [`Block`](../src/model.py) |
| LayerNorm, channels-first *and* channels-last | ✅ | ✅ | [`LayerNorm`](../src/model.py) |
| Global Response Normalization (the V1→V2 change) | ✅ | ✅ | [`GRN`](../src/model.py) |
| LayerScale removed (redundant with GRN) | ✅ | ✅ | — |
| Four-stage hierarchical encoder, patchify stem | ✅ | ✅ | [`ConvNeXtV2Encoder`](../src/model.py) |
| Depths / dims | Atto `[2,2,6,2]` / `[40,80,160,320]` | same | [`CONFIGS`](../src/model.py) |

GRN is the component that makes ConvNeXt V2 what it is: it restores inter-channel
competition and prevents the feature collapse V1 shows under masked pre-training.
It is implemented exactly as described — L2 aggregation over the spatial dims,
divisive normalization across channels, learnable `gamma`/`beta`, residual.

### Naming correction

The original course report calls this encoder **"ConvNeXt V2 Nano"**. That is
wrong: `depths=[2,2,6,2]`, `dims=[40,80,160,320]` is the **Atto** configuration
(~3.7M parameters). Nano is `depths=[2,2,8,2]`, `dims=[80,160,320,640]` (~15.6M).
The network itself was never changed — only the label was incorrect. This repo
uses the correct name throughout; the archived PDF report still says "Nano".

## Pre-training framework — deliberately simplified

The pre-training here is **not** FCMAE as published. It is a masked
*denoising* autoencoder. The differences are not cosmetic:

| Component | Paper (FCMAE) | This repo | Consequence |
|---|---|---|---|
| Masking granularity | 32×32 patches | **per-pixel i.i.d.** | A 7×7 depthwise conv can interpolate a missing pixel from its immediate neighbours. The pretext task becomes close to trivial. |
| Encoder convolutions | sparse submanifold | **dense** | Information leaks across the mask — the exact failure FCMAE was designed to prevent. |
| Reconstruction loss | masked regions only | **whole image** | ~40% of the loss is an identity mapping. He et al. ablate this and show it degrades representation quality. |
| Decoder | one ConvNeXt block, dim 512 | **4-stage ConvTranspose stack** | Heavy and symmetric, the opposite of the paper's asymmetric design. |
| Pre-training length | 800–1600 epochs | **15 and 100 epochs** | Orders of magnitude shorter. |

These simplifications were in the submitted project and are **kept unchanged** here,
so that what gets measured is the pipeline as actually handed in rather than a
different method wearing its name.

They are the most likely explanation for the shape of the
[measured results](../README.md#results-in-full): under a linear probe the
pre-trained encoder is ~10 points *worse* than a frozen randomly-initialised one,
yet the same weights are a better fine-tuning initialisation than random in the
scarce-label band. That is the signature of a pretext task that teaches useful
low-level filters but no semantics — which is what per-pixel masking with dense
convolutions and a whole-image loss should be expected to teach. At 60% per-pixel
masking the cell remains plainly visible, so the task is largely solvable by local
interpolation.

Implementing true FCMAE is therefore the change that would turn a negative result
about *this* pretext task into a statement about FCMAE itself.

Implementing true FCMAE requires sparse convolutions (`spconv` or MinkowskiEngine)
and is the single most valuable extension of this work.

## Deviations from the original submission

Three things were changed relative to the notebook that was handed in. All are
corrections, none affect the method:

1. **Seeding.** The notebook seeded only the data split, so weight init, dropout,
   mask sampling and shuffling were uncontrolled — the reported 94.8% was one
   unrepeatable draw. [`set_seed`](../src/engine.py) now seeds Python, NumPy and
   Torch, and every result is reported as mean ± SD over three seeds.
2. **Augmentation.** The report describes `RandomResizedCrop` during pre-training
   and "richer augmentation" during fine-tuning. The submitted code applied
   **none** — just resize, to-tensor, normalize. This repo matches the *code*, not
   the report, and says so rather than silently adding augmentation.
3. **Masked-input visualization.** The notebook plotted `denormalize(x) * mask`,
   painting masked pixels black. The model actually sees zero in *normalized*
   space, which is the ImageNet mean colour. The figure here shows the real input.

## Evaluation protocol

- The dataset is split once, with a fixed generator seed, into four disjoint
  subsets: 50% SSL pre-training (unlabelled), 25% supervised training, 10%
  validation, 15% test — 2,500 / 1,250 / 500 / 750 images.
- The split is identical across every condition and every run seed, so all
  comparisons are paired on the same 750-image test set. Only training
  stochasticity varies with the seed.
- Label-budget subsets are drawn **stratified** by class: at 1% (13 images over
  5 classes) uniform sampling can drop a class entirely.
- Model selection is by validation loss with early stopping (patience 5); the
  best weights are restored before the single test-set evaluation.

## Known limitations of the protocol

- **One data split.** The test set is fixed rather than cross-validated, so the
  ±SD reported here captures training stochasticity, not split variance.
- **No patient/slide grouping.** The Kaggle dataset carries no slide or subject
  identifiers, so near-duplicate images from the same source slide could fall on
  both sides of the split. This cannot be ruled out with the metadata available.
- **Linear probes use lr 1e-3** rather than the 5e-4 used for full fine-tuning,
  since only a 5-way head is being trained. Both values are stated in
  [`experiments.py`](../src/experiments.py).
