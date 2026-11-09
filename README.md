# Do Explanations Survive the Camera?
### Cross-site attribution consistency in diabetic retinopathy screening

**Author:** Sadaf Zehra · pilot study
**Notebook:** `fundus_cross_site_xai_KAGGLE.ipynb` (runs on Kaggle Notebooks, free GPU)

---

## 1. Motivation

A diabetic retinopathy (DR) classifier trained at one site is often deployed at another site with a different fundus camera. Accuracy may hold up, but does the model still look at the **same evidence** (the lesions) when the camera changes?

Recent work measures attribution consistency across *demographic groups*. This project asks the same question across **acquisition sites / devices**.

## 2. What the notebook does

Task: **referable DR** (ICDR grade ≥ 2) vs. non-referable.

1. **Train two models on one source site (APTOS 2019)**
   - **ResNet-50 + Grad-CAM**: a standard classifier with a post-hoc explanation.
   - **ResNet-50 Evidence-Map model**: an inherently interpretable model. A 1×1 conv produces a class-evidence map and the logit is the spatial mean of that map, so the map is an exact additive decomposition of the decision.
2. **External testing** on IDRiD and (optionally) Messidor-2, reporting AUC with bootstrap confidence intervals.
3. **Where does each model look at every site?**
   - attribution energy *outside* the retina (a shortcut indicator)
   - spread (entropy) and centre of mass of the attribution
   - distribution shift of these statistics vs. the source site (Wasserstein distance, KS test)
4. **Lesion grounding** on IDRiD pixel-level masks: energy-in-lesion, lift over lesion area, pointing game, AUPRC.
5. **Paired site-swap test**: the *same* image is re-coloured to look like each target site (FOV-restricted histogram matching), then the two explanations are compared with Spearman rank correlation, top-10 % IoU and prediction flips.
6. **Deletion test (faithfulness)**: blurring the most-attributed patches should hurt the prediction more than blurring random patches.

## 3. Datasets

| Site | Role | How the notebook finds it |
|---|---|---|
| **APTOS 2019** | Source site (train / val / test split) | `train.csv` next to a `train_images` folder |
| **IDRiD** | External site + pixel-level lesion masks (MA, HE, EX, SE; OD = optic disc) | folders *B. Disease Grading* and *A. Segmentation* |
| **Messidor-2** *(optional)* | External site | `messidor_data.csv` + image files in your attached inputs |

Notes:
- APTOS 2019: accept the competition rules once at https://www.kaggle.com/competitions/aptos2019-blindness-detection/rules
- IDRiD is downloaded automatically with `kagglehub`. If that copy disappears, attach any IDRiD dataset with *Add Input* or put its name in `IDRID_SLUGS`.
- Messidor-2 images cannot be redistributed. If you have them, upload them as your own **private** dataset and attach it together with *MESSIDOR-2 DR Grades* (Google Brain). Without Messidor-2 the study runs on APTOS + IDRiD.
- Please follow each dataset's terms and cite them. Do not make Messidor-2 images public.

## 4. How to run (Kaggle)

1. **One-time:** Kaggle → Settings → *Phone Verification* (required for GPU and internet).
2. Right panel → *Session options*: **Accelerator = GPU T4 x2** (one GPU is used) and **Internet = On** (needed once to download ImageNet ResNet-50 weights).
3. Add the datasets (see section 3).
4. **First run with `DEMO_MODE = True`** (synthetic data, about 10 minutes) to check the whole pipeline. Then set `DEMO_MODE = False`.
5. For the full run use **Save Version → Save & Run All (Commit)**. It runs in the background (up to 12 h) and keeps everything in `/kaggle/working/fundus_xai/outputs/`.

Kaggle gives about 30 GPU hours per week; one full run needs well under that.

Set `FORCE_RETRAIN = True` to ignore saved checkpoints and train again.

## 5. Key settings (`CFG`)

| Setting | Value | Meaning |
|---|---|---|
| `img_size` | 384 | Cached image size (lesions are small; 224 loses them) |
| `batch_size` | 16 | |
| `epochs` / `patience` | 12 / 4 | Early stopping on validation AUC |
| `lr`, `weight_decay` | 3e-4, 1e-4 | Cosine schedule with 1 warm-up epoch |
| `dilate_layer4` | True | Stride-16 feature maps (24×24 at 384 px) for sharper explanations |
| `evidence_l1` | 1e-4 | Sparsity on the evidence maps |
| `val_frac` / `test_frac` | 0.15 / 0.15 | Stratified split of the source site |
| `n_bootstrap` | 500 | Bootstrap resamples for AUC CIs |
| `n_swap`, `swap_ref_pool` | 200, 40 | Images and reference images per site for the swap test |
| `n_deletion` | 150 | Images for the deletion test |

Training uses mixed precision, a fixed seed (42) and light augmentation (flips, rotation, colour jitter).

## 6. Outputs

Everything is written to `/kaggle/working/fundus_xai/outputs/`:

**Tables**
- `performance.csv`: AUC (with CI), sensitivity, specificity per model and site
- `attribution_per_image.csv`, `attribution_shift.csv`: attribution statistics and their shift vs. source
- `lesion_grounding_per_image.csv`, `lesion_grounding_summary.csv`
- `site_swap_per_image.csv`, `site_swap_summary.csv`
- `deletion_summary.csv`
- `summary.md`: auto-generated results summary

**Models and logs**
- `<model>_best.pt`: best checkpoint per model
- `<model>_history.csv`: training history

**Figures (generated by the notebook, `fig1` to `fig6`)**
1. Performance per site
2. Attribution statistics per site
3. Accuracy vs. explanation stability
4. Site-swap example
5. Lesion grounding examples
6. Deletion curves

## 7. How to read the results

- **Accuracy.** A drop on external sites is expected. The interesting case is when AUC stays similar but the explanation changes.
- **Energy outside the retina.** Values clearly above ~0.02 suggest the model uses the image border, camera text or black background. In DEMO mode the source site has a planted white marker, so this should be high for at least one model.
- **Site-swap stability.** Spearman close to 1 means the explanation survives the camera change. A low value together with a small AUC change is the key finding: *accuracy holds but evidence moves*.
- **Lesion grounding.** `lift` > 1 means more attribution on lesions than their area predicts. `pointing_game` is the share of images where the hottest pixel falls on (or near) a lesion.
- **Deletion test.** If blurring top-attributed patches does not hurt more than blurring random ones, the explanation is not faithful.

## 8. Limitations

- Site-swap uses colour/contrast matching only. Real cameras also differ in resolution, blur, field of view and compression.
- DR grading conventions differ between datasets, so external AUC mixes device shift with label shift.
- One training seed. For a paper, repeat with 3 seeds and report mean ± std.
- Grad-CAM and the evidence map differ in what their values mean; compare trends, not absolute values.

## 9. Next steps

1. Add a third explanation method (e.g. Integrated Gradients) and a ViT backbone.
2. Add resolution / blur / JPEG differences to the site-swap test.
3. Train with a consistency loss that keeps explanations stable across simulated sites, and test whether it improves external AUC.

## 10. Requirements

Python 3, PyTorch + torchvision, NumPy, pandas, SciPy, scikit-learn, Pillow, matplotlib, tqdm, `kagglehub`. All are pre-installed on Kaggle.
