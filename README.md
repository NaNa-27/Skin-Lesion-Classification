# Handling Class Imbalance in Seven-Class Skin Lesion Classification on HAM10000 under Lesion-Level Evaluation

Course project (DSR301m). This repository holds the code, logs and result files.

The study asks one question: **under a lesion-level split, do conditional StyleGAN2-ADA images for the rarest classes (AKIEC, DF, VASC) improve a ResNet-50 classifier beyond simple imbalance baselines?**

## Main findings

- **Earlier protocols leak lesions between train and test:** HAM10000 has several images per lesion. In the split used by Bissoto et al. (2021), 568 of 1,923 HAM10000 test lesions (29.5%) also have an image in the train pool. A random image-level 8:2 split as in Su et al. (2024) gives 725 of 1,880 test lesions (38.6%) and 791 of 2,003 test images (39.5%).
- **This study splits by lesion:** 70/15/15 of lesions, stratified by diagnosis, seed 42. No lesion is shared between train, validation and test.
- **GAN images did not help:** A3 (oversampling + GAN images) did not beat A2 (oversampling alone):

| Arm | Test macro-F1 | Test balanced acc. | Test accuracy |
|---|---|---|---|
| A0: cross-entropy | 0.697 ± 0.018 | 0.667 ± 0.020 | 0.829 |
| A1: weighted CE | 0.693 ± 0.012 | 0.717 ± 0.011 | 0.777 |
| A2: oversampling | **0.723 ± 0.016** | 0.715 ± 0.010 | 0.805 |
| A3: A2 + GAN images | 0.708 ± 0.023 | 0.705 ± 0.021 | 0.794 |

Mean ± SD over 3 training seeds, test set of 1,502 images. Paired A3 - A2 difference with 95% lesion-level bootstrap CI: macro-F1 -0.016 [-0.042, +0.013], balanced accuracy -0.010 [-0.042, +0.023]. MEL recall dropped (-0.086 [-0.147, -0.031]). This is a post-hoc observation, not a main result.

The synthetic images were almost perfectly separable from real ones (real/fake detector AUC 0.98-1.00).

## Repository structure

```
Skin-Lesion-Classification/
├── dataset/                                  HAM10000 CSV files only
│   ├── HAM10000_metadata.csv
│   ├── ISIC2018_Task3_Training_LesionGroupings.csv
│   └── ISIC2018_Task3_Training_GroundTruth/   ground truth CSV + LICENSE.txt, ATTRIBUTION.txt
├── data-understanding/
│   ├── 01_ham10000_lesion_split.ipynb        lesion-level split v2 + leakage checks
│   ├── check_bissoto2021_lesion_overlap.py   lesion overlap in the Bissoto et al. (2021) split
│   ├── check_su_split_overlap.py             lesion overlap in a Su-style image-level split
│   ├── output-d.u/                           split manifest v2, checks, summaries, script outputs
│   └── archive/output-d.u_v1/                first split version (superseded)
├── gan/
│   ├── 01-prepare-gan-data.ipynb             256x256 centre crops (real_sq256) + gan_train.zip (train split only)
│   ├── outputs-notebook-01/                  prep_info.json, gan_train_manifest.csv, class_map.json
│   ├── 02-train-gan-5th.ipynb                StyleGAN2-ADA training, notebook of the 5th (last) run
│   ├── training_logs/
│   │   ├── progress_run5.json                history of all 5 runs (dates, hours, kimg, augment p, GPU)
│   │   ├── logs/                             run01-run05: log, stats.jsonl, training_options.json
│   │   └── grids/                            fakes grids at the end of each run (half resolution) + resize_image.ipynb
│   ├── 03-generate-and-qc-1480.ipynb         snapshot comparison + synthetic_v1 (snapshot 1480)
│   ├── 03-generate-and-qc-3320.ipynb         snapshot comparison + synthetic_v2 (snapshot 3320, used in A3)
│   ├── qc_outputs/synthetic_v1/, synthetic_v2/   quality metrics, manifests, grids, nearest-neighbour pairs
│   └── vasc_check/                           VASC colour check, real train vs synthetic
├── class-imbalance/
│   ├── 04-classifier-sq256.ipynb             ResNet-50 classifier, arms A0-A3
│   ├── classifier_sq256_A0-A2/               outputs of the A0-A2 run (per-run metrics, predictions, epoch logs)
│   ├── classifier_sq256_A3/                  outputs of the A3 run + list of the 392 synthetic images used
│   ├── make_tables_final.py                  builds the result tables from the files above
│   ├── report_tables/inputs/                 GAN quality files read by the script
│   ├── report_tables/outputs/                generated tables + paired_bootstrap.csv
│   └── archive/present2_baseline_A0-A2/      earlier baseline on original-aspect images
└── reference-replication/
    └── han1000.ipynb                         re-implementation of STGAN (Su et al., 2024), image-level split
```

## Where each number in the report comes from

| Report content | Files |
|---|---|
| Split v2, Tables I and III | `data-understanding/01_ham10000_lesion_split.ipynb`, `data-understanding/output-d.u/` |
| Leakage of earlier protocols (29.5%, 38.6% / 39.5%), Table II | `data-understanding/check_*.py`, `bissoto2021_lesion_overlap.txt`, `su_split_overlap_*` |
| GAN training (G0, 3,320 kimg) | `gan/02-train-gan-5th.ipynb`, `gan/training_logs/` |
| Synthetic-image quality, Tables IV and V, VASC colour check | `gan/03-*.ipynb`, `gan/qc_outputs/`, `gan/vasc_check/` |
| Classifier results A0-A3, Tables VI–VIII | `class-imbalance/04-classifier-sq256.ipynb`, `classifier_sq256_*/`, `make_tables_final.py`, `report_tables/outputs/` |
| Earlier run on original-aspect images | `class-imbalance/archive/present2_baseline_A0-A2/` |
| STGAN reference replication | `reference-replication/han1000.ipynb` |

## Data

The images are not included. Download HAM10000 (ISIC 2018 Task 3 training set, 10,015 images) from the [ISIC Archive](https://challenge.isic-archive.com/data/#2018) or [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T).

The CSV files in `dataset/` come from the Harvard Dataverse. They are from ViDIR Group, Department of Dermatology, Medical University of Vienna, and licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (see `dataset/ISIC2018_Task3_Training_GroundTruth/ATTRIBUTION.txt`). Source paper: Tschandl, Rosendahl & Kittler, *The HAM10000 dataset*, Scientific Data 5, 180161 (2018).

## Not included
 
- HAM10000 images, `real_sq256/`, `gan_train.zip` and synthetic images,
- GAN snapshots (`.pkl`). The snapshot used for A3 is `network-snapshot-003320.pkl`, kept in the Kaggle output of `gan/02-train-gan-5th.ipynb`,
- the Bissoto et al. repository `gan-aug-analysis/`, which is cloned only to run `check_bissoto2021_lesion_overlap.py`.

## References

- Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
- Bissoto, A., Valle, E., & Avila, S. (2021). GAN-based data augmentation and anonymization for skin-lesion analysis: A critical review. *CVPR Workshops*. Code and splits: https://github.com/alceubissoto/gan-aug-analysis
- Su et al. (2024). A GAN-based data augmentation method for imbalanced multi-class skin lesion classification. *IEEE Access*.
- Karras, T., et al. (2020). Training generative adversarial networks with limited data. *NeurIPS*.
