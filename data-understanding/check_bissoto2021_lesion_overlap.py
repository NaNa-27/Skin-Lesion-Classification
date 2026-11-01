"""Check lesion overlap between the ISIC 2019 train pool and the isic19 test set used by
Bissoto, Valle & Avila (CVPRW 2021), restricted to the HAM10000 images (the only part of
ISIC 2019 with public lesion IDs).

Usage:
    git clone https://github.com/alceubissoto/gan-aug-analysis
    python check_bissoto2021_lesion_overlap.py gan-aug-analysis/splits ISIC2018_Task3_Training_LesionGroupings.csv HAM10000_metadata.csv

Result on 29/10/2026 (repo commit 0ee27b93):
    image overlap train/test = 0 (image-level split)
    HAM10000 lesions: train 4,948 | val 988 | test 1,923
    lesion overlap train∩test = 568 (29.5% of HAM test lesions)
    619 test images (16.0% of the 3,863-image isic19 test) have a sibling image of the same lesion
    in the train pool, incl. 75 of 396 malignant test images.
"""
import sys
import pandas as pd

splits, groupings, metadata = sys.argv[1:4]
g = pd.read_csv(groupings)
m = pd.read_csv(metadata)
lesion = dict(zip(g["image"], g["lesion_id"]))
dx = dict(zip(m["image_id"], m["dx"]))


def load(rel):
    d = pd.read_csv(f"{splits}/{rel}")
    d["id"] = d["image"].str.split("/").str[-1]
    d["lesion"] = d["id"].map(lesion)
    return d


tr = load("isic2019-full-train-pool.csv")
va = load("isic2019-val.csv")
te = load("testsets/isic2019-test.csv")

for name, d in [("train", tr), ("val", va), ("test", te)]:
    print(f"{name:5s} images {len(d):6,d} | HAM10000 images {d['lesion'].notna().sum():5,d} | "
          f"labels {d['label'].value_counts().to_dict()}")
print("image overlap train/test:", len(set(tr["id"]) & set(te["id"])),
      "| train/val:", len(set(tr["id"]) & set(va["id"])))

L_tr, L_va, L_te = (set(d["lesion"].dropna()) for d in (tr, va, te))
print(f"HAM10000 lesions: train {len(L_tr):,} | val {len(L_va):,} | test {len(L_te):,}")
print(f"lesion overlap train∩test {len(L_tr & L_te)} ({len(L_tr & L_te) / len(L_te):.1%} of test lesions) | "
      f"train∩val {len(L_tr & L_va)} | val∩test {len(L_va & L_te)}")

leak = te[te["lesion"].isin(L_tr)]
print(f"test images with a same-lesion sibling in train: {len(leak)} "
      f"({len(leak) / len(te):.1%} of the whole isic19 test) | "
      f"malignant {int((leak['label'] == 1).sum())} of {int((te['label'] == 1).sum())} | "
      f"dx {leak['id'].map(dx).value_counts().to_dict()}")
