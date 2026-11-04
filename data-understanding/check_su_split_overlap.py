"""Sibling exposure of a random image-level 8:2 split of HAM10000, as used by Su et al. (IEEE Access 2024)
and by the team's reference replication (notebook HAN1000)
The split is re-created exactly as in HAN1000's prepare_data.py (no --split-csv, no --group-by-lesion):
    sklearn.model_selection.train_test_split(metadata in file order, test_size=0.2,
                                             stratify=dx, random_state=42)
Su et al. do not report their split seed, so this measures our seed-42 version of their protocol,
their exact overlap may differ.

Usage:
    python check_su_split_overlap.py HAM10000_metadata.csv [--out-dir output] [--save su_style_split_seed42.csv]

    --out-dir writes su_split_overlap_summary.json (all numbers below) and
    su_split_overlap_per_class.csv (the per-class table) to that folder.

Result on 01/11/2026 (pandas 3.0.5, scikit-learn 1.9.1):
    train 8,012 / test 2,003 images, per-class counts identical to the HAN1000 notebook output
    test lesions with an image of the same lesion in train: 725 of 1,880 (38.6%)
    test images with a same-lesion image in train:          791 of 2,003 (39.5%)
    (per class, e.g. DF 17 of 23 test images, MEL 154 of 223)
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# per-class test counts printed by the HAN1000 notebook (cell "prepare_data"), used as a check
EXPECTED_TEST = {"akiec": 65, "bcc": 103, "bkl": 220, "df": 23, "mel": 223, "nv": 1341, "vasc": 28}

ap = argparse.ArgumentParser()
ap.add_argument("metadata", help="HAM10000_metadata.csv (official file, original row order)")
ap.add_argument("--test-size", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out-dir", default=None, help="optional: folder for the summary JSON and per-class CSV")
ap.add_argument("--save", default=None, help="optional: write image_id, lesion_id, dx, split to this CSV")
a = ap.parse_args()

df = pd.read_csv(a.metadata).reset_index(drop=True)
assert len(df) == 10015 and df.image_id.is_unique, "expected the full HAM10000 metadata (10,015 images)"
train, test = train_test_split(df, test_size=a.test_size, stratify=df.dx, random_state=a.seed)

print(f"train {len(train):,} | test {len(test):,} images")
got = test.dx.value_counts().to_dict()
print("test per class:", dict(sorted(got.items())))
if a.test_size == 0.2 and a.seed == 42:
    print("matches HAN1000 notebook per-class counts:", got == EXPECTED_TEST)

L_train = set(train.lesion_id)
test_les = test.drop_duplicates("lesion_id")
shared = test_les.lesion_id.isin(L_train)
exposed = test.lesion_id.isin(L_train)
print(f"test lesions with an image of the same lesion in train: {shared.sum():,} of {len(test_les):,} "
      f"({shared.mean():.1%})")
print(f"test images with a same-lesion image in train: {exposed.sum():,} of {len(test):,} ({exposed.mean():.1%})")

per = pd.DataFrame({"test_images": test.dx.value_counts(),
                    "exposed_images": test[exposed].dx.value_counts()}).fillna(0).astype(int)
per["exposed_%"] = (100 * per.exposed_images / per.test_images).round(1)
print(per.sort_index().to_string())

if a.out_dir:
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per.sort_index().to_csv(out_dir / "su_split_overlap_per_class.csv", index_label="dx")
    summary = {"metadata": a.metadata, "test_size": a.test_size, "seed": a.seed,
               "n_train_images": len(train), "n_test_images": len(test),
               "test_per_class": dict(sorted(got.items())),
               "matches_han1000_counts": (got == EXPECTED_TEST) if (a.test_size == 0.2 and a.seed == 42) else None,
               "test_lesions": int(len(test_les)), "test_lesions_shared_with_train": int(shared.sum()),
               "test_lesions_shared_pct": round(100 * float(shared.mean()), 1),
               "test_images_exposed": int(exposed.sum()),
               "test_images_exposed_pct": round(100 * float(exposed.mean()), 1)}
    (out_dir / "su_split_overlap_summary.json").write_text(json.dumps(summary, indent=2))
    print("saved", out_dir / "su_split_overlap_summary.json", "and", out_dir / "su_split_overlap_per_class.csv")

if a.save:
    out = pd.concat([train.assign(split="train"), test.assign(split="test")])
    out[["image_id", "lesion_id", "dx", "split"]].sort_values("image_id").to_csv(a.save, index=False)
    print("saved", a.save)
