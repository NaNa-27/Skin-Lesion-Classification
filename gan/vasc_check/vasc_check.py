# VASC check: real TRAIN images vs synthetic sets.
# Add Input: output of 01-prepare-gan-data (real_sq256), dataset dsr301m (manifest v2),
#            and, if available, the outputs of notebook 03 that contain synthetic_v1 (1480) and synthetic_v2 (3320).
# Uses TRAIN images only: val/test images must not influence the choice of generator snapshot.
import glob, os
import numpy as np, pandas as pd
from PIL import Image

ROOT = "/kaggle/input"
OUT = "/kaggle/working/vasc_check"
os.makedirs(OUT, exist_ok=True)

def find_one(pattern):
    hits = sorted(glob.glob(f"{ROOT}/**/{pattern}", recursive=True))
    assert hits, f"not found: {pattern}"
    return hits[0]

manifest = pd.read_csv(find_one("HAM10000_lesion_split_seed42_v2.csv"))
assert set(manifest.split) == {"train", "val", "test"} and len(manifest) == 10015, "wrong manifest"

real_dir = os.path.dirname(find_one("real_sq256/*.png"))
real = manifest[(manifest.dx == "vasc") & (manifest.split == "train")]
real_paths = [f"{real_dir}/{i}.png" for i in real.image_id]
print("real train VASC:", len(real_paths))
assert len(real_paths) == 99, "expected 99 train VASC images (v2 split)"

sets = {"real_train": real_paths}
for p in glob.glob(f"{ROOT}/**/synthetic_manifest.csv", recursive=True):
    sm = pd.read_csv(p)
    snap = str(sm.generator_snapshot.iloc[0]).split("-")[-1].split(".")[0]
    sm = sm[(sm.dx == "vasc") & (~sm.flag_memorized.astype(str).str.lower().eq("true"))]
    sets[f"syn_{int(snap)}"] = [os.path.join(os.path.dirname(p), x) for x in sm.path.head(200)]
print({k: len(v) for k, v in sets.items()})

def contact_sheet(paths, fname, cols=10, size=96):
    rows = int(np.ceil(len(paths) / cols))
    sheet = Image.new("RGB", (cols * size, rows * size), "white")
    for k, p in enumerate(paths):
        im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
        sheet.paste(im, ((k % cols) * size, (k // cols) * size))
    sheet.save(f"{OUT}/{fname}")

def lesion_stats(p):
    """Descriptive colour of the darkest part of the central region (where the lesion usually is)."""
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
    h, w, _ = a.shape
    c = a[h // 4: 3 * h // 4, w // 4: 3 * w // 4].reshape(-1, 3)
    lum = 0.299 * c[:, 0] + 0.587 * c[:, 1] + 0.114 * c[:, 2]
    dark = c[lum <= np.percentile(lum, 10)]               # darkest 10% of central pixels
    r, g, b = dark.mean(0)
    return {"p05_lum": float(np.percentile(lum, 5)),       # low = dark lesion present
            "contrast": float(np.median(lum) - np.percentile(lum, 5)),   # low = little/no visible lesion
            "red_minus_blue": float(r - b)}                # high = red/violet, low/negative = blue/grey

rows = []
for name, paths in sets.items():
    for p in paths:
        rows.append({"set": name, **lesion_stats(p)})
st = pd.DataFrame(rows)
st.to_csv(f"{OUT}/vasc_colour_stats.csv", index=False)

summary = st.groupby("set").agg(
    n=("p05_lum", "size"),
    p05_lum_median=("p05_lum", "median"),
    frac_dark_lt60=("p05_lum", lambda x: float((x < 60).mean())),
    frac_dark_lt90=("p05_lum", lambda x: float((x < 90).mean())),
    red_minus_blue_median=("red_minus_blue", "median"),
    frac_blueish=("red_minus_blue", lambda x: float((x < 10).mean())),
    frac_low_contrast_lt25=("contrast", lambda x: float((x < 25).mean())),
).round(3)
summary.to_csv(f"{OUT}/vasc_colour_summary.csv")
print(summary.to_string())

contact_sheet(real_paths, "real_train_vasc.png")
for name, paths in sets.items():
    if name != "real_train":
        contact_sheet(paths[:100], f"{name}_vasc.png")
print("saved to", OUT, os.listdir(OUT))
