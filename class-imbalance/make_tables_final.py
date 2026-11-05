"""Generate the result and synthetic-image tables of the final report from project files.

Usage
  python make_tables_final.py --results RUNS_DIR [RUNS_DIR ...] [--inputs inputs] [--out tables]

RUNS_DIR = a downloaded `classifier_sq256/` folder (notebook 04). Several folders can be given
(A0-A2 run of 30/10 and the A3 run), runs are collected from RUNS_DIR/runs/*/metrics.json.
Every run must use the v2 manifest and the real_sq256 input.

Writes to --out:
  tab_results.tex   test metrics per arm (mean +- SD over seeds, lesion-bootstrap CI)
  tab_perclass.tex  per-class test recall
  tab_paired.tex    A3 - A2 per seed and paired lesion-bootstrap CI (only if both arms exist)
  tab_quality.tex   synthetic-image quality along G0 training (snapshot comparison, notebook 03)
  tab_synthetic.tex final synthetic set vs real-real reference + real/fake detector
  tab_vasc.tex      VASC colour check (real train vs synthetic)
  results_runs_used.csv, paired_bootstrap.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

EXPECTED_SHA256 = "bda53d198ff403653548b8f072116e78350067417ddf41ec3e342f034bf464d7"
CLASS_ORDER = ["nv", "mel", "bkl", "bcc", "akiec", "vasc", "df"]
ARM_LABEL = {"A0_ce": "A0: cross-entropy",
             "A1_weighted_ce": "A1: weighted CE",
             "A2_oversample": "A2: class-balanced oversampling",
             "A3_gan_topup": "A3: A2 + GAN images"}
SHORT = {"A0_ce": "A0", "A1_weighted_ce": "A1", "A2_oversample": "A2", "A3_gan_topup": "A3"}
N_BOOT = 2000


def pm(s, d=3):
    return f"{s.iloc[0]:.{d}f}" if len(s) == 1 else f"{s.mean():.{d}f} $\\pm$ {s.std(ddof=1):.{d}f}"


def collect(dirs):
    rows, preds = [], {}
    for d in dirs:
        for f in sorted(Path(d).glob("runs/*/metrics.json")):
            r = json.loads(f.read_text())
            assert r["manifest_sha256"] == EXPECTED_SHA256, f"{f}: wrong manifest"
            assert not r["config"].get("DRY_RUN"), f"{f}: dry run"
            base = r.get("base_arm") or r["arm"].split("-")[0]
            t = r["test"]
            row = {"run": f.parent.name, "arm": base, "seed": r["seed"], "n": t["n"], "best_epoch": r["best_epoch"],
                   "n_train": r["n_train_images"], "macro_f1": t["macro_f1"], "bacc": t["balanced_accuracy"],
                   "acc": t["accuracy"], "auroc": t["macro_ovr_auroc"], "val_macro_f1": r["val"]["macro_f1"],
                   "ci_lo": r["test_bootstrap"]["macro_f1_ci95"][0], "ci_hi": r["test_bootstrap"]["macro_f1_ci95"][1],
                   "synthetic": (r.get("synthetic") or {}).get("version", "") if isinstance(r.get("synthetic"), dict) else ""}
            for c in CLASS_ORDER:
                row[f"rec_{c}"] = t["per_class"][c]["recall"]
            key = (base, r["seed"])
            assert key not in preds, f"duplicate run for {key}"
            p = pd.read_csv(f.parent / "pred_test.csv")
            # recompute from predictions: must match metrics.json
            assert abs(f1_score(p.y_true, p.y_pred, average="macro") - t["macro_f1"]) < 1e-6, f"{f}: macro-F1 mismatch"
            assert abs(balanced_accuracy_score(p.y_true, p.y_pred) - t["balanced_accuracy"]) < 1e-6, f"{f}: BA mismatch"
            preds[key] = p
            rows.append(row)
            cfg = r["config"]
    df = pd.DataFrame(rows)
    test_ids = {tuple(p.image_id) for p in preds.values()}
    assert len(test_ids) == 1, "runs were evaluated on different test sets"
    return df, preds, cfg


def results_tables(df, cfg, out):
    arms = [a for a in ARM_LABEL if a in set(df.arm)]
    lines = []
    for a in arms:
        d = df[df.arm == a]
        lines.append(f"{ARM_LABEL[a]} & {pm(d.macro_f1)} & [{d.ci_lo.mean():.3f}, {d.ci_hi.mean():.3f}] & "
                     f"{pm(d.bacc)} & {pm(d.acc)} & {pm(d.auroc)} & {d.val_macro_f1.mean():.3f} \\\\")
    (out / "tab_results.tex").write_text(rf"""\begin{{table*}}[t]
\centering
\caption{{Test results on the lesion-level split (n = {int(df.n.iloc[0]):,} real images). ResNet-50 (ImageNet), {cfg['IMG_SIZE']}$\times${cfg['IMG_SIZE']} crops of 256$\times$256 centre-square images, {cfg['EPOCHS']} epochs, best epoch chosen on validation macro-F1, test evaluated once per run. A3 adds GAN images for AKIEC, DF and VASC only. Mean $\pm$ SD over seeds 1--3 (3 runs per arm). CI = 95\% lesion-level bootstrap interval of test macro-F1 per run, bounds averaged over seeds. Last column: validation macro-F1 (model-selection metric).}}
\label{{tab:results}}
\footnotesize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lcccccc}}
\toprule
Arm & Macro-F1 & Macro-F1 95\% CI & Bal.\ acc. & Accuracy & Macro AUROC & Val.\ macro-F1 \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}
\end{{table*}}
""")
    head = " & ".join(c.upper() for c in CLASS_ORDER)
    pl = [f"{SHORT[a]} & " + " & ".join(f"{df[df.arm == a][f'rec_{c}'].mean():.2f}" for c in CLASS_ORDER) + " \\\\"
          for a in arms]
    (out / "tab_perclass.tex").write_text(rf"""\begin{{table}}[t]
\centering
\caption{{Per-class test recall (mean over seeds 1--3). Test support: DF 20 images / 11 lesions, VASC 21 / 14, AKIEC 52 images.}}
\label{{tab:perclass}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{l{'c' * len(CLASS_ORDER)}}}
\toprule
Arm & {head} \\
\midrule
{chr(10).join(pl)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


def metric_vals(p, idx):
    yt, yp = p.y_true.values[idx], p.y_pred.values[idx]
    labels = list(range(len(CLASS_ORDER)))
    rec = recall_score(yt, yp, labels=labels, average=None, zero_division=0)
    return f1_score(yt, yp, labels=labels, average="macro", zero_division=0), balanced_accuracy_score(yt, yp), rec


def paired(df, preds, out, a="A3_gan_topup", b="A2_oversample"):
    seeds = sorted(set(df[df.arm == a].seed) & set(df[df.arm == b].seed))
    if not seeds:
        print("paired table skipped: need both", a, "and", b)
        return
    p0 = preds[(b, seeds[0])]
    lesions = p0.lesion_id.values
    uniq = np.unique(lesions)
    idx_of = {l: np.where(lesions == l)[0] for l in uniq}
    # class index used in pred files (alphabetical order of class names)
    alpha = sorted(CLASS_ORDER)
    minority = {c: alpha.index(c) for c in ["akiec", "df", "vasc", "mel"]}
    rng = np.random.default_rng(0)
    full = np.arange(len(p0))

    def diff(idx):
        out_ = []
        for s in seeds:
            fa, ba, ra = metric_vals(preds[(a, s)], idx)
            fb, bb, rb = metric_vals(preds[(b, s)], idx)
            out_.append([fa - fb, ba - bb] + [ra[k] - rb[k] for k in minority.values()])
        return np.mean(out_, axis=0), np.array(out_)

    point, per_seed = diff(full)
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        boots.append(diff(np.concatenate([idx_of[l] for l in pick]))[0])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    names = ["Macro-F1", "Bal.\\ acc."] + [f"Recall {c.upper()}" for c in minority]
    pd.DataFrame({"metric": names, "mean_diff": point, "ci_lo": lo, "ci_hi": hi,
                  **{f"seed{s}": per_seed[i] for i, s in enumerate(seeds)}}).to_csv(out / "paired_bootstrap.csv", index=False)
    rows = []
    for i, n in enumerate(names):
        rows.append(f"{n} & " + " & ".join(f"${per_seed[j, i]:+.3f}$" for j in range(len(seeds)))
                    + f" & ${point[i]:+.3f}$ & $[{lo[i]:+.3f}, {hi[i]:+.3f}]$ \\\\")
    sc = " & ".join(f"Seed {s}" for s in seeds)
    (out / "tab_paired.tex").write_text(rf"""\begin{{table}}[t]
\centering
\caption{{Paired difference A3 $-$ A2 on the same test lesions (same seed). CI = 95\% lesion-level bootstrap of the seed-averaged difference ({N_BOOT} resamples of {len(uniq):,} test lesions; it reflects test-set sampling, not training variation).}}
\label{{tab:paired}}
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{l{'c' * len(seeds)}cc}}
\toprule
Metric & {sc} & Mean & 95\% CI \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    print(pd.read_csv(out / "paired_bootstrap.csv").round(4).to_string(index=False))


def quality_tables(inp, out):
    pr = pd.read_csv(inp / "snapshot_comparison.csv")
    fk = pd.read_csv(inp / "snapshot_comparison_fid_kid.csv")
    ref = fk[~fk.snapshot.str.startswith("network")].iloc[0]
    fk = fk[fk.snapshot.str.startswith("network")]
    g = pr.groupby("snapshot").agg(prec=("precision", "mean"), cov=("coverage", "mean"),
                                   rec=("recall", "mean"), mem=("mem_rate", "max")).reset_index()
    t = fk.merge(g, on="snapshot")
    t["kimg"] = t.snapshot.str.extract(r"(\d+)").astype(int)
    lines = []
    for _, r in t.sort_values("kimg").iterrows():
        mark = r"$^\dagger$" if r.kimg == 3320 else (r"$^\ddagger$" if r.kimg in (1480, 2400) else "")
        lines.append(f"{r.kimg:,}{mark} & {r.FID_inception:.1f} & {r.FID_dinov2:.0f} & {r.prec:.3f} & {r.rec:.3f} & "
                     f"{r['cov']:.3f} & {r.mem:.3f} \\\\")
    lines.append(f"Real val vs.\\ real train & {ref.FID_inception:.1f} & {ref.FID_dinov2:.0f} & -- & -- & -- & -- \\\\")
    (out / "tab_quality.tex").write_text(rf"""\begin{{table}}[t]
\centering
\caption{{Synthetic-image quality along G0 training (snapshot comparison: 200 images per class from fixed latents; FID/FD on 1,532 images matching the class mix of the validation split). P, R, C = precision, recall, coverage (ResNet-50 features, $k=5$), averaged over AKIEC, DF and VASC; Mem.\ = largest per-class near-duplicate rate. $^\ddagger$snapshots compared during selection (1,480 kimg as reference, 2,400 kimg as candidate); $^\dagger$snapshot used for A3 (also a candidate).}}
\label{{tab:quality}}
\setlength{{\tabcolsep}}{{3.5pt}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{lrrcccc}}
\toprule
kimg & FID & FD$_\mathrm{{DINOv2}}$ & P & R & C & Mem. \\
\midrule
{chr(10).join(lines)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    fq = pd.read_csv(inp / "final_quality_3320.csv").set_index("metric")
    d33 = json.loads((inp / "final_detector_3320.json").read_text())
    d14 = json.loads((inp / "final_detector_1480.json").read_text())
    nice = {"FID_inception": "FID (Inception)", "FID_dinov2": "FD (DINOv2)"}
    rows = []
    for k in ["FID_inception", "FID_dinov2"]:
        r = fq.loc[k]
        rows.append(f"{nice[k]} & {r.fake_mean:.1f} $\\pm$ {r.fake_std:.1f} & {r.ref:.1f} \\\\")
    for c in ["akiec", "df", "vasc"]:
        r = fq.loc[f"KIDx1000_inception_{c}"]
        rows.append(f"KID$\\times10^3$ {c.upper()} & {r.fake_mean:.1f} $\\pm$ {r.fake_std:.1f} & {r.ref:.1f} \\\\")
    rows.append("\\midrule")
    for k, lab in [("AUC_probe_resnet50", "AUC, probe (ResNet-50)"),
                   ("AUC_probe_dinov2", "AUC, probe (DINOv2)"),
                   ("AUC_finetune_resnet18", "AUC, fine-tuned ResNet-18")]:
        rows.append(f"{lab} & {d33[k]:.3f} & {d14[k]:.3f} (1,480) \\\\")
    (out / "tab_synthetic.tex").write_text(rf"""\begin{{table}}[t]
\centering
\caption{{Synthetic set used for A3 (3,320 kimg). Distances: mean $\pm$ SD over 5 independent synthetic sets, next to the real--real reference (real validation vs.\ real train, same sample sizes). AUC: detector separating real (train/val) from unused synthetic AKIEC, DF and VASC images (198 test images); reference column: same detector on the 1,480-kimg set; 0.5 = indistinguishable.}}
\label{{tab:synthetic}}
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\columnwidth}}{{!}}{{\begin{{tabular}}{{lcc}}
\toprule
Measure & Synthetic (3,320) & Reference \\
\midrule
{chr(10).join(rows)}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")
    v = pd.read_csv(inp / "vasc_colour_summary.csv").set_index("set")
    lab = {"real_train": "Real train (99)", "syn_1480": "Synthetic 1,480 kimg", "syn_3320": "Synthetic 3,320 kimg"}
    vr = [f"{lab[s]} & {v.loc[s, 'frac_dark_lt60']:.2f} & {v.loc[s, 'frac_blueish']:.2f} & "
          f"{v.loc[s, 'frac_low_contrast_lt25']:.2f} \\\\" for s in ["real_train", "syn_1480", "syn_3320"] if s in v.index]
    (out / "tab_vasc.tex").write_text(rf"""\begin{{table}}[t]
\centering
\caption{{VASC colour check on the central image region (training images only). Dark: 5th-percentile luminance $<60$; bluish: red$-$blue of the darkest 10\% of pixels $<10$; low contrast (lesion barely visible): median$-$5th-percentile luminance $<25$.}}
\label{{tab:vasc}}
\begin{{tabular}}{{lccc}}
\toprule
Set & Dark & Bluish & Low contrast \\
\midrule
{chr(10).join(vr)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--inputs", default="inputs")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    df, preds, cfg = collect(a.results)
    df.sort_values(["arm", "seed"]).to_csv(out / "results_runs_used.csv", index=False)
    print(df.groupby("arm")[["macro_f1", "bacc", "acc", "val_macro_f1"]].agg(["mean", "std"]).round(4))
    results_tables(df, cfg, out)
    paired(df, preds, out)
    quality_tables(Path(a.inputs), out)
    print("tables written to", out.resolve())
