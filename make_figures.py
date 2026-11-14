#!/usr/bin/env python3
"""
Make every results figure of the paper from the saved re-run outputs (no retraining),
in the same design as the figures already in the paper (Inter font, same sizes, colours and layout).

ONE COMMAND (recommended):
  python make_figures.py --both
      -> deletes the old figures_paper/ folder, makes all WDBC figures (paper file names)
         and all WBCO figures (same names + _wbco), and writes figures_paper.zip

One dataset only:
  python make_figures.py --res results_v3   --out figures_paper
  python make_figures.py --res results_wbco --out figures_paper --suffix _wbco

All numbers come from the CSV files written by wdbc_rerun.py.
"""
import argparse, glob, json, os, shutil, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--res", default=None)
p.add_argument("--both", action="store_true", help="clean output folder, make WDBC + WBCO figures, zip them")
p.add_argument("--wdbc", default="results_v3", help="WDBC results folder (with --both)")
p.add_argument("--wbco", default="results_wbco", help="WBCO results folder (with --both)")
p.add_argument("--no_zip", action="store_true")
p.add_argument("--out", default="figures_paper")
p.add_argument("--suffix", default="")
p.add_argument("--name", default=None, help="dataset label in titles (default from run_log)")
a = p.parse_args()

if a.both:   # start from an empty folder so no old image is left behind
    if os.path.isdir(a.out):
        shutil.rmtree(a.out)
    me = os.path.abspath(__file__)
    for res, suf in [(a.wdbc, ""), (a.wbco, "_wbco")]:
        r = os.system(f'"{sys.executable}" "{me}" --res "{res}" --out "{a.out}" --suffix "{suf}" --no_zip')
        if r:
            sys.exit(1)
    zp = shutil.make_archive(a.out, "zip", root_dir=".", base_dir=a.out)
    print(f"{len(glob.glob(os.path.join(a.out, '*.png')))} figures in {a.out}/  ->  zipped to {zp}")
    sys.exit(0)
if a.res is None:
    p.error("give --res, or use --both")

R, O, S = a.res, a.out, a.suffix
os.makedirs(O, exist_ok=True)

# paper design uses the Inter font; it is bundled in ./fonts so every computer gives the same look
from matplotlib import font_manager
for f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "*.otf")):
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Inter", "DejaVu Sans"]
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 11, "axes.titlesize": 10,
                     "legend.fontsize": 9})

NICE = {"FTTransformer": "FT-Transformer", "DCNv2": "DCN", "TabNet": "TabNet", "TabMLPMixer": "TabMLP-Mixer"}
LONG = {"FTTransformer": "FT-Transformer", "DCNv2": "Deep & Cross Network", "TabNet": "TabNet",
        "TabMLPMixer": "Tabular MLP-Mixer"}
B = list(NICE)
HEADS = ["XGBoost", "ExtraTrees", "kNN"]


def nice(m):
    if m.endswith("-EndToEnd"):
        return f"{NICE[m[:-9]]} (end-to-end)"
    if "+" in m:
        b, h = m.split("+")
        return f"{NICE.get(b, b)} + {h}"
    return m


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(O, name.replace(".png", f"{S}.png")))   # fixed size, same as the paper figures
    plt.close(fig)


test = pd.read_csv(os.path.join(R, "single_split_test_probabilities.csv"))
yte = test["y_true"].values
NT = len(yte)
cvs = pd.read_csv(os.path.join(R, "cv_metrics_summary.csv")).set_index("Model")
fold = pd.read_csv(os.path.join(R, "cv_metrics_per_fold.csv"))
sel = json.load(open(os.path.join(R, "single_split_selection.json")))
reps = sorted(f for f in os.listdir(R) if f.startswith("cv_oof_probabilities_repeat"))
oof = [pd.read_csv(os.path.join(R, f)) for f in reps]
N = len(oof[0])
label = a.name or ("WBCO" if "wbco" in S.lower() else "WDBC")

# ---------------- 1. reliability diagrams (pooled OOF over all repeats) ----------------
pool = pd.concat(oof, ignore_index=True)
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
edges = np.linspace(0, 1, 11)
for j, b in enumerate(B):
    for i, m in enumerate([f"{b}-EndToEnd", f"{b}+XGBoost"]):
        pr, y = pool[m].values, pool["y_true"].values
        idx = np.clip(np.digitize(pr, edges[1:-1]), 0, 9)
        xs, ys, ns, ece = [], [], [], 0.0
        for k in range(10):
            mk = idx == k
            if mk.sum():
                xs.append(pr[mk].mean()); ys.append(y[mk].mean()); ns.append(mk.sum())
                ece += mk.sum() / len(y) * abs(pr[mk].mean() - y[mk].mean())
        A = ax[i, j]
        A.plot([0, 1], [0, 1], "k--", lw=1)
        A.plot(xs, ys, "o-")
        for x_, y_, n_ in zip(xs, ys, ns):
            A.annotate(str(n_), (x_, y_), textcoords="offset points", xytext=(4, -9), fontsize=7)
        A.set_title(f"{m}\nECE={ece:.3f} (pooled OOF, n={len(y)})", fontsize=9)
        A.set_xlabel("Mean predicted P(malignant)"); A.set_ylabel("Observed fraction malignant")
fig.tight_layout(); save(fig, "reliability_diagram.png")

# ---------------- 2. OOF ROC of top-5 deep/hybrid models (repeat 0) ----------------
deep = [m for m in cvs.index if any(m.startswith(b) for b in B)]
top5 = cvs.loc[deep, "AUC_mean"].sort_values(ascending=False).index[:5]
fig, A = plt.subplots(figsize=(5.5, 5.5))
for m in top5:
    fpr, tpr, _ = roc_curve(oof[0]["y_true"], oof[0][m])
    A.plot(fpr, tpr, label=f"{nice(m)} (AUC = {roc_auc_score(oof[0]['y_true'], oof[0][m]):.3f})")
A.plot([0, 1], [0, 1], "k--", lw=1)
A.set_xlabel("False positive rate"); A.set_ylabel("True positive rate")
A.set_title(f"Out-of-fold ROC curves (n = {N})"); A.legend(fontsize=8, loc="lower right")
save(fig, "roc_top5.png")

# ---------------- 3. confusion matrices (test set) ----------------
for m, fn in [("FTTransformer-EndToEnd", "cm_FTTransformer_EndToEnd.png"),
              ("FTTransformer+XGBoost", "cm_FTTransformer+XGBoost.png"),
              ("FTTransformer+kNN", "cm_FTTransformer+KNN.png"),
              ("FTTransformer+ExtraTrees", "cm_FTTransformer+ExtraTrees.png"),
              ("TabMLPMixer-EndToEnd", "cm_TabMLPMixer_EndToEnd.png")]:
    cm = confusion_matrix(yte, (test[m].values >= 0.5).astype(int), labels=[1, 0])
    fig, A = plt.subplots(figsize=(4, 3.6))
    A.imshow(cm, cmap="Blues", vmin=0, vmax=1.1 * cm.max())
    for r in range(2):
        for c in range(2):
            A.text(c, r, cm[r, c], ha="center", va="center", fontsize=16,
                   color="white" if cm[r, c] > cm.max() / 2 else "black")
    A.set_xticks([0, 1]); A.set_xticklabels(["Malignant", "Benign"])
    A.set_yticks([0, 1]); A.set_yticklabels(["Malignant", "Benign"])
    A.set_xlabel("Predicted"); A.set_ylabel("True")
    A.set_title(f"{nice(m)}\n(test set, n = {NT})")
    save(fig, fn)

# ---------------- 4. training curves ----------------
short = {"FTTransformer": "FTTransformer", "DCNv2": "DCN", "TabNet": "TabNet", "TabMLPMixer": "TabMLPMixer"}
for b in B:
    tc = pd.read_csv(os.path.join(R, f"training_curve_{b}.csv"))
    ep = sel["early_stop_epoch"][b]
    fig, A = plt.subplots(figsize=(4.5, 3.5))
    A.plot(tc.epoch, tc.val_auc, color="tab:green", label="Validation AUC")
    A.axvline(ep, color="red", ls=":", label=f"Selected epoch ({ep})")
    A.set_xlabel("Epoch"); A.set_ylabel("Validation AUC"); A.set_title(f"{NICE[b]}: validation AUC"); A.legend()
    save(fig, f"phase3redo_{short[b]}_{'val_auc' if b == 'TabNet' else 'acc'}.png")
    fig, A = plt.subplots(figsize=(4.5, 3.5))
    A.plot(tc.epoch, tc.train_loss, label="Training loss"); A.plot(tc.epoch, tc.val_loss, label="Validation loss")
    A.axvline(ep, color="red", ls=":", label=f"Selected epoch ({ep})")
    A.set_xlabel("Epoch"); A.set_ylabel("Cross-entropy loss"); A.set_title(f"{NICE[b]}: loss"); A.legend()
    save(fig, f"phase3redo_{short[b]}_loss.png")

# ---------------- 5. classical baseline boxplots (LR vs RF) ----------------
mets = [("Accuracy", "Accuracy"), ("F1", "F1-score"), ("AUC", "AUC"), ("Recall(Sens)", "Sensitivity"),
        ("Specificity", "Specificity")]
fig, A = plt.subplots(figsize=(9, 4.2))
for k, (mdl, col) in enumerate([("LogisticRegression", "tab:blue"), ("RandomForest", "tab:orange")]):
    d = fold[fold.Model == mdl]
    bp = A.boxplot([d[c].values for c, _ in mets], positions=np.arange(5) + (k - 0.5) * 0.3, widths=0.25,
                   patch_artist=True)
    for bx in bp["boxes"]:
        bx.set_facecolor(col); bx.set_alpha(0.6)
    A.plot([], [], color=col, lw=8, alpha=0.6, label="Logistic Regression" if k == 0 else "Random Forest")
A.set_xticks(range(5)); A.set_xticklabels([n for _, n in mets])
A.set_ylabel(f"Value over {fold.Repeat.nunique() * fold.Fold.nunique()} folds")
A.set_title(f"Repeated stratified cross-validation ({fold.Repeat.nunique()} x {fold.Fold.nunique()} folds)", fontsize=10)
A.legend(fontsize=10); A.grid(axis="y", alpha=0.3)
save(fig, "phase2_metric_boxplots.png")

# ---------------- 6. per-backbone bars (accuracy, F1, AUC; CV mean +/- SD) ----------------
for b in B:
    ms = [f"{b}-EndToEnd"] + [f"{b}+{h}" for h in HEADS]
    fig, A = plt.subplots(figsize=(6, 4))
    x = np.arange(4); w = 0.25
    for k, (c, n) in enumerate([("Accuracy", "Accuracy"), ("F1", "F1-score"), ("AUC", "AUC")]):
        A.bar(x + (k - 1) * w, [cvs.loc[m, f"{c}_mean"] for m in ms], w,
              yerr=[cvs.loc[m, f"{c}_sd"] for m in ms], capsize=3, label=n)
    A.set_xticks(x); A.set_xticklabels(["End-to-end", "+ XGBoost", "+ ExtraTrees", "+ kNN"])
    A.set_ylim(0.85, 1.01); A.set_ylabel(f"Mean ± SD ({fold.Repeat.nunique()} × {fold.Fold.nunique()}-fold CV)")
    A.set_title(LONG[b]); A.legend(fontsize=8, loc="lower right"); A.grid(axis="y", alpha=0.3)
    save(fig, f"{LONG[b]}_bars_acc_f1_auc.png")

# ---------------- 7. hybrid gain: AUC(hybrid) - AUC(end-to-end), mean and 95% CI over folds ----------------
rows = []
for b in B:
    e2e = fold[fold.Model == f"{b}-EndToEnd"].set_index(["Repeat", "Fold"]).AUC
    for h in HEADS:
        hy = fold[fold.Model == f"{b}+{h}"].set_index(["Repeat", "Fold"]).AUC
        d = (hy - e2e).dropna().values
        ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        rows.append((f"{NICE[b]} + {h}", d.mean(), ci, len(d)))
order = ["DCNv2", "FTTransformer", "TabNet", "TabMLPMixer"]
rows = sorted(rows, key=lambda r: (order.index([k for k in B if r[0].startswith(NICE[k] + " ")][0]), HEADS.index(r[0].split(" + ")[1])))[::-1]
fig, A = plt.subplots(figsize=(7.5, 5))
A.errorbar([r[1] for r in rows], range(len(rows)), xerr=[r[2] for r in rows], fmt="o", capsize=4)
A.axvline(0, color="k", lw=1)
A.set_yticks(range(len(rows))); A.set_yticklabels([r[0] for r in rows], fontsize=9)
A.set_xlabel(f"AUC difference (hybrid − end-to-end)\nmean and 95% CI over {rows[0][3]} CV folds")
A.grid(axis="x", alpha=0.3)
save(fig, "phase4_hybrid_ablation.png")
pd.DataFrame(rows, columns=["Model", "AUC_diff_mean", "CI95_halfwidth", "n_folds"]).to_csv(
    os.path.join(O, f"hybrid_gain{S}.csv"), index=False)

# ---------------- 8. robustness ----------------
rb = pd.read_csv(os.path.join(R, "robustness_noise.csv"))
fig, A = plt.subplots(figsize=(7, 5))
for m in sorted(rb.Model.unique()):
    d = rb[rb.Model == m].sort_values("sigma")
    A.errorbar(d.sigma, d.AUC_mean, yerr=d.AUC_sd, marker="o", ms=3, capsize=2,
               ls="--" if m.endswith("EndToEnd") else "-", label=nice(m))
A.set_xlabel("Gaussian noise σ (standardised features)")
A.set_ylabel(f"Test AUC (mean ± SD, {int(rb.n_repeats.max())} repeats)")
A.legend(fontsize=7, loc="lower left"); A.grid(alpha=0.3)
save(fig, "phase4_robustness.png")

# ---------------- 9. SHAP vs LIME importance ----------------
fi = pd.read_csv(os.path.join(R, "xai_feature_importance_test.csv"))
ag = pd.read_csv(os.path.join(R, "xai_shap_lime_agreement.csv")).iloc[0]
top = fi.sort_values("mean|SHAP|", ascending=False).head(10)[::-1]
lab = [f.replace("_", " ") for f in top.Feature]
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
ax[0].barh(lab, top["mean|SHAP|"], color="tab:blue"); ax[0].set_title("Mean |SHAP value| (test set)", fontsize=9)
ax[1].barh(lab, top["mean|LIME|"], color="tab:orange"); ax[1].set_title("Mean |LIME weight| (test set)", fontsize=9)
fig.suptitle(f"{nice(ag.Model)}: Spearman ρ = {ag.Spearman_global:.2f} between SHAP and LIME importances "
             f"({len(fi)} features)", fontsize=9)
fig.tight_layout(); save(fig, "lime_shap_importance.png")

# ---------------- 10. SHAP summary (beeswarm made by the run itself) ----------------
src = os.path.join(R, "figures", "shap_summary_test.png")
if os.path.exists(src):
    from PIL import Image                      # same canvas as the paper figure (8 x 7.5 in at 300 dpi)
    im = Image.open(src).convert("RGB"); W, H = 2400, 2250
    if im.size != (W, H):
        im.thumbnail((W, H), Image.LANCZOS)
        c = Image.new("RGB", (W, H), "white"); c.paste(im, ((W - im.width) // 2, (H - im.height) // 2)); im = c
    im.save(os.path.join(O, f"shap_hybrid_summary{S}.png"), dpi=(300, 300))

print(f"{label}: figures written to {O}/ with suffix '{S}'")

if not a.no_zip:
    zp = shutil.make_archive(O.rstrip("/"), "zip", root_dir=".", base_dir=O)
    print(f"zipped to {zp}")
