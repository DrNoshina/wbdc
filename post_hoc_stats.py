#!/usr/bin/env python3
"""
Post-hoc statistics computed from the saved predictions (no retraining):
  1. ECE with 10 equal-width and 10 equal-mass (adaptive) bins on the pooled out-of-fold predictions
  2. Nadeau-Bengio corrected resampled t-test: each hybrid model vs its end-to-end model,
     fold-wise AUC and accuracy over the 10 x 5 CV folds (test/train ratio 0.25), Holm-corrected

  python post_hoc_stats.py --res results_v3      (WDBC)
  python post_hoc_stats.py --res results_wbco    (WBCO)
Outputs: <res>/extra/ece_equal_width_vs_equal_mass.csv, <res>/extra/nadeau_bengio_hybrid_vs_e2e.csv
"""
import argparse, glob, os
import numpy as np, pandas as pd
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument("--res", required=True)
p.add_argument("--bins", type=int, default=10)
a = p.parse_args()
EX = os.path.join(a.res, "extra"); os.makedirs(EX, exist_ok=True)


def ece_width(y, pr, b):
    e = np.linspace(0, 1, b + 1); idx = np.clip(np.digitize(pr, e[1:-1]), 0, b - 1); s = 0.0
    for k in range(b):
        m = idx == k
        if m.any():
            s += m.sum() / len(pr) * abs(pr[m].mean() - y[m].mean())
    return s


def ece_mass(y, pr, b):
    return sum(len(ix) / len(pr) * abs(pr[ix].mean() - y[ix].mean()) for ix in np.array_split(np.argsort(pr), b))


def holm(pv):
    pv = np.asarray(pv); o = np.argsort(pv); m = len(pv); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * pv[i]); adj[i] = min(1.0, run)
    return adj


# 1. ECE, pooled out-of-fold predictions of every repeat (main + extra baselines)
reps = []
for f in sorted(glob.glob(os.path.join(a.res, "cv_oof_probabilities_repeat*.csv"))):
    d = pd.read_csv(f)
    r = f.split("repeat")[-1].split(".")[0]
    fe = os.path.join(EX, f"cv_oof_probabilities_extra_repeat{r}.csv")
    if os.path.exists(fe):
        d = pd.concat([d, pd.read_csv(fe).drop(columns="y_true")], axis=1)
    reps.append(d)
rows = []
for m in [c for c in reps[0].columns if c != "y_true"]:
    w = [ece_width(d.y_true.values, d[m].values, a.bins) for d in reps]
    q = [ece_mass(d.y_true.values, d[m].values, a.bins) for d in reps]
    rows.append({"Model": m, "ECE_equal_width_mean": np.mean(w), "ECE_equal_width_sd": np.std(w, ddof=1),
                 "ECE_equal_mass_mean": np.mean(q), "ECE_equal_mass_sd": np.std(q, ddof=1), "n_repeats": len(reps)})
pd.DataFrame(rows).to_csv(os.path.join(EX, "ece_equal_width_vs_equal_mass.csv"), index=False)

# 2. Nadeau-Bengio corrected resampled t-test
per = pd.read_csv(os.path.join(a.res, "cv_metrics_per_fold.csv"))
res = []
for b in ["FTTransformer", "DCNv2", "TabNet", "TabMLPMixer"]:
    e2 = per[per.Model == f"{b}-EndToEnd"].set_index(["Repeat", "Fold"])
    for h in ["XGBoost", "ExtraTrees", "kNN"]:
        hy = per[per.Model == f"{b}+{h}"].set_index(["Repeat", "Fold"])
        for met in ["AUC", "Accuracy"]:
            d = (hy[met] - e2[met]).dropna().values; J = len(d); v = d.var(ddof=1)
            t = d.mean() / np.sqrt((1 / J + 0.25) * v) if v > 0 else 0.0
            res.append({"Metric": met, "Hybrid": f"{b}+{h}", "MeanDiff_vs_E2E": d.mean(), "t_corrected": t,
                        "p": 2 * stats.t.sf(abs(t), J - 1), "n_folds": J})
res = pd.DataFrame(res)
for met in ["AUC", "Accuracy"]:
    m = res.Metric == met
    res.loc[m, "p_Holm"] = holm(res.loc[m, "p"].values)
res.to_csv(os.path.join(EX, "nadeau_bengio_hybrid_vs_e2e.csv"), index=False)
print(f"written to {EX}/")
