#!/usr/bin/env python3
"""
Extra experiments asked for in the second review round. They do NOT change the main results in
results/<dataset>/; everything is written to results/<dataset>/extra/.

  1. baselines : CatBoost and TabPFN with exactly the same splits and preprocessing as wdbc_rerun.py
                 (single 70/15/15 split and 10 x 5 repeated CV), plus McNemar/DeLong tests against
                 the main models. CatBoost uses early stopping on the same validation split as the
                 deep models; TabPFN needs no tuning.
  2. robust    : feature-wise robustness on the single split: Gaussian noise on ONE feature at a time,
                 and randomly missing values (10/20/30%, mean-imputed), for end-to-end, hybrid and raw models.
  3. stats     : Friedman test and Nemenyi critical difference over the 50 CV folds (AUC and accuracy).
  4. dca       : decision-curve analysis (net benefit) on the pooled out-of-fold predictions.

Usage (same settings as the paper, Table 2):
  python extra_experiments.py --dataset wdbc --res results/wdbc
  python extra_experiments.py --dataset wbco --res results/wbco
  options: --no_tabpfn (skip TabPFN)   --steps baselines,robust,stats,dca   --quick (smoke test only)

The script first checks that its splits are identical to the main run: it refits Logistic Regression
on every split and compares the predictions with the saved ones (the difference must be ~0).
If it stops, run the same command again: finished CV folds are reloaded from <res>/extra/cv_cache/.
"""
import argparse, glob, json, os, sys, time, warnings

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="wdbc", choices=["wdbc", "wbco", "coimbra"])
p.add_argument("--res", default=None, help="main results folder of this dataset (default results/<dataset>)")
p.add_argument("--steps", default="baselines,robust,stats,dca")
p.add_argument("--no_tabpfn", action="store_true")
p.add_argument("--feat_sigma", type=float, default=0.5, help="noise SD for the one-feature-at-a-time test")
p.add_argument("--feat_reps", type=int, default=20)
p.add_argument("--miss_rates", default="0.1,0.2,0.3")
p.add_argument("--miss_reps", type=int, default=100)
p.add_argument("--quick", action="store_true", help="smoke test; numbers are NOT valid")
a = p.parse_args()
RES = a.res or os.path.join("results", a.dataset)
EX = os.path.join(RES, "extra_quick" if a.quick else "extra")   # quick test never mixes with the real run
os.makedirs(os.path.join(EX, "figures"), exist_ok=True)
STEPS = [s.strip() for s in a.steps.split(",")]

# import the main pipeline with the same settings, so preprocessing, splits and models are identical
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = sys.argv
sys.argv = [_argv[0], "--dataset", a.dataset, "--out", EX] + (["--quick"] if a.quick else [])
import wdbc_rerun as W                                             # noqa: E402
sys.argv = _argv

import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from scipy import stats                                            # noqa: E402
from sklearn.linear_model import LogisticRegression                # noqa: E402
from sklearn.metrics import roc_auc_score                          # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402
import matplotlib                                                  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib import font_manager                                # noqa: E402

warnings.filterwarnings("ignore")
for f in glob.glob(os.path.join(HERE, "fonts", "*.otf")):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 10, "savefig.dpi": 300})

args = W.args
X_ALL, Y_ALL, N_ALL = W.X_ALL, W.Y_ALL, W.N_ALL
NICE = {"FTTransformer": "FT-Transformer", "DCNv2": "DCN", "TabNet": "TabNet", "TabMLPMixer": "TabMLP-Mixer"}


def log(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(os.path.join(EX, "extra_log.txt"), "a") as fh:
        fh.write(msg + "\n")


W.log = log


def nice(m):
    if m.endswith("-EndToEnd"):
        return f"{NICE[m[:-9]]} (end-to-end)"
    if "+" in m:
        b, h = m.split("+")
        return f"{NICE.get(b, b)} + {h}"
    return m


# ----------------------------------------------------------------------------- 1. extra baselines
def extra_models(seed):
    ms = {}
    try:
        from catboost import CatBoostClassifier
        ms["CatBoost"] = CatBoostClassifier(random_seed=seed, verbose=0, allow_writing_files=False,
                                            iterations=200 if a.quick else 1000)
    except ImportError:
        log("catboost is not installed (pip install catboost) -> skipped")
    if not a.no_tabpfn:
        try:
            from tabpfn import TabPFNClassifier
            ms["TabPFN"] = TabPFNClassifier(device="cuda" if W.GPU else "cpu", random_state=seed)
        except Exception as e:
            log("TabPFN not available -> skipped:", repr(e)[:200]); a.no_tabpfn = True
    return ms


def run_split(Xtr, ytr, Xva, yva, Xte, seed):
    Xtr_p, Xva_p, Xte_p, idx, sc = W.preprocess(Xtr, ytr, Xva, yva, Xte, seed)
    out = {"_check_LR": LogisticRegression(max_iter=5000).fit(Xtr_p, ytr).predict_proba(Xte_p)[:, 1]}
    for n, m in extra_models(seed).items():
        try:
            if n == "CatBoost":    # early stopping on the same validation split as the deep models
                m.fit(Xtr_p, ytr, eval_set=(Xva_p, yva), early_stopping_rounds=50, use_best_model=True)
            else:
                m.fit(Xtr_p, ytr)
            out[n] = m.predict_proba(Xte_p)[:, 1]
        except Exception as e:
            log(f"{n} failed -> skipped from now on: {repr(e)[:300]}")
            if n == "TabPFN":
                a.no_tabpfn = True
    return out


def summarise(per):
    keys = ["Accuracy", "Precision", "Recall(Sens)", "Specificity", "F1", "AUC", "Brier", "ECE", "CalSlope", "CalIntercept"]
    rows = []
    for n, d in per.groupby("Model"):
        row = {"Model": n}
        for k in keys:
            rep = d.groupby("Repeat")[k].mean().values
            mu, sd = d[k].mean(), d[k].std(ddof=1)
            half = stats.t.ppf(.975, len(rep) - 1) * rep.std(ddof=1) / np.sqrt(len(rep)) if len(rep) > 1 else np.nan
            row[k] = f"{mu:.4f} ± {sd:.4f}"; row[k + "_mean"] = mu; row[k + "_sd"] = sd
            row[k + "_CI95"] = f"[{mu - half:.4f}, {mu + half:.4f}]" if len(rep) > 1 else "n/a"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("AUC_mean", ascending=False)


def step_baselines():
    t0 = time.time()
    # single split, exactly as in wdbc_rerun.main()
    W.set_seed(args.seed)
    Xtmp, Xte, ytmp, yte = train_test_split(X_ALL, Y_ALL, test_size=W.N_TEST, stratify=Y_ALL, random_state=args.seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=W.N_VAL, stratify=ytmp, random_state=args.seed)
    pr = run_split(Xtr, ytr, Xva, yva, Xte, args.seed)
    saved = pd.read_csv(os.path.join(RES, "single_split_test_probabilities.csv"))
    diff = np.abs(saved["LogisticRegression"].values - pr["_check_LR"]).max()
    log(f"Split check (single split): max |LR difference| = {diff:.2e} -> {'same splits (difference is only numerical)' if diff < 1e-2 else 'WARNING: splits differ'}")
    rows = []
    for n, v in pr.items():
        if n.startswith("_"):
            continue
        mm = W.metrics(yte, v); mm["Model"] = n; rows.append(mm)
    pd.DataFrame(rows).to_csv(os.path.join(EX, "single_split_test_metrics_extra.csv"), index=False)
    pd.DataFrame({k: v for k, v in pr.items() if not k.startswith("_")}).assign(y_true=yte).to_csv(
        os.path.join(EX, "single_split_test_probabilities_extra.csv"), index=False)

    # repeated CV, exactly as in wdbc_rerun.main()
    rows, oof_all, worst = [], [], 0.0
    for r in range(args.repeats):
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed + r)
        oof = {}
        for f, (tr, te) in enumerate(skf.split(X_ALL, Y_ALL)):
            s = args.seed + 1000 * r + f
            Xa, Xv, ya, yv = train_test_split(X_ALL[tr], Y_ALL[tr], test_size=0.15, stratify=Y_ALL[tr], random_state=s)
            cache = os.path.join(EX, "cv_cache", f"r{r}_f{f}.pkl")
            if os.path.exists(cache):
                pr = pd.read_pickle(cache)
            else:
                pr = run_split(Xa, ya, Xv, yv, X_ALL[te], s)
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                pd.to_pickle(pr, cache)
            for n, v in pr.items():
                oof.setdefault(n, np.zeros(N_ALL))[te] = v
                if not n.startswith("_"):
                    mm = W.metrics(Y_ALL[te], v); mm.update({"Model": n, "Repeat": r, "Fold": f}); rows.append(mm)
            log(f"CV repeat {r + 1}/{args.repeats} fold {f + 1}/{args.folds} done ({(time.time() - t0) / 60:.1f} min)")
        main_oof = os.path.join(RES, f"cv_oof_probabilities_repeat{r}.csv")
        if os.path.exists(main_oof):
            worst = max(worst, np.abs(pd.read_csv(main_oof)["LogisticRegression"].values - oof["_check_LR"]).max())
        oof = {k: v for k, v in oof.items() if not k.startswith("_")}
        oof_all.append(oof)
        pd.DataFrame(oof).assign(y_true=Y_ALL).to_csv(os.path.join(EX, f"cv_oof_probabilities_extra_repeat{r}.csv"), index=False)
    if a.quick:
        log("Split check (CV folds): not done in quick mode (quick mode uses 3 folds, the main run used 5)")
    else:
        log(f"Split check (all CV folds): max |LR difference| = {worst:.2e} -> {'same splits (difference is only numerical)' if worst < 1e-2 else 'WARNING: splits differ'}")
    per = pd.DataFrame(rows); per.to_csv(os.path.join(EX, "cv_metrics_per_fold_extra.csv"), index=False)
    summarise(per).to_csv(os.path.join(EX, "cv_metrics_summary_extra.csv"), index=False)

    # pooled out-of-fold calibration (same as the main run)
    cal = []
    for n in oof_all[0]:
        v = np.array([[W.metrics(Y_ALL, o[n])[k] for k in ["AUC", "Accuracy", "Brier", "ECE", "CalSlope", "CalIntercept"]] for o in oof_all], float)
        row = {"Model": n}
        for j, k in enumerate(["AUC", "Accuracy", "Brier", "ECE", "CalSlope", "CalIntercept"]):
            row[k + "_mean"], row[k + "_sd"] = np.nanmean(v[:, j]), (np.nanstd(v[:, j], ddof=1) if len(v) > 1 else np.nan)
        cal.append(row)
    pd.DataFrame(cal).to_csv(os.path.join(EX, "cv_pooled_oof_calibration_extra.csv"), index=False)

    # McNemar + DeLong on pooled OOF predictions of repeat 0, against the main models
    main0 = pd.read_csv(os.path.join(RES, "cv_oof_probabilities_repeat0.csv"))
    refs = [m for m in ["LogisticRegression", "Raw+XGBoost", "DCNv2-EndToEnd", "FTTransformer-EndToEnd",
                        "TabMLPMixer-EndToEnd", "FTTransformer+XGBoost", "DCNv2+XGBoost"] if m in main0]
    st = []
    for n, v in oof_all[0].items():
        for m in refs:
            b_, c_, pm = W.mcnemar_exact(Y_ALL, v, main0[m].values)
            a1, a2, pdl = W.delong_test(Y_ALL, v, main0[m].values)
            st.append({"ModelA": n, "ModelB": m, "McNemar_b(A right,B wrong)": b_, "McNemar_c(A wrong,B right)": c_,
                       "McNemar_p_exact": pm, "AUC_A": a1, "AUC_B": a2, "DeLong_p": pdl})
    st = pd.DataFrame(st)
    if len(st):
        st["McNemar_p_Holm"], st["DeLong_p_Holm"] = W.holm(st["McNemar_p_exact"]), W.holm(st["DeLong_p"])
    st.to_csv(os.path.join(EX, "stats_extra_vs_main_cv_oof_repeat0.csv"), index=False)
    log("Extra baselines written.")


# ----------------------------------------------------------------------------- 2. feature-wise robustness
def step_robust():
    W.set_seed(args.seed)
    Xtmp, Xte, ytmp, yte = train_test_split(X_ALL, Y_ALL, test_size=W.N_TEST, stratify=Y_ALL, random_state=args.seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=W.N_VAL, stratify=ytmp, random_state=args.seed)
    log("Retraining the single-split models for the robustness tests ...")
    probs, art = W.run_fold(Xtr, ytr, Xva, yva, Xte, yte, args.seed, keep=True)
    saved = pd.read_csv(os.path.join(RES, "single_split_test_probabilities.csv"))
    same = {m: float(np.abs(saved[m].values - probs[m]).max()) for m in probs if m in saved}
    cls = max(v for k, v in same.items() if not any(k.startswith(b) for b in W.BACKBONES))
    dl = max(v for k, v in same.items() if any(k.startswith(b) for b in W.BACKBONES))
    log(f"Reproduction check (single split): classical models max |prob diff| = {cls:.2e}; deep models = {dl:.2e} "
        "(deep models retrained on another GPU/CPU can differ; the robustness test uses these retrained models)")
    json.dump(same, open(os.path.join(EX, "robust_reproduction_check.json"), "w"), indent=2)

    Xtr_p, Xva_p, Xte_p = art["Xtr"], art["Xva"], art["Xte"]
    lr = LogisticRegression(max_iter=5000).fit(Xtr_p, ytr)
    models = {"LogisticRegression": lambda X: lr.predict_proba(X)[:, 1],
              "Raw+XGBoost": lambda X: art["heads"]["Raw+XGBoost"].predict_proba(X)[:, 1]}
    for b in W.BACKBONES:
        m = art["backbones"][b]
        models[f"{b}-EndToEnd"] = (lambda X, m=m: W.e2e_proba(m, X))
        models[f"{b}+XGBoost"] = (lambda X, m=m, h=art["heads"][f"{b}+XGBoost"]: h.predict_proba(W.embed_np(m, X))[:, 1])
    rng = np.random.default_rng(args.seed)
    feats = [W.FEATS[i] for i in art["idx"]]

    # (a) Gaussian noise on one feature at a time (all other features clean)
    rows = []
    for name, fpred in models.items():
        base = roc_auc_score(yte, fpred(Xte_p))
        for j, fn in enumerate(feats):
            aucs = []
            for _ in range(a.feat_reps):
                X = Xte_p.copy(); X[:, j] += rng.normal(0, a.feat_sigma, len(X)).astype(X.dtype)
                aucs.append(roc_auc_score(yte, fpred(X)))
            rows.append({"Model": name, "Feature": fn, "AUC_clean": base, "AUC_mean": np.mean(aucs), "AUC_sd": np.std(aucs, ddof=1),
                         "AUC_drop": base - np.mean(aucs)})
    fdf = pd.DataFrame(rows); fdf.to_csv(os.path.join(EX, "robustness_single_feature_noise.csv"), index=False)
    fsum = fdf.groupby("Model").agg(AUC_clean=("AUC_clean", "first"), mean_drop=("AUC_drop", "mean"),
                                    max_drop=("AUC_drop", "max")).reset_index()
    fsum["worst_feature"] = [fdf[fdf.Model == m].sort_values("AUC_drop").iloc[-1].Feature for m in fsum.Model]
    fsum.to_csv(os.path.join(EX, "robustness_single_feature_noise_summary.csv"), index=False)

    # (b) randomly missing values, replaced by the training mean (= 0 after standardisation)
    rows = []
    for rate in [0.0] + [float(v) for v in a.miss_rates.split(",")]:
        for name, fpred in models.items():
            aucs = []
            for _ in range(a.miss_reps if rate > 0 else 1):
                X = Xte_p.copy(); X[rng.random(X.shape) < rate] = 0.0
                aucs.append(roc_auc_score(yte, fpred(X)))
            rows.append({"missing_rate": rate, "Model": name, "AUC_mean": np.mean(aucs),
                         "AUC_sd": np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0, "n_repeats": len(aucs)})
    mdf = pd.DataFrame(rows); mdf.to_csv(os.path.join(EX, "robustness_missing_values.csv"), index=False)

    fig, A = plt.subplots(figsize=(7, 5))
    for name in sorted(mdf.Model.unique()):
        d = mdf[mdf.Model == name]
        A.errorbar(d.missing_rate * 100, d.AUC_mean, yerr=d.AUC_sd, marker="o", ms=3, capsize=2,
                   ls="--" if name.endswith("EndToEnd") else "-", label=nice(name))
    A.set_xlabel("Missing feature values (%), mean-imputed"); A.set_ylabel(f"Test AUC (mean ± SD, {a.miss_reps} repeats)")
    A.legend(fontsize=7, loc="lower left"); A.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(EX, "figures", "robustness_missing_values.png")); plt.close(fig)
    fig, A = plt.subplots(figsize=(7, 5))
    fs = fsum.sort_values("max_drop")
    A.barh([nice(m) for m in fs.Model], fs.max_drop, color="tab:orange", label="worst single feature")
    A.barh([nice(m) for m in fs.Model], fs.mean_drop, color="tab:blue", label="mean over features")
    A.set_xlabel(f"AUC drop with noise (σ = {a.feat_sigma}) on one feature"); A.legend(fontsize=8); A.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(EX, "figures", "robustness_single_feature_noise.png")); plt.close(fig)
    log("Feature-wise robustness written.")


# ----------------------------------------------------------------------------- 3. Friedman / Nemenyi
def all_per_fold():
    per = pd.read_csv(os.path.join(RES, "cv_metrics_per_fold.csv"))
    ex = os.path.join(EX, "cv_metrics_per_fold_extra.csv")
    if os.path.exists(ex):
        per = pd.concat([per, pd.read_csv(ex)], ignore_index=True)
    return per


def step_stats():
    per = all_per_fold()
    out = []
    for metric in ["AUC", "Accuracy"]:
        M = per.pivot_table(index=["Repeat", "Fold"], columns="Model", values=metric).dropna(axis=1)
        n, k = M.shape
        chi2, pval = stats.friedmanchisquare(*[M[c].values for c in M.columns])
        ranks = M.rank(axis=1, ascending=False).mean().sort_values()
        q = stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        cd = q * np.sqrt(k * (k + 1) / (6 * n))
        best = ranks.index[0]
        for m, rk in ranks.items():
            out.append({"Metric": metric, "Model": m, "AverageRank": rk, "RankDiffToBest": rk - ranks.iloc[0],
                        "DiffersFromBest(Nemenyi)": bool(rk - ranks.iloc[0] > cd), "Best": best,
                        "Friedman_chi2": chi2, "Friedman_p": pval, "Nemenyi_CD": cd, "n_folds": n, "n_models": k})
        log(f"Friedman ({metric}): chi2 = {chi2:.1f}, p = {pval:.2e}; Nemenyi CD = {cd:.2f} ranks ({k} models, {n} folds)")
        fig, A = plt.subplots(figsize=(7, 0.28 * k + 1.2))
        y = np.arange(k)[::-1]
        A.scatter(ranks.values, y, color="tab:blue", zorder=3)
        A.set_yticks(y); A.set_yticklabels([nice(m) for m in ranks.index], fontsize=8)
        A.axvspan(ranks.iloc[0], ranks.iloc[0] + cd, color="tab:green", alpha=0.15, label=f"within CD of best ({cd:.2f})")
        A.set_xlabel(f"Average rank of {metric} over {n} CV folds (1 = best)"); A.grid(axis="x", alpha=0.3)
        A.set_title(f"Friedman p = {pval:.1e}; Nemenyi critical difference = {cd:.2f}", fontsize=9)
        A.legend(fontsize=8, loc="lower right"); fig.tight_layout()
        fig.savefig(os.path.join(EX, "figures", f"friedman_nemenyi_{metric}.png")); plt.close(fig)
    pd.DataFrame(out).to_csv(os.path.join(EX, "friedman_nemenyi.csv"), index=False)
    log("Friedman / Nemenyi written. Note: CV folds overlap, so this test is approximate.")


# ----------------------------------------------------------------------------- 4. decision curves
def step_dca():
    mains = sorted(glob.glob(os.path.join(RES, "cv_oof_probabilities_repeat*.csv")))
    want = ["LogisticRegression", "Raw+XGBoost", "DCNv2-EndToEnd", "FTTransformer-EndToEnd",
            "TabMLPMixer-EndToEnd", "FTTransformer+XGBoost", "DCNv2+XGBoost", "CatBoost", "TabPFN"]
    th = np.round(np.arange(0.01, 0.61, 0.01), 2)
    rows = []
    for r, fmain in enumerate(mains):
        d = pd.read_csv(fmain)
        fe = os.path.join(EX, f"cv_oof_probabilities_extra_repeat{r}.csv")
        if os.path.exists(fe):
            d = pd.concat([d, pd.read_csv(fe).drop(columns="y_true")], axis=1)
        y = d["y_true"].values; n = len(y); prev = y.mean()
        for pt in th:
            w = pt / (1 - pt)
            rows.append({"Repeat": r, "threshold": pt, "Model": "Treat all", "NetBenefit": prev - (1 - prev) * w})
            rows.append({"Repeat": r, "threshold": pt, "Model": "Treat none", "NetBenefit": 0.0})
            for m in [c for c in want if c in d]:
                pos = d[m].values >= pt
                tp, fp = np.sum(pos & (y == 1)), np.sum(pos & (y == 0))
                rows.append({"Repeat": r, "threshold": pt, "Model": m, "NetBenefit": tp / n - fp / n * w})
    dca = pd.DataFrame(rows).groupby(["Model", "threshold"]).NetBenefit.agg(["mean", "std"]).reset_index()
    dca.to_csv(os.path.join(EX, "decision_curve_net_benefit.csv"), index=False)
    fig, A = plt.subplots(figsize=(7, 5))
    for m in [x for x in want if x in set(dca.Model)] + ["Treat all", "Treat none"]:
        d = dca[dca.Model == m]
        A.plot(d.threshold, d["mean"], label=nice(m), lw=1.2 if m.startswith("Treat") else 1.5,
               ls=":" if m.startswith("Treat") else ("--" if m.endswith("EndToEnd") else "-"),
               color="black" if m == "Treat none" else ("grey" if m == "Treat all" else None))
    prev = pd.read_csv(mains[0])["y_true"].mean()
    A.set_ylim(-0.05, prev + 0.03); A.set_xlabel("Threshold probability"); A.set_ylabel("Net benefit")
    A.set_title(f"Decision curves (pooled out-of-fold predictions, mean over {len(mains)} CV repeats)", fontsize=9)
    A.legend(fontsize=7); A.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(EX, "figures", "decision_curve.png")); plt.close(fig)
    log("Decision curves written.")


if __name__ == "__main__":
    t0 = time.time()
    log(f"=== extra experiments: {W.DS_NAME}, results in {EX} | steps: {STEPS}" + (" | QUICK MODE (not valid)" if a.quick else ""))
    for s, fn in [("baselines", step_baselines), ("robust", step_robust), ("stats", step_stats), ("dca", step_dca)]:
        if s in STEPS:
            fn()
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min. Results in: {EX}/")
