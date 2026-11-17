#!/usr/bin/env python3
"""
Further experiments asked for in the review (they do NOT change the main results).
Everything is written to <res>/more/.

  seeds    : the same single 70/15/15 split, deep backbones and heads retrained with 5 weight seeds
  embsize  : embedding size m in {8, 16, 32, 64, 128}, repeated CV
  finetune : frozen backbone vs fine-tuned backbone (both with the same small MLP head), repeated CV
  tuning   : equal tuning budget - every model family gets the same number of random-search trials,
             selected on the same inner validation split (deep backbones, XGBoost, LightGBM, CatBoost, LR)

The CV folds are the first --repeats repeats of the main 10 x 5 CV (same seeds), so they are directly
comparable with the main results. Table 2 settings are used unless a step changes one setting on purpose.

  python more_experiments.py --dataset wdbc --res results_v3
  python more_experiments.py --dataset wbco --res results_wbco
  python more_experiments.py --dataset coimbra --res results_coimbra
  options: --steps seeds,embsize,finetune,tuning  --repeats 3  --trials 10  --quick (smoke test only)
If it stops, run the same command again: finished folds are reloaded from <res>/more/cache/.
"""
import argparse, os, sys, time, warnings
from copy import deepcopy

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="wdbc", choices=["wdbc", "wbco", "coimbra"])
p.add_argument("--res", default=None)
p.add_argument("--steps", default="seeds,embsize,finetune,tuning")
p.add_argument("--repeats", type=int, default=3, help="CV repeats (x 5 folds) for embsize/finetune/tuning")
p.add_argument("--seeds", default="42,43,44,45,46")
p.add_argument("--emb_grid", default="8,16,32,64,128")
p.add_argument("--trials", type=int, default=10, help="random-search trials per model family (equal budget)")
p.add_argument("--quick", action="store_true", help="smoke test; numbers are NOT valid")
a = p.parse_args()
RES = a.res or {"wdbc": "results_v3", "wbco": "results_wbco", "coimbra": "results_coimbra"}[a.dataset]
OUTD = os.path.join(RES, "more_quick" if a.quick else "more")
os.makedirs(os.path.join(OUTD, "cache"), exist_ok=True)
STEPS = [s.strip() for s in a.steps.split(",")]

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv = sys.argv
sys.argv = [_argv[0], "--dataset", a.dataset, "--out", OUTD] + (["--quick"] if a.quick else [])
import wdbc_rerun as W                                                # noqa: E402
sys.argv = _argv

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
import torch.nn.functional as F                                       # noqa: E402
from scipy import stats                                               # noqa: E402
from sklearn.linear_model import LogisticRegression                   # noqa: E402
from sklearn.metrics import roc_auc_score, silhouette_score           # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split # noqa: E402
from sklearn.preprocessing import StandardScaler                      # noqa: E402

warnings.filterwarnings("ignore")
args = W.args
X_ALL, Y_ALL = W.X_ALL, W.Y_ALL
BASE = {k: getattr(args, k) for k in ["lr", "dropout", "emb", "wd"]}
REPEATS = 1 if a.quick else a.repeats
T0 = time.time()


def log(*x):
    msg = " ".join(str(v) for v in x)
    print(msg, flush=True)
    with open(os.path.join(OUTD, "more_log.txt"), "a") as fh:
        fh.write(msg + "\n")


W.log = log


def reset_args():
    for k, v in BASE.items():
        setattr(args, k, v)


def folds():
    """Same folds and seeds as the first REPEATS repeats of the main CV."""
    for r in range(REPEATS):
        skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed + r)
        for f, (tr, te) in enumerate(skf.split(X_ALL, Y_ALL)):
            s = args.seed + 1000 * r + f
            Xa, Xv, ya, yv = train_test_split(X_ALL[tr], Y_ALL[tr], test_size=0.15, stratify=Y_ALL[tr], random_state=s)
            yield r, f, s, Xa, ya, Xv, yv, X_ALL[te], Y_ALL[te]


def cached(name, fn):
    path = os.path.join(OUTD, "cache", name + ".pkl")
    if os.path.exists(path):
        return pd.read_pickle(path)
    out = fn(); pd.to_pickle(out, path)
    return out


def met(y, pr):
    return {"AUC": roc_auc_score(y, pr), "Accuracy": float(np.mean((pr >= 0.5) == y))}


def heads_on(m, Xtr, ytr, Xva, yva, Xte, seed, b):
    out = {f"{b}-EndToEnd": W.e2e_proba(m, Xte)}
    Ztr, Zva, Zte = W.embed_np(m, Xtr), W.embed_np(m, Xva), W.embed_np(m, Xte)
    for h in W.HEADS:
        mdl = W.tuned_knn(Ztr, ytr, Zva, yva)[0] if h == "kNN" else W.make_head(h, seed).fit(Ztr, ytr)
        out[f"{b}+{h}"] = mdl.predict_proba(Zte)[:, 1]
    return out, Zte


def nb_test(d, ratio=0.25):
    """Nadeau-Bengio corrected resampled t-test of the fold-wise differences d."""
    J, v = len(d), np.var(d, ddof=1)
    t = np.mean(d) / np.sqrt((1 / J + ratio) * v) if v > 0 else 0.0
    return t, 2 * stats.t.sf(abs(t), J - 1)


def summary(df, by):
    g = df.groupby(by)
    return g.agg(AUC_mean=("AUC", "mean"), AUC_sd=("AUC", "std"),
                 Accuracy_mean=("Accuracy", "mean"), Accuracy_sd=("Accuracy", "std"), n=("AUC", "size")).reset_index()


# ----------------------------------------------------------------------------- 1. seeds
def step_seeds():
    W.set_seed(args.seed)
    Xtmp, Xte, ytmp, yte = train_test_split(X_ALL, Y_ALL, test_size=W.N_TEST, stratify=Y_ALL, random_state=args.seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=W.N_VAL, stratify=ytmp, random_state=args.seed)
    Xtr_p, Xva_p, Xte_p, _, _ = W.preprocess(Xtr, ytr, Xva, yva, Xte, args.seed)   # fixed preprocessing
    rows = []
    for s in [int(v) for v in a.seeds.split(",")]:
        def run():
            out = {}
            for b in W.BACKBONES:
                m, _, _ = W.train_backbone(b, Xtr_p, ytr, Xva_p, yva, s)
                out.update(heads_on(m, Xtr_p, ytr, Xva_p, yva, Xte_p, s, b)[0]); del m
            return out
        pr = cached(f"seeds_{s}", run)
        for k, v in pr.items():
            rows.append({"Seed": s, "Model": k, **met(yte, v)})
        log(f"[seeds] seed {s} done ({(time.time() - T0) / 60:.1f} min)")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUTD, "seeds_per_run.csv"), index=False)
    sm = summary(df, "Model")
    sm["AUC_min"] = df.groupby("Model").AUC.min().values; sm["AUC_max"] = df.groupby("Model").AUC.max().values
    sm.to_csv(os.path.join(OUTD, "seeds_summary.csv"), index=False)
    log("[seeds] written: seeds_per_run.csv, seeds_summary.csv")


# ----------------------------------------------------------------------------- 2. embedding size
def step_embsize():
    rows = []
    for mdim in [int(v) for v in a.emb_grid.split(",")]:
        for r, f, s, Xa, ya, Xv, yv, Xt, yt in folds():
            def run():
                args.emb = mdim
                Xtr_p, Xva_p, Xte_p, _, _ = W.preprocess(Xa, ya, Xv, yv, Xt, s)
                out, sil = {}, {}
                for b in W.BACKBONES:
                    m, _, _ = W.train_backbone(b, Xtr_p, ya, Xva_p, yv, s)
                    o, Zte = heads_on(m, Xtr_p, ya, Xva_p, yv, Xte_p, s, b); out.update(o)
                    try:
                        sil[b] = silhouette_score(StandardScaler().fit_transform(Zte), yt)
                    except Exception:
                        sil[b] = np.nan
                    del m
                reset_args()
                return out, sil
            pr, sil = cached(f"emb{mdim}_r{r}_f{f}", run)
            for k, v in pr.items():
                b = k.split("-")[0].split("+")[0]
                rows.append({"m": mdim, "Repeat": r, "Fold": f, "Model": k, **met(yt, v), "Silhouette_test": sil.get(b, np.nan)})
        log(f"[embsize] m = {mdim} done ({(time.time() - T0) / 60:.1f} min)")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUTD, "embsize_per_fold.csv"), index=False)
    sm = summary(df, ["m", "Model"])
    sm["Silhouette_mean"] = df.groupby(["m", "Model"]).Silhouette_test.mean().values
    sm.to_csv(os.path.join(OUTD, "embsize_summary.csv"), index=False)
    log("[embsize] written: embsize_per_fold.csv, embsize_summary.csv")


# ----------------------------------------------------------------------------- 3. frozen vs fine-tuned
def train_mlp(backbone, Xtr, ytr, Xva, yva, finetune, seed):
    W.set_seed(seed)
    d = W.embed_np(backbone, Xva[:2]).shape[1]
    head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(64, 2)).to(W.DEV)
    groups = [{"params": head.parameters(), "lr": 1e-3}]
    if finetune:
        groups.append({"params": backbone.parameters(), "lr": args.lr})
    opt = torch.optim.Adam(groups, weight_decay=args.wd)
    xt, yt_ = torch.tensor(Xtr, device=W.DEV), torch.tensor(ytr, device=W.DEV)
    xv, yv_ = torch.tensor(Xva, device=W.DEV), torch.tensor(yva, device=W.DEV)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, state, wait = np.inf, None, 0
    for ep in range(args.epochs):
        backbone.train(finetune); head.train()
        perm = torch.randperm(len(xt), generator=g).to(W.DEV)
        for i in range(0, len(xt), args.batch):
            bidx = perm[i:i + args.batch]
            if len(bidx) < 2:
                continue
            if finetune:
                z, aux = backbone.embed(xt[bidx])
            else:
                with torch.no_grad():
                    z, aux = backbone.embed(xt[bidx])
                aux = 0.0
            loss = F.cross_entropy(head(z), yt_[bidx]) + aux
            opt.zero_grad(); loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(list(head.parameters()) + (list(backbone.parameters()) if finetune else []), args.clip)
            opt.step()
        backbone.eval(); head.eval()
        with torch.no_grad():
            zv, _ = backbone.embed(xv); vl = F.cross_entropy(head(zv), yv_).item()
        if vl < best - 1e-6:
            best, wait = vl, 0
            state = (deepcopy(head.state_dict()), deepcopy(backbone.state_dict()) if finetune else None)
        else:
            wait += 1
            if wait >= args.patience:
                break
    head.load_state_dict(state[0])
    if finetune:
        backbone.load_state_dict(state[1])
    backbone.eval(); head.eval()

    @torch.no_grad()
    def predict(X):
        z, _ = backbone.embed(torch.tensor(X, device=W.DEV))
        return F.softmax(head(z), 1)[:, 1].cpu().numpy()
    return predict


def step_finetune():
    rows = []
    for r, f, s, Xa, ya, Xv, yv, Xt, yt in folds():
        def run():
            Xtr_p, Xva_p, Xte_p, _, _ = W.preprocess(Xa, ya, Xv, yv, Xt, s)
            out = {}
            for b in W.BACKBONES:
                m, _, _ = W.train_backbone(b, Xtr_p, ya, Xva_p, yv, s)
                out[f"{b}|End-to-end (softmax)"] = W.e2e_proba(m, Xte_p)
                out[f"{b}|Frozen + XGBoost"] = W.make_head("XGBoost", s).fit(W.embed_np(m, Xtr_p), ya).predict_proba(W.embed_np(m, Xte_p))[:, 1]
                out[f"{b}|Frozen + MLP head"] = train_mlp(m, Xtr_p, ya, Xva_p, yv, False, s)(Xte_p)
                out[f"{b}|Fine-tuned + MLP head"] = train_mlp(deepcopy(m), Xtr_p, ya, Xva_p, yv, True, s)(Xte_p)
                del m
            return out
        pr = cached(f"finetune_r{r}_f{f}", run)
        for k, v in pr.items():
            b, setting = k.split("|")
            rows.append({"Backbone": b, "Setting": setting, "Repeat": r, "Fold": f, **met(yt, v)})
        log(f"[finetune] repeat {r + 1} fold {f + 1} done ({(time.time() - T0) / 60:.1f} min)")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUTD, "finetune_per_fold.csv"), index=False)
    summary(df, ["Backbone", "Setting"]).to_csv(os.path.join(OUTD, "finetune_summary.csv"), index=False)
    tests = []
    for b in W.BACKBONES:
        d0 = df[df.Backbone == b].pivot_table(index=["Repeat", "Fold"], columns="Setting", values="AUC")
        for other in ["Frozen + MLP head", "End-to-end (softmax)", "Frozen + XGBoost"]:
            t, pv = nb_test((d0["Fine-tuned + MLP head"] - d0[other]).values)
            tests.append({"Backbone": b, "Comparison": f"Fine-tuned + MLP head vs {other}",
                          "MeanDiff_AUC": (d0["Fine-tuned + MLP head"] - d0[other]).mean(), "t_corrected": t, "p": pv})
    tests = pd.DataFrame(tests); tests["p_Holm"] = W.holm(tests["p"])
    tests.to_csv(os.path.join(OUTD, "finetune_tests.csv"), index=False)
    log("[finetune] written: finetune_per_fold.csv, finetune_summary.csv, finetune_tests.csv")


# ----------------------------------------------------------------------------- 4. equal tuning budget
def sample(space, n, rng):
    return [{k: v[rng.integers(len(v))] for k, v in space.items()} for _ in range(n)]


DEEP_SPACE = {"lr": [1e-4, 3e-4, 1e-3], "dropout": [0.1, 0.3, 0.5], "emb": [64, 128], "wd": [1e-5, 1e-4]}
XGB_SPACE = {"n_estimators": [100, 300, 600], "max_depth": [3, 4, 6, 8], "learning_rate": [0.01, 0.05, 0.1], "subsample": [0.7, 1.0]}
LGB_SPACE = {"n_estimators": [100, 300, 600], "num_leaves": [7, 15, 31], "learning_rate": [0.01, 0.05, 0.1], "min_child_samples": [5, 10, 20]}
CB_SPACE = {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1], "l2_leaf_reg": [1, 3, 10]}
LR_C = [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]


def step_tuning():
    import xgboost as xgb
    rows, chosen = [], []
    for r, f, s, Xa, ya, Xv, yv, Xt, yt in folds():
        def run():
            rng = np.random.default_rng(s)
            Xtr_p, Xva_p, Xte_p, _, _ = W.preprocess(Xa, ya, Xv, yv, Xt, s)
            out, ch = {}, {}
            for b in W.BACKBONES:                                   # deep: same budget of trials
                best = None
                for cfg in sample(DEEP_SPACE, a.trials, rng):
                    for k, v in cfg.items():
                        setattr(args, k, v)
                    m, _, _ = W.train_backbone(b, Xtr_p, ya, Xva_p, yv, s)
                    sc = roc_auc_score(yv, W.e2e_proba(m, Xva_p))
                    if best is None or sc > best[0]:
                        best = (sc, cfg, m)
                    else:
                        del m
                    reset_args()
                _, cfg, m = best
                out[f"{b}-EndToEnd (tuned)"] = W.e2e_proba(m, Xte_p)
                out[f"{b}+XGBoost (tuned backbone)"] = W.make_head("XGBoost", s).fit(W.embed_np(m, Xtr_p), ya).predict_proba(W.embed_np(m, Xte_p))[:, 1]
                ch[b] = cfg; del m, best
            fams = {"XGBoost (tuned)": (XGB_SPACE, lambda c: xgb.XGBClassifier(**c, tree_method="hist", eval_metric="logloss",
                                                                              random_state=s, device="cuda" if W.GPU else "cpu"))}
            try:
                import lightgbm as lgb
                fams["LightGBM (tuned)"] = (LGB_SPACE, lambda c: lgb.LGBMClassifier(**c, random_state=s, verbose=-1))
            except ImportError:
                pass
            try:
                from catboost import CatBoostClassifier
                fams["CatBoost (tuned)"] = (CB_SPACE, lambda c: CatBoostClassifier(**c, iterations=200 if a.quick else 1000,
                                            random_seed=s, verbose=0, allow_writing_files=False))
            except ImportError:
                pass
            for name, (space, make) in fams.items():
                best = None
                for cfg in sample(space, a.trials, rng):
                    mdl = make(cfg)
                    if name.startswith("CatBoost"):
                        mdl.fit(Xtr_p, ya, eval_set=(Xva_p, yv), early_stopping_rounds=50, use_best_model=True)
                    else:
                        mdl.fit(Xtr_p, ya)
                    sc = roc_auc_score(yv, mdl.predict_proba(Xva_p)[:, 1])
                    if best is None or sc > best[0]:
                        best = (sc, cfg, mdl)
                out[name] = best[2].predict_proba(Xte_p)[:, 1]; ch[name] = best[1]
            best = None
            for C in LR_C[:a.trials]:
                mdl = LogisticRegression(C=C, max_iter=5000).fit(Xtr_p, ya)
                sc = roc_auc_score(yv, mdl.predict_proba(Xva_p)[:, 1])
                if best is None or sc > best[0]:
                    best = (sc, C, mdl)
            out["LogisticRegression (tuned)"] = best[2].predict_proba(Xte_p)[:, 1]; ch["LogisticRegression"] = {"C": best[1]}
            return out, ch
        pr, ch = cached(f"tuning_r{r}_f{f}", run)
        for k, v in pr.items():
            rows.append({"Model": k, "Repeat": r, "Fold": f, **met(yt, v)})
        chosen.append({"Repeat": r, "Fold": f, **{k: str(v) for k, v in ch.items()}})
        log(f"[tuning] repeat {r + 1} fold {f + 1} done ({(time.time() - T0) / 60:.1f} min)")
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUTD, "tuning_per_fold.csv"), index=False)
    summary(df, "Model").sort_values("AUC_mean", ascending=False).to_csv(os.path.join(OUTD, "tuning_summary.csv"), index=False)
    pd.DataFrame(chosen).to_csv(os.path.join(OUTD, "tuning_chosen_configs.csv"), index=False)
    # tuned hybrid vs tuned end-to-end, and best tuned deep vs tuned GBDTs
    P = df.pivot_table(index=["Repeat", "Fold"], columns="Model", values="AUC")
    tests = []
    for b in W.BACKBONES:
        t, pv = nb_test((P[f"{b}+XGBoost (tuned backbone)"] - P[f"{b}-EndToEnd (tuned)"]).values)
        tests.append({"A": f"{b}+XGBoost (tuned backbone)", "B": f"{b}-EndToEnd (tuned)",
                      "MeanDiff_AUC": (P[f"{b}+XGBoost (tuned backbone)"] - P[f"{b}-EndToEnd (tuned)"]).mean(), "t_corrected": t, "p": pv})
        for g in [c for c in P.columns if c.startswith(("XGBoost", "LightGBM", "CatBoost", "Logistic"))]:
            t, pv = nb_test((P[f"{b}-EndToEnd (tuned)"] - P[g]).values)
            tests.append({"A": f"{b}-EndToEnd (tuned)", "B": g, "MeanDiff_AUC": (P[f"{b}-EndToEnd (tuned)"] - P[g]).mean(),
                          "t_corrected": t, "p": pv})
    tests = pd.DataFrame(tests); tests["p_Holm"] = W.holm(tests["p"])
    tests.to_csv(os.path.join(OUTD, "tuning_tests.csv"), index=False)
    log("[tuning] written: tuning_per_fold.csv, tuning_summary.csv, tuning_chosen_configs.csv, tuning_tests.csv")


if __name__ == "__main__":
    log(f"=== more experiments: {W.DS_NAME}, results in {OUTD} | steps: {STEPS} | CV repeats: {REPEATS} x {args.folds}"
        f" | trials: {a.trials}" + (" | QUICK MODE (not valid)" if a.quick else ""))
    for s_, fn in [("seeds", step_seeds), ("embsize", step_embsize), ("finetune", step_finetune), ("tuning", step_tuning)]:
        if s_ in STEPS:
            fn(); reset_args()
    log(f"ALL DONE in {(time.time() - T0) / 60:.1f} min. Results in: {OUTD}/")
