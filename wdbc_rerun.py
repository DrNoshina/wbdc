#!/usr/bin/env python3
"""
Full re-run pipeline for the WDBC representation-decision decoupling paper.

What it does (all numbers are computed from predictions, so impossible values cannot occur):
  1. Single stratified split (70/15/15, 86 test samples) -> test metrics, confusion matrices,
     calibration (Brier, ECE, slope, intercept), exact McNemar + DeLong tests,
     embedding geometry (raw features vs embeddings), noise robustness (sigma up to 1.0,
     100 repeats), inference latency, training curves, SHAP/LIME + rank agreement.
  2. Repeated stratified k-fold CV (default 10 x 5) for EVERY model with the same protocol
     -> mean +/- SD and 95% CI, plus McNemar/DeLong on pooled out-of-fold predictions.

Models
  Backbones (trained end-to-end with a temporary softmax head, then frozen):
      FTTransformer, DCNv2, TabNet, TabMLPMixer           -> "<B>-EndToEnd"
  Hybrid heads on frozen 128-d embeddings: XGBoost, ExtraTrees, kNN -> "<B>+<Head>"
  Ablation (same heads on raw features): Raw+XGBoost, Raw+ExtraTrees, Raw+kNN
  Classical baselines: LogisticRegression, RandomForest, SVM-RBF, HistGB, LightGBM, TabPFN (optional)

Version 3: deep-model settings follow Table 2 of the paper exactly (Adam, lr 1e-4, batch 32, max 100 epochs,
patience 10, dropout 0.3, 4 attention heads, gradient clipping 1.0, weight decay 1e-5, latent dim 128).
kNN (k, weighting) is chosen on the VALIDATION set inside every split (k is not fixed in the paper);
TabNet width tuning is optional (--tune_tabnet).
latency also reported for full-batch and CPU-XGBoost; pooled out-of-fold calibration added.

Usage
  python wdbc_rerun.py                    # full run on GPU (about 1-3 h on one NVIDIA GPU)
  python wdbc_rerun.py --repeats 5        # faster CV
  python wdbc_rerun.py --quick            # smoke test (minutes, numbers NOT for the paper)
  python wdbc_rerun.py --dataset wbco --out results_wbco   # second dataset: Wisconsin Breast Cancer (Original)
  If the run stops, start the same command again: finished CV folds are loaded from results/cv_cache/.
"""
import argparse, json, os, platform, random, time, warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             brier_score_loss, confusion_matrix, roc_curve, silhouette_score,
                             davies_bouldin_score, calinski_harabasz_score, log_loss)
from scipy import stats
import xgboost as xgb

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------- config
p = argparse.ArgumentParser()
p.add_argument("--out", default="results")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--repeats", type=int, default=10)
p.add_argument("--folds", type=int, default=5)
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--patience", type=int, default=10)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--wd", type=float, default=1e-5)
p.add_argument("--emb", type=int, default=128, help="embedding dimension m")
p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"])
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--heads", type=int, default=4, help="attention heads (FT-Transformer)")
p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clipping (0 = off)")
p.add_argument("--knn_k", type=int, default=5, help="used only if --no_tune")
p.add_argument("--knn_grid", default="3,5,7,9,11,15", help="kNN k values tried on validation")
p.add_argument("--tabnet_grid", default="8,16,32,64", help="TabNet widths tried on validation (only with --tune_tabnet)")
p.add_argument("--tune_tabnet", action="store_true", help="tune TabNet width on validation (paper default: fixed 128)")
p.add_argument("--no_tune", action="store_true", help="disable validation tuning (reproduces v1)")
p.add_argument("--k_grid", default=None, help="MI feature counts tried on validation (default: 10,15,20,25,30 for WDBC; 3,5,7,9 for WBCO)")
p.add_argument("--dataset", default="wdbc", choices=["wdbc", "wbco", "coimbra"], help="wdbc = Diagnostic (569x30); wbco = Original (699x9); coimbra = Breast Cancer Coimbra (116x9)")
p.add_argument("--data_file", default=None, help="path to breast-cancer-wisconsin.data (WBCO); default: next to this script")
p.add_argument("--robust_reps", type=int, default=100)
p.add_argument("--sigmas", default="0,0.01,0.03,0.05,0.1,0.2,0.3,0.5,0.75,1.0")
p.add_argument("--xai_model", default="FTTransformer+XGBoost")
p.add_argument("--tabpfn", action="store_true", help="include TabPFN (needs licence/API key once)")
p.add_argument("--skip_tabpfn", action="store_true", help="kept for compatibility")
p.add_argument("--skip_xai", action="store_true")
p.add_argument("--skip_cv", action="store_true")
p.add_argument("--quick", action="store_true", help="tiny smoke test; numbers are NOT valid")
args = p.parse_args()

if args.quick:
    args.repeats, args.folds, args.epochs, args.robust_reps = 1, 3, 5, 5
    args.k_grid = args.k_grid or ("15,30" if args.dataset == "wdbc" else "5,9"); args.tabnet_grid = "8,32"; args.knn_grid = "5,9"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU = DEV.type == "cuda"
OUT = args.out
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
if args.k_grid is None:
    args.k_grid = "10,15,20,25,30" if args.dataset == "wdbc" else "3,5,7,9"
K_GRID = [int(k) for k in args.k_grid.split(",")]
SIGMAS = [float(s) for s in args.sigmas.split(",")]
BACKBONES = ["FTTransformer", "DCNv2", "TabNet", "TabMLPMixer"]
HEADS = ["XGBoost", "ExtraTrees", "kNN"]


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(os.path.join(OUT, "run_log.txt"), "a") as f:
        f.write(msg + "\n")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if GPU:
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------- data
if args.dataset == "wdbc":
    data = load_breast_cancer()          # identical to UCI WDBC: 569 samples x 30 features
    X_ALL = data.data.astype(np.float32)
    Y_ALL = (data.target == 0).astype(int)   # sklearn: 0 = malignant -> recode malignant = 1 (positive)
    FEATS = list(data.feature_names)
    DS_NAME = "WDBC"
elif args.dataset == "coimbra":
    # UCI Breast Cancer Coimbra (id 451): 116 women x 9 routine blood/anthropometric variables,
    # Classification 1 = healthy control, 2 = breast cancer patient (positive). No missing values.
    here = os.path.dirname(os.path.abspath(__file__))
    fp = args.data_file or next((c for c in [os.path.join(here, "dataR2.csv"), os.path.join(here, "data", "dataR2.csv")]
                                 if os.path.exists(c)), os.path.join(here, "dataR2.csv"))
    df = pd.read_csv(fp)
    assert len(df) == 116, f"expected 116 rows, got {len(df)}"
    FEATS = [c for c in df.columns if c != "Classification"]
    X_ALL = df[FEATS].to_numpy(dtype=np.float32)
    Y_ALL = (df["Classification"] == 2).astype(int).to_numpy()
    DS_NAME = "Coimbra"
else:
    # UCI Breast Cancer Wisconsin (Original), id 15: 699 samples x 9 cytology features (1-10),
    # 16 missing Bare Nuclei values ('?') -> imputed with the TRAINING median inside every split.
    here = os.path.dirname(os.path.abspath(__file__))
    fp = args.data_file or next((c for c in [os.path.join(here, "breast-cancer-wisconsin.data"),
                                             os.path.join(here, "data", "breast-cancer-wisconsin.data")]
                                 if os.path.exists(c)), os.path.join(here, "breast-cancer-wisconsin.data"))
    cols = ["id", "Clump_Thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape", "Marginal_Adhesion",
            "Single_Epithelial_Cell_Size", "Bare_Nuclei", "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class"]
    df = pd.read_csv(fp, header=None, names=cols, na_values="?")
    assert len(df) == 699, f"expected 699 rows, got {len(df)}"
    FEATS = cols[1:-1]
    X_ALL = df[FEATS].to_numpy(dtype=np.float32)
    Y_ALL = (df["Class"] == 4).astype(int).to_numpy()   # 4 = malignant (positive), 2 = benign
    DS_NAME = "WBCO"
N_ALL = len(Y_ALL)
N_TEST = 86 if args.dataset == "wdbc" else int(round(0.15 * N_ALL))
N_VAL = 85 if args.dataset == "wdbc" else int(round(0.15 * N_ALL))


def preprocess(Xtr, ytr, Xva, yva, Xte, seed):
    """Scaler + MI ranking fitted on the training part only; k chosen by validation AUC (LR)."""
    med = np.nanmedian(Xtr, axis=0)                        # no-op for WDBC (no missing values)
    Xtr, Xva, Xte = [np.where(np.isnan(a), med, a) for a in (Xtr, Xva, Xte)]
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    mi = mutual_info_classif(Xtr_s, ytr, random_state=seed)
    order = np.argsort(-mi)
    best_k, best_auc = K_GRID[-1], -1
    for k in K_GRID:
        idx = order[:k]
        lr = LogisticRegression(max_iter=5000).fit(Xtr_s[:, idx], ytr)
        auc = roc_auc_score(yva, lr.predict_proba(Xva_s[:, idx])[:, 1])
        if auc > best_auc + 1e-9:
            best_k, best_auc = k, auc
    idx = np.sort(order[:best_k])
    return (Xtr_s[:, idx].astype(np.float32), Xva_s[:, idx].astype(np.float32),
            Xte_s[:, idx].astype(np.float32), idx, sc)


# ----------------------------------------------------------------------------- backbones
class FeatureTokenizer(nn.Module):
    def __init__(self, d_in, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(d_in, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(d_in, d))

    def forward(self, x):
        return x.unsqueeze(-1) * self.w + self.b


class FTTransformer(nn.Module):
    def __init__(self, d_in, emb=128, layers=3, heads=4, drop=0.3):
        super().__init__()
        self.tok = FeatureTokenizer(d_in, emb)
        self.cls = nn.Parameter(torch.zeros(1, 1, emb))
        enc = nn.TransformerEncoderLayer(emb, heads, dim_feedforward=2 * emb, dropout=drop,
                                         batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(emb)
        self.head = nn.Linear(emb, 2)

    def embed(self, x):
        t = torch.cat([self.cls.expand(x.size(0), -1, -1), self.tok(x)], 1)
        return F.relu(self.norm(self.enc(t)[:, 0])), 0.0


class DCNv2(nn.Module):
    def __init__(self, d_in, emb=128, n_cross=3, drop=0.3):
        super().__init__()
        self.cross = nn.ModuleList([nn.Linear(d_in, d_in) for _ in range(n_cross)])
        self.deep = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(256, emb), nn.ReLU())
        self.proj = nn.Sequential(nn.Linear(d_in + emb, emb), nn.ReLU())
        self.head = nn.Linear(emb, 2)

    def embed(self, x):
        x0, xl = x, x
        for lin in self.cross:
            xl = x0 * lin(xl) + xl
        return self.proj(torch.cat([xl, self.deep(x)], 1)), 0.0


class MixerBlock(nn.Module):
    def __init__(self, n_tok, d, drop=0.3):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.tok = nn.Sequential(nn.Linear(n_tok, 2 * n_tok), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * n_tok, n_tok))
        self.ch = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.tok(self.n1(x).transpose(1, 2)).transpose(1, 2)
        return x + self.ch(self.n2(x))


class TabMLPMixer(nn.Module):
    def __init__(self, d_in, emb=128, blocks=4, drop=0.3):
        super().__init__()
        self.tok = FeatureTokenizer(d_in, emb)
        self.blocks = nn.Sequential(*[MixerBlock(d_in, emb, drop) for _ in range(blocks)])
        self.norm = nn.LayerNorm(emb)
        self.head = nn.Linear(emb, 2)

    def embed(self, x):
        return F.relu(self.norm(self.blocks(self.tok(x)).mean(1))), 0.0


class TabNetBackbone(nn.Module):
    """pytorch-tabnet network; embedding = summed decision-step outputs (n_d = emb)."""
    def __init__(self, d_in, emb=128):
        super().__init__()
        from pytorch_tabnet.tab_network import TabNet
        from pytorch_tabnet.utils import create_group_matrix
        self.net = TabNet(d_in, 2, n_d=emb, n_a=emb, n_steps=3, gamma=1.3, virtual_batch_size=32,
                          momentum=0.02, mask_type="sparsemax",
                          group_attention_matrix=create_group_matrix([], d_in).to(DEV))
        self.head = self.net.tabnet.final_mapping
        self.lambda_sparse = 1e-3

    def embed(self, x):
        steps, m_loss = self.net.tabnet.encoder(self.net.embedder(x))
        return torch.sum(torch.stack(steps, 0), 0), -self.lambda_sparse * m_loss


def make_backbone(name, d_in, width=None):
    if name == "FTTransformer":
        return FTTransformer(d_in, args.emb, heads=args.heads, drop=args.dropout)
    if name == "DCNv2":
        return DCNv2(d_in, args.emb, drop=args.dropout)
    if name == "TabMLPMixer":
        return TabMLPMixer(d_in, args.emb, drop=args.dropout)
    return TabNetBackbone(d_in, width or args.emb)


def train_backbone(name, Xtr, ytr, Xva, yva, seed, width=None):
    """Stage 1: train backbone + temporary softmax head with cross-entropy; early stop on val loss."""
    set_seed(seed)
    m = make_backbone(name, Xtr.shape[1], width).to(DEV)
    Opt = torch.optim.Adam if args.optimizer == "adam" else torch.optim.AdamW
    opt = Opt(m.parameters(), lr=args.lr, weight_decay=args.wd)
    xt, yt = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, device=DEV)
    xv, yv = torch.tensor(Xva, device=DEV), torch.tensor(yva, device=DEV)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, wait, curve = np.inf, None, 0, []
    for ep in range(args.epochs):
        m.train()
        perm = torch.randperm(len(xt), generator=g).to(DEV)
        tl, nb = 0.0, 0
        for i in range(0, len(xt), args.batch):
            b = perm[i:i + args.batch]
            if len(b) < 2:      # BatchNorm (TabNet) needs >= 2 samples
                continue
            z, aux = m.embed(xt[b])
            loss = F.cross_entropy(m.head(z), yt[b]) + aux
            opt.zero_grad(); loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip)
            opt.step()
            tl += loss.item(); nb += 1
        m.eval()
        with torch.no_grad():
            zv, _ = m.embed(xv)
            lv = m.head(zv)
            vl = F.cross_entropy(lv, yv).item()
            vauc = roc_auc_score(yva, F.softmax(lv, 1)[:, 1].cpu().numpy())
        curve.append({"epoch": ep + 1, "train_loss": tl / max(nb, 1), "val_loss": vl, "val_auc": vauc})
        if vl < best - 1e-6:
            best, best_state, wait = vl, deepcopy(m.state_dict()), 0
        else:
            wait += 1
            if wait >= args.patience:
                break
    m.load_state_dict(best_state)
    m.eval()
    stop_epoch = len(curve) - wait
    m.best_val_loss = best
    return m, pd.DataFrame(curve), stop_epoch


@torch.no_grad()
def embed_np(m, X):
    z, _ = m.embed(torch.tensor(X, device=DEV))
    return z.cpu().numpy()


@torch.no_grad()
def e2e_proba(m, X):
    z, _ = m.embed(torch.tensor(X, device=DEV))
    return F.softmax(m.head(z), 1)[:, 1].cpu().numpy()


def make_head(name, seed):
    if name == "XGBoost":
        return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, tree_method="hist",
                                 device="cuda" if GPU else "cpu", eval_metric="logloss", random_state=seed)
    if name == "ExtraTrees":
        return ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
    if name == "kNN":
        return KNeighborsClassifier(n_neighbors=args.knn_k, metric="euclidean")
    raise ValueError(name)


def tuned_knn(Ztr, ytr, Zva, yva):
    """Choose k and weighting on the VALIDATION set (AUC, ties broken by log-loss, then smaller k)."""
    if args.no_tune:
        return KNeighborsClassifier(n_neighbors=args.knn_k).fit(Ztr, ytr), {"k": args.knn_k, "weights": "uniform"}
    best, best_cfg = None, None
    for k in [int(v) for v in args.knn_grid.split(",")]:
        for w in ("uniform", "distance"):
            pv = KNeighborsClassifier(n_neighbors=k, weights=w).fit(Ztr, ytr).predict_proba(Zva)[:, 1]
            score = (roc_auc_score(yva, pv), -log_loss(yva, np.clip(pv, 1e-7, 1 - 1e-7)), -k)
            if best is None or score > best:
                best, best_cfg = score, {"k": k, "weights": w}
    return KNeighborsClassifier(n_neighbors=best_cfg["k"], weights=best_cfg["weights"]).fit(Ztr, ytr), best_cfg


def classical_models(seed):
    ms = {
        "LogisticRegression": LogisticRegression(max_iter=5000),
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1),
        "SVM-RBF": SVC(probability=True, random_state=seed),
        "HistGB": HistGradientBoostingClassifier(random_state=seed),
    }
    try:
        import lightgbm as lgb
        ms["LightGBM"] = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=seed, verbose=-1)
    except ImportError:
        pass
    if args.tabpfn and not args.skip_tabpfn:
        try:
            from tabpfn import TabPFNClassifier
            ms["TabPFN"] = TabPFNClassifier(device="cuda" if GPU else "cpu", random_state=seed)
        except Exception as e:
            log("TabPFN not available, skipped:", repr(e)[:200])
            args.skip_tabpfn = True
    return ms


def run_fold(Xtr, ytr, Xva, yva, Xte, yte, seed, keep=False):
    """Train every model on one split. Returns test probabilities per model (+ artifacts if keep)."""
    Xtr_p, Xva_p, Xte_p, idx, sc = preprocess(Xtr, ytr, Xva, yva, Xte, seed)
    probs, art = {}, {"idx": idx, "scaler": sc, "Xtr": Xtr_p, "Xte": Xte_p, "Xva": Xva_p,
                      "backbones": {}, "heads": {}, "curves": {}, "stop": {}}
    # raw-feature heads (ablation) + classical baselines
    art["choices"] = {}
    for h in HEADS:
        if h == "kNN":
            mdl, cfg = tuned_knn(Xtr_p, ytr, Xva_p, yva); art["choices"]["Raw+kNN"] = cfg
        else:
            mdl = make_head(h, seed).fit(Xtr_p, ytr)
        probs[f"Raw+{h}"] = mdl.predict_proba(Xte_p)[:, 1]
        if keep: art["heads"][f"Raw+{h}"] = mdl
    for n, mdl in classical_models(seed).items():
        try:
            mdl.fit(Xtr_p, ytr)
            probs[n] = mdl.predict_proba(Xte_p)[:, 1]
        except Exception as e:
            log(f"{n} failed: {repr(e)[:200]}")
    # deep backbones: end-to-end + hybrid heads on frozen embeddings
    for b in BACKBONES:
        if b == "TabNet" and args.tune_tabnet and not args.no_tune:
            # TabNet width chosen on the validation loss (embedding dim = width)
            cands = []
            for wdt in [int(v) for v in args.tabnet_grid.split(",")]:
                cands.append((train_backbone(b, Xtr_p, ytr, Xva_p, yva, seed, width=wdt), wdt))
            (m, curve, stop), wdt = min(cands, key=lambda c: c[0][0].best_val_loss)
            art["choices"]["TabNet_width"] = wdt
            del cands
        else:
            m, curve, stop = train_backbone(b, Xtr_p, ytr, Xva_p, yva, seed)
        probs[f"{b}-EndToEnd"] = e2e_proba(m, Xte_p)
        Ztr, Zte, Zva = embed_np(m, Xtr_p), embed_np(m, Xte_p), embed_np(m, Xva_p)
        for h in HEADS:
            if h == "kNN":
                mdl, cfg = tuned_knn(Ztr, ytr, Zva, yva); art["choices"][f"{b}+kNN"] = cfg
            else:
                mdl = make_head(h, seed).fit(Ztr, ytr)
            probs[f"{b}+{h}"] = mdl.predict_proba(Zte)[:, 1]
            if keep: art["heads"][f"{b}+{h}"] = mdl
        if keep:
            art["backbones"][b], art["curves"][b], art["stop"][b] = m, curve, stop
        else:
            del m
    if GPU:
        torch.cuda.empty_cache()
    return probs, art


# ----------------------------------------------------------------------------- metrics & tests
def ece_score(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ids = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        mask = ids == b
        if mask.any():
            e += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return e


def calib_slope_intercept(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    try:
        import statsmodels.api as sm
        warnings.simplefilter("ignore")
        slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
        inter = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
        return float(slope), float(inter)
    except Exception:
        return np.nan, np.nan          # perfect separation -> undefined


def metrics(y, p, thr=0.5):
    yp = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    slope, inter = calib_slope_intercept(y, p)
    return {"Accuracy": accuracy_score(y, yp), "Precision": precision_score(y, yp, zero_division=0),
            "Recall(Sens)": recall_score(y, yp, zero_division=0), "Specificity": tn / (tn + fp) if tn + fp else np.nan,
            "F1": f1_score(y, yp, zero_division=0), "AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p), "ECE": ece_score(y, p), "LogLoss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7)),
            "CalSlope": slope, "CalIntercept": inter, "TP": tp, "FN": fn, "FP": fp, "TN": tn}


def mcnemar_exact(y, p1, p2):
    c1, c2 = (p1 >= .5) == y, (p2 >= .5) == y
    b, c = int(np.sum(c1 & ~c2)), int(np.sum(~c1 & c2))
    pval = 1.0 if b + c == 0 else stats.binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, min(pval, 1.0)


def _midrank(x):
    j = np.argsort(x); z = x[j]; n = len(x); t = np.zeros(n); i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1; i = k
    out = np.empty(n); out[j] = t
    return out


def delong_test(y, p1, p2):
    """Fast DeLong (Sun & Xu, 2014) for two correlated AUCs."""
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for pr in (p1, p2):
        tx, ty, tz = _midrank(pr[pos]), _midrank(pr[neg]), _midrank(np.r_[pr[pos], pr[neg]])
        aucs.append(tz[:m].sum() / m / n - (m + 1) / 2 / n)
        v10.append((tz[:m] - tx) / n); v01.append(1 - (tz[m:] - ty) / m)
    s = np.cov(np.array(v10)) / m + np.cov(np.array(v01)) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 1e-12:
        return aucs[0], aucs[1], 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return aucs[0], aucs[1], float(2 * stats.norm.sf(abs(z)))


def holm(pvals):
    p = np.asarray(pvals, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def comparison_pairs(names):
    pairs = []
    for b in BACKBONES:
        for h in HEADS:
            pairs.append((f"{b}+{h}", f"{b}-EndToEnd"))       # RQ1: hybrid vs its own end-to-end
            pairs.append((f"{b}+{h}", f"Raw+{h}"))            # embedding vs raw features, same head
    return [(a, c) for a, c in pairs if a in names and c in names]


def stat_table(y, probs):
    rows = []
    for a, c in comparison_pairs(probs):
        b_, c_, pm = mcnemar_exact(y, probs[a], probs[c])
        a1, a2, pd_ = delong_test(y, probs[a], probs[c])
        rows.append({"ModelA": a, "ModelB": c, "McNemar_b(A right,B wrong)": b_, "McNemar_c(A wrong,B right)": c_,
                     "McNemar_p_exact": pm, "AUC_A": a1, "AUC_B": a2, "DeLong_p": pd_})
    df = pd.DataFrame(rows)
    if len(df):
        df["McNemar_p_Holm"], df["DeLong_p_Holm"] = holm(df["McNemar_p_exact"]), holm(df["DeLong_p"])
    return df


# ----------------------------------------------------------------------------- geometry
def geometry(Z, y, label):
    Zs = StandardScaler().fit_transform(Z)
    c0, c1 = Zs[y == 0].mean(0), Zs[y == 1].mean(0)
    intra = np.mean([np.linalg.norm(Zs[y == k] - Zs[y == k].mean(0), axis=1).mean() for k in (0, 1)])
    inter = np.linalg.norm(c0 - c1)
    return {"Representation": label, "Dim": Z.shape[1], "Silhouette": silhouette_score(Zs, y),
            "DaviesBouldin": davies_bouldin_score(Zs, y), "CalinskiHarabasz": calinski_harabasz_score(Zs, y),
            "IntraClassDist": intra, "InterCentroidDist": inter, "Separability(inter/intra)": inter / intra}


# ----------------------------------------------------------------------------- plots
def plots_single(y, probs, art):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = [f"{b}-EndToEnd" for b in BACKBONES] + [f"{b}+XGBoost" for b in BACKBONES]
    # confusion matrices
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for a, n in zip(ax.ravel(), main):
        cm = confusion_matrix(y, (probs[n] >= .5).astype(int), labels=[1, 0])
        a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
        a.set_xticks([0, 1], ["Malignant", "Benign"]); a.set_yticks([0, 1], ["Malignant", "Benign"])
        a.set_xlabel("Predicted"); a.set_ylabel("True"); a.set_title(n, fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "confusion_matrices.png"), dpi=300); plt.close()
    # reliability diagrams
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for a, n in zip(ax.ravel(), main):
        bins = np.linspace(0, 1, 11); ids = np.clip(np.digitize(probs[n], bins[1:-1]), 0, 9)
        xs, ys, cs = [], [], []
        for k in range(10):
            mk = ids == k
            if mk.any():
                xs.append(probs[n][mk].mean()); ys.append(y[mk].mean()); cs.append(mk.sum())
        a.plot([0, 1], [0, 1], "k--", lw=1); a.plot(xs, ys, "o-")
        for x_, y_, c_ in zip(xs, ys, cs):
            a.annotate(str(c_), (x_, y_), fontsize=7, xytext=(3, 3), textcoords="offset points")
        a.set_title(f"{n}\nECE={ece_score(y, probs[n]):.3f}", fontsize=9)
        a.set_xlabel("Mean predicted P(malignant)"); a.set_ylabel("Observed fraction malignant")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "reliability_diagrams.png"), dpi=300); plt.close()
    # ROC
    plt.figure(figsize=(6, 6))
    for n in main + ["Raw+XGBoost", "LogisticRegression"]:
        if n in probs:
            f, t, _ = roc_curve(y, probs[n]); plt.plot(f, t, label=f"{n} ({roc_auc_score(y, probs[n]):.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "roc_curves.png"), dpi=300); plt.close()
    # training curves
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for j, b in enumerate(BACKBONES):
        c = art["curves"][b]
        ax[0, j].plot(c.epoch, c.train_loss, label="train"); ax[0, j].plot(c.epoch, c.val_loss, label="val")
        ax[0, j].axvline(art["stop"][b], color="r", ls=":", label=f"best epoch {art['stop'][b]}")
        ax[0, j].set_title(f"{b} loss"); ax[0, j].legend(fontsize=7)
        ax[1, j].plot(c.epoch, c.val_auc); ax[1, j].set_title(f"{b} validation AUC"); ax[1, j].set_xlabel("epoch")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "training_curves.png"), dpi=300); plt.close()


# ----------------------------------------------------------------------------- main
def main():
    open(os.path.join(OUT, "run_log.txt"), "a").close()
    t0 = time.time()
    env = {"python": platform.python_version(), "torch": torch.__version__, "xgboost": xgb.__version__,
           "device": torch.cuda.get_device_name(0) if GPU else "CPU", "cuda": torch.version.cuda,
           "seed": args.seed, "args": vars(args)}
    import sklearn; env["sklearn"] = sklearn.__version__
    json.dump(env, open(os.path.join(OUT, "env.json"), "w"), indent=2)
    log("Device:", env["device"], "| quick mode" if args.quick else "")
    log(f"Dataset: {DS_NAME} ({N_ALL} samples, {X_ALL.shape[1]} features, {int(Y_ALL.sum())} malignant) | MI k grid: {K_GRID}")
    log(f"Settings (Table 2): optimizer={args.optimizer} lr={args.lr} batch={args.batch} max_epochs={args.epochs} "
        f"patience={args.patience} dropout={args.dropout} heads={args.heads} clip={args.clip} wd={args.wd} "
        f"latent={args.emb} seed={args.seed} | kNN tuned on validation | TabNet width tuning={'on' if args.tune_tabnet else 'off (128)'}")
    if args.quick:
        log("!!! QUICK MODE: numbers are NOT valid for the paper !!!")

    # ===================== 1. single stratified split (70/15/15) =====================
    set_seed(args.seed)
    Xtmp, Xte, ytmp, yte = train_test_split(X_ALL, Y_ALL, test_size=N_TEST, stratify=Y_ALL, random_state=args.seed)
    Xtr, Xva, ytr, yva = train_test_split(Xtmp, ytmp, test_size=N_VAL, stratify=ytmp, random_state=args.seed)
    log(f"Single split: train={len(ytr)} val={len(yva)} test={len(yte)} (malignant in test={yte.sum()})")
    probs, art = run_fold(Xtr, ytr, Xva, yva, Xte, yte, args.seed, keep=True)
    sel = [FEATS[i] for i in art["idx"]]
    log(f"Selected k={len(sel)} MI features:", ", ".join(sel))
    json.dump({"k": len(sel), "features": sel, "early_stop_epoch": art["stop"], "tuned": art["choices"]},
              open(os.path.join(OUT, "single_split_selection.json"), "w"), indent=2)

    tab = pd.DataFrame({n: metrics(yte, pr) for n, pr in probs.items()}).T
    tab.to_csv(os.path.join(OUT, "single_split_test_metrics.csv"))
    pd.DataFrame(probs).assign(y_true=yte).to_csv(os.path.join(OUT, "single_split_test_probabilities.csv"), index=False)
    stat_table(yte, probs).to_csv(os.path.join(OUT, "stats_single_split_test.csv"), index=False)
    for b in BACKBONES:
        art["curves"][b].to_csv(os.path.join(OUT, f"training_curve_{b}.csv"), index=False)
    plots_single(yte, probs, art)
    log("Single-split metrics written.")

    # ---- geometry: raw features vs embeddings, on test and train
    geo = []
    for part, Xp, yp in (("test", art["Xte"], yte), ("train", art["Xtr"], ytr)):
        g = geometry(Xp, yp, "Raw features (MI-selected)"); g["Set"] = part; geo.append(g)
        for b in BACKBONES:
            g = geometry(embed_np(art["backbones"][b], Xp), yp, f"{b} embedding"); g["Set"] = part; geo.append(g)
    pd.DataFrame(geo).to_csv(os.path.join(OUT, "geometry_raw_vs_embeddings.csv"), index=False)
    log("Geometry written.")

    # ---- robustness: Gaussian noise on standardized test features, hybrid vs end-to-end vs raw
    rng = np.random.default_rng(args.seed); rob = []
    for s in SIGMAS:
        res = {}
        for r in range(args.robust_reps if s > 0 else 1):
            Xn = (art["Xte"] + rng.normal(0, s, art["Xte"].shape)).astype(np.float32)
            for b in BACKBONES:
                m = art["backbones"][b]
                res.setdefault(f"{b}-EndToEnd", []).append(roc_auc_score(yte, e2e_proba(m, Xn)))
                res.setdefault(f"{b}+XGBoost", []).append(
                    roc_auc_score(yte, art["heads"][f"{b}+XGBoost"].predict_proba(embed_np(m, Xn))[:, 1]))
            res.setdefault("Raw+XGBoost", []).append(roc_auc_score(yte, art["heads"]["Raw+XGBoost"].predict_proba(Xn)[:, 1]))
        for n, v in res.items():
            rob.append({"sigma": s, "Model": n, "AUC_mean": np.mean(v), "AUC_sd": np.std(v, ddof=1) if len(v) > 1 else 0.0,
                        "n_repeats": len(v)})
    rob = pd.DataFrame(rob); rob.to_csv(os.path.join(OUT, "robustness_noise.csv"), index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 5))
    for n, d in rob.groupby("Model"):
        plt.errorbar(d.sigma, d.AUC_mean, yerr=d.AUC_sd, marker="o", capsize=2, label=n,
                     ls="--" if "EndToEnd" in n else "-")
    plt.xlabel("Gaussian noise sigma (standardized features)"); plt.ylabel(f"Test AUC (mean ± SD, {args.robust_reps} repeats)")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(FIG, "robustness.png"), dpi=300); plt.close()
    log("Robustness written.")

    # ---- latency (per sample, batch size 1, and full-batch), measured on this device
    lat = []
    for b in BACKBONES:
        m = art["backbones"][b]; xt = art["Xte"]
        for _ in range(10): e2e_proba(m, xt[:1])
        for kind in ("EndToEnd", "+XGBoost"):
            if GPU: torch.cuda.synchronize()
            t = time.perf_counter()
            for i in range(len(xt)):
                if kind == "EndToEnd": e2e_proba(m, xt[i:i + 1])
                else: art["heads"][f"{b}+XGBoost"].predict_proba(embed_np(m, xt[i:i + 1]))
            if GPU: torch.cuda.synchronize()
            lat.append({"Model": f"{b}{'-' if kind == 'EndToEnd' else ''}{kind}", "ms_per_sample_batch1": (time.perf_counter() - t) / len(xt) * 1e3,
                        "device": env["device"]})
    t = time.perf_counter()
    for i in range(len(art["Xte"])): art["heads"]["Raw+XGBoost"].predict_proba(art["Xte"][i:i + 1])
    lat.append({"Model": "Raw+XGBoost", "ms_per_sample_batch1": (time.perf_counter() - t) / len(art["Xte"]) * 1e3, "device": env["device"]})
    # extra latency views: whole test batch at once, and XGBoost head on CPU (embedding still on GPU)
    for b in BACKBONES:
        m = art["backbones"][b]; xt = art["Xte"]; hd = art["heads"][f"{b}+XGBoost"]
        if GPU: torch.cuda.synchronize()
        t = time.perf_counter(); e2e_proba(m, xt)
        if GPU: torch.cuda.synchronize()
        bt_e2e = (time.perf_counter() - t) / len(xt) * 1e3
        t = time.perf_counter(); hd.predict_proba(embed_np(m, xt)); bt_h = (time.perf_counter() - t) / len(xt) * 1e3
        hc = deepcopy(hd); hc.set_params(device="cpu")
        t = time.perf_counter()
        for i in range(len(xt)): hc.predict_proba(embed_np(m, xt[i:i + 1]))
        cpu_h = (time.perf_counter() - t) / len(xt) * 1e3
        for row in lat:
            if row["Model"] == f"{b}-EndToEnd": row["ms_per_sample_batch86"] = bt_e2e
            if row["Model"] == f"{b}+XGBoost": row["ms_per_sample_batch86"] = bt_h; row["ms_per_sample_batch1_xgb_cpu"] = cpu_h
    hc = deepcopy(art["heads"]["Raw+XGBoost"]); hc.set_params(device="cpu")
    t = time.perf_counter()
    for i in range(len(art["Xte"])): hc.predict_proba(art["Xte"][i:i + 1])
    lat[-1]["ms_per_sample_batch1_xgb_cpu"] = (time.perf_counter() - t) / len(art["Xte"]) * 1e3
    t = time.perf_counter(); art["heads"]["Raw+XGBoost"].predict_proba(art["Xte"])
    lat[-1]["ms_per_sample_batch86"] = (time.perf_counter() - t) / len(art["Xte"]) * 1e3
    pd.DataFrame(lat).to_csv(os.path.join(OUT, "latency.csv"), index=False)
    log("Latency written.")

    # ---- SHAP + LIME + agreement (on the TEST set, model given by --xai_model)
    if not args.skip_xai:
        try:
            run_xai(art, ytr, yte, sel)
        except Exception as e:
            log("XAI failed:", repr(e)[:300])

    # ===================== 2. repeated stratified k-fold CV, same protocol for all models =====================
    if not args.skip_cv:
        rows, oof_all, choices = [], [], []
        for r in range(args.repeats):
            skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed + r)
            oof = {}
            for f, (tr, te) in enumerate(skf.split(X_ALL, Y_ALL)):
                s = args.seed + 1000 * r + f
                Xa, Xv, ya, yv = train_test_split(X_ALL[tr], Y_ALL[tr], test_size=0.15, stratify=Y_ALL[tr], random_state=s)
                cache = os.path.join(OUT, "cv_cache", f"r{r}_f{f}.pkl")
                if os.path.exists(cache):
                    pr, ch = pd.read_pickle(cache)
                else:
                    pr, a_ = run_fold(Xa, ya, Xv, yv, X_ALL[te], Y_ALL[te], s)
                    ch = a_["choices"]; del a_
                    os.makedirs(os.path.dirname(cache), exist_ok=True)
                    pd.to_pickle((pr, ch), cache)
                choices.append({"Repeat": r, "Fold": f, **{k: (f"k={v['k']},{v['weights']}" if isinstance(v, dict) else v) for k, v in ch.items()}})
                for n, v in pr.items():
                    mm = metrics(Y_ALL[te], v); mm.update({"Model": n, "Repeat": r, "Fold": f}); rows.append(mm)
                    oof.setdefault(n, np.zeros(len(Y_ALL)))[te] = v
                log(f"CV repeat {r + 1}/{args.repeats} fold {f + 1}/{args.folds} done ({(time.time() - t0) / 60:.1f} min)")
            oof_all.append(oof)
            pd.DataFrame(oof).assign(y_true=Y_ALL).to_csv(os.path.join(OUT, f"cv_oof_probabilities_repeat{r}.csv"), index=False)
        per = pd.DataFrame(rows); per.to_csv(os.path.join(OUT, "cv_metrics_per_fold.csv"), index=False)
        pd.DataFrame(choices).to_csv(os.path.join(OUT, "cv_tuning_choices.csv"), index=False)
        # pooled out-of-fold calibration (stable slope/intercept): one value per repeat, then mean +/- SD
        pc = {}
        for oof in oof_all:
            for n, v in oof.items():
                mm = metrics(Y_ALL, v)
                pc.setdefault(n, []).append([mm["AUC"], mm["Accuracy"], mm["Brier"], mm["ECE"], mm["CalSlope"], mm["CalIntercept"]])
        cal = []
        for n, v in pc.items():
            v = np.array(v, float)
            row = {"Model": n}
            for j, k in enumerate(["AUC", "Accuracy", "Brier", "ECE", "CalSlope", "CalIntercept"]):
                col = v[:, j]
                ok = np.isfinite(col) & (np.abs(col) < 1e3)      # kNN with 0/1 probabilities -> undefined
                row[k + "_mean"] = col[ok].mean() if ok.all() else np.nan
                row[k + "_sd"] = col[ok].std(ddof=1) if ok.all() and len(col) > 1 else np.nan
            cal.append(row)
        pd.DataFrame(cal).to_csv(os.path.join(OUT, "cv_pooled_oof_calibration.csv"), index=False)
        keys = ["Accuracy", "Precision", "Recall(Sens)", "Specificity", "F1", "AUC", "Brier", "ECE", "CalSlope", "CalIntercept"]
        summ = []
        for n, d in per.groupby("Model"):
            row = {"Model": n}
            for k in keys:
                rep = d.groupby("Repeat")[k].mean().values       # one value per repeat
                mu, sd = d[k].mean(), d[k].std(ddof=1)
                half = stats.t.ppf(.975, len(rep) - 1) * rep.std(ddof=1) / np.sqrt(len(rep)) if len(rep) > 1 else np.nan
                row[k] = f"{mu:.4f} ± {sd:.4f}"; row[k + "_mean"] = mu; row[k + "_sd"] = sd
                row[k + "_CI95"] = f"[{mu - half:.4f}, {mu + half:.4f}]" if len(rep) > 1 else "n/a"
            summ.append(row)
        summ = pd.DataFrame(summ).sort_values("AUC_mean", ascending=False)
        summ.to_csv(os.path.join(OUT, "cv_metrics_summary.csv"), index=False)
        # pooled OOF predictions of every repeat -> tests on all samples (repeat 0 reported; others saved)
        st = stat_table(Y_ALL, oof_all[0]); st.to_csv(os.path.join(OUT, "stats_cv_oof_repeat0.csv"), index=False)
        with open(os.path.join(OUT, "latex_tables.tex"), "w") as fh:
            cols = ["Model"] + keys[:8]
            fh.write(summ[cols].to_latex(index=False, escape=True, caption=f"Repeated stratified {args.folds}-fold CV ({args.repeats} repeats): mean ± SD.", label="tab:cv"))
            fh.write("\n\n")
            fh.write(st.round(4).to_latex(index=False, escape=True, caption="Exact McNemar and DeLong tests on pooled out-of-fold predictions (n = " + str(N_ALL) + "), Holm-corrected.", label="tab:stats"))
        log("CV written.")
        try:
            cv_figures(); log("CV figures written.")
        except Exception as e:
            log("CV figures failed:", repr(e)[:200])

    # ---- human-readable summary
    write_summary(tab, env, t0)
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min. Results in: {OUT}/")


def run_xai(art, ytr, yte, sel):
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    name = args.xai_model
    b, h = name.split("+")
    m, head = art["backbones"][b], art["heads"][name]
    f = lambda X: head.predict_proba(embed_np(m, np.asarray(X, np.float32)))
    Xtr, Xte = art["Xtr"], art["Xte"]
    bg = shap.kmeans(Xtr, 20)
    sv = shap.KernelExplainer(lambda X: f(X)[:, 1], bg).shap_values(Xte, nsamples=200 if not args.quick else 50, silent=True)
    shap_imp = np.abs(sv).mean(0)
    lime_exp = LimeTabularExplainer(Xtr, feature_names=sel, class_names=["Benign", "Malignant"],
                                    discretize_continuous=False, random_state=args.seed)
    lw = np.zeros_like(sv)
    for i in range(len(Xte)):
        e = lime_exp.explain_instance(Xte[i], f, num_features=len(sel), num_samples=1000 if not args.quick else 200)
        for fi, w in e.as_map()[1]:
            lw[i, fi] = w
    lime_imp = np.abs(lw).mean(0)
    rho, prho = stats.spearmanr(shap_imp, lime_imp)
    per_rho = [stats.spearmanr(np.abs(sv[i]), np.abs(lw[i]))[0] for i in range(len(Xte))]
    top = lambda v: set(np.argsort(-v)[:5])
    jac = len(top(shap_imp) & top(lime_imp)) / len(top(shap_imp) | top(lime_imp))
    pd.DataFrame({"Feature": sel, "mean|SHAP|": shap_imp, "mean|LIME|": lime_imp}).sort_values(
        "mean|SHAP|", ascending=False).to_csv(os.path.join(OUT, "xai_feature_importance_test.csv"), index=False)
    pd.DataFrame([{"Model": name, "Set": f"test (n={len(art['Xte'])})", "Spearman_global": rho, "p_global": prho,
                   "Spearman_per_sample_mean": np.nanmean(per_rho), "Spearman_per_sample_sd": np.nanstd(per_rho),
                   "Top5_Jaccard": jac}]).to_csv(os.path.join(OUT, "xai_shap_lime_agreement.csv"), index=False)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    shap.summary_plot(sv, Xte, feature_names=sel, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_summary_test.png"), dpi=300); plt.close()
    log(f"XAI written (Spearman SHAP vs LIME = {rho:.3f}).")


def cv_figures():
    """Extra paper figures from CV outputs: head ablation bars, box plots, pooled-OOF reliability, OOF ROC."""
    import glob, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    per = pd.read_csv(os.path.join(OUT, "cv_metrics_per_fold.csv"))
    Bn = {"FTTransformer": "FT-Transformer", "DCNv2": "DCN-V2", "TabNet": "TabNet", "TabMLPMixer": "TabMLP-Mixer"}
    H = ["EndToEnd", "XGBoost", "ExtraTrees", "kNN"]
    lab = {"EndToEnd": "End-to-end (softmax) / LR", "XGBoost": "XGBoost head", "ExtraTrees": "ExtraTrees head", "kNN": "kNN head"}
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for k, met in enumerate(["Accuracy", "AUC"]):
        for j, h in enumerate(H):
            names = [f"{b}-EndToEnd" if h == "EndToEnd" else f"{b}+{h}" for b in BACKBONES] + ["LogisticRegression" if h == "EndToEnd" else f"Raw+{h}"]
            mu = [per[per.Model == n][met].mean() for n in names]; sd = [per[per.Model == n][met].std() for n in names]
            ax[k].bar(np.arange(5) + (j - 1.5) * 0.2, mu, 0.2, yerr=sd, capsize=2, label=lab[h])
        ax[k].set_xticks(range(5), [Bn[b] for b in BACKBONES] + ["Raw features"])
        ax[k].set_ylim(0.85 if met == "Accuracy" else 0.93, 1.005)
        ax[k].set_ylabel(f"{met} (mean ± SD, {args.repeats}×{args.folds}-fold CV)"); ax[k].grid(axis="y", alpha=.3)
    ax[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "cv_ablation_heads.png"), dpi=300); plt.close()
    order = per.groupby("Model").AUC.mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for a, met in zip(ax, ["AUC", "Accuracy"]):
        a.boxplot([per[per.Model == m][met] for m in order], showfliers=False); a.set_ylabel(met); a.grid(axis="y", alpha=.3)
    ax[1].set_xticks(range(1, len(order) + 1), order, rotation=75, fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "cv_boxplots_all_models.png"), dpi=300); plt.close()
    D = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(OUT, "cv_oof_probabilities_repeat*.csv")))]
    y = np.concatenate([d.y_true.values for d in D])
    main = [f"{b}-EndToEnd" for b in BACKBONES] + [f"{b}+XGBoost" for b in BACKBONES]
    fig, ax = plt.subplots(2, 4, figsize=(16, 8))
    for a, m in zip(ax.ravel(), main):
        pr = np.concatenate([d[m].values for d in D]); bi = np.clip(np.digitize(pr, np.linspace(0, 1, 11)[1:-1]), 0, 9)
        ks = [k for k in range(10) if (bi == k).any()]
        xs, ys, cs = [pr[bi == k].mean() for k in ks], [y[bi == k].mean() for k in ks], [(bi == k).sum() for k in ks]
        a.plot([0, 1], [0, 1], "k--", lw=1); a.plot(xs, ys, "o-")
        for x_, y_, c_ in zip(xs, ys, cs):
            a.annotate(str(c_), (x_, y_), fontsize=7, xytext=(3, -9), textcoords="offset points")
        a.set_title(f"{m}\nECE={ece_score(y, pr):.3f} (pooled OOF, n={len(y)})", fontsize=9)
        a.set_xlabel("Mean predicted P(malignant)"); a.set_ylabel("Observed fraction malignant")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "cv_reliability_pooled_oof.png"), dpi=300); plt.close()
    d = D[0]; yy = d.y_true.values
    top = d.drop(columns="y_true").apply(lambda c: roc_auc_score(yy, c)).sort_values(ascending=False).index[:6]
    plt.figure(figsize=(6, 6))
    for m in top:
        f_, t_, _ = roc_curve(yy, d[m]); plt.plot(f_, t_, label=f"{m} (AUC={roc_auc_score(yy, d[m]):.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.legend(fontsize=8); plt.title(f"Out-of-fold ROC, repeat 1 (n={len(yy)})", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "cv_roc_oof.png"), dpi=300); plt.close()


def write_summary(tab, env, t0):
    cols = ["Accuracy", "Recall(Sens)", "Specificity", "F1", "AUC", "Brier", "ECE", "TP", "FN", "FP", "TN"]
    lines = ["# Re-run summary", "", f"Device: {env['device']}  |  seed: {args.seed}  |  runtime: {(time.time() - t0) / 60:.1f} min", ""]
    if args.quick:
        lines += ["**QUICK MODE - NOT VALID FOR THE PAPER**", ""]
    lines += [f"## {DS_NAME}: single split, test set (n = {N_TEST})", "", tab[cols].astype(float).round(4).to_markdown(), ""]
    cv = os.path.join(OUT, "cv_metrics_summary.csv")
    if os.path.exists(cv):
        s = pd.read_csv(cv)
        lines += [f"## Repeated stratified CV ({args.repeats} x {args.folds}), mean ± SD", "",
                  s[["Model", "Accuracy", "AUC", "Brier", "ECE", "AUC_CI95"]].to_markdown(index=False), ""]
    open(os.path.join(OUT, "SUMMARY.md"), "w").write("\n".join(lines))


if __name__ == "__main__":
    main()
