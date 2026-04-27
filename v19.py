"""
Nedbank Transaction Forecasting - v19
======================================
Adds a GRU-based neural net on the per-customer 24-month transaction-count
sequence as a 7th base predictor. Goal: introduce inductive-bias diversity
that the v18 GBT ensemble lacks (every GBT model in v18 is >0.999 correlated
with every other in log space — NNLS gives ~0 weight to anything past LGB+
XGB+CAT, so the blend is at a hard ceiling).

Pipeline:
  1. Build the same features as v18 (so the count sequence m1..m24 and static
     features come out identically aligned to v18 OOF rows).
  2. Train a 5-fold StratifiedKFold GRU model (matching v18's CV split exactly
     so OOFs are row-aligned).
  3. Blend [v18_blend_cal, nn] in log space via NNLS, then linear-cal the result.
  4. Save submission_v19.csv.

Notes:
  - We don't have v18's per-model test predictions saved, so we treat v18's
    final calibrated blend as a single base predictor here. If NNLS gives the
    NN nonzero weight, the architecture is contributing.
  - CPU training, ~5-10 min per fold for 8K orig + ~100K aug rows.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn

DATA_DIR  = Path(__file__).parent
TXN_PATH  = DATA_DIR / "transactions_features/transactions_features.parquet"
DEMO_PATH = DATA_DIR / "demographics_clean/demographics_clean.parquet"
FIN_PATH  = DATA_DIR / "financials_features/financials_features.parquet"

RANDOM_STATE = 42
N_FOLDS      = 5
DEVICE = torch.device("cpu")  # GPU not available on this host

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("Loading labels ...")
train_labels = pd.read_csv(DATA_DIR / "Train.csv")
test_ids     = pd.read_csv(DATA_DIR / "Test.csv")
all_train_ids = set(train_labels["UniqueID"])
all_test_ids  = set(test_ids["UniqueID"])

print("Loading transactions ...")
txn_all = pd.read_parquet(TXN_PATH)
txn_all["TransactionDate"] = pd.to_datetime(txn_all["TransactionDate"])
all_customer_ids = txn_all["UniqueID"].unique()
print(f"  {len(txn_all):,} rows  |  {len(all_customer_ids):,} customers")

print("Loading demographics ...")
demo_raw = pd.read_parquet(DEMO_PATH)


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE BUILDERS  (focused — only what NN needs: 24-month sequence + a
#    handful of static features. Full v18 features aren't needed since the GBT
#    side comes from oof_v18.csv / submission_v18.csv.)
# ══════════════════════════════════════════════════════════════════════════════

SEQ_LEN = 24

def build_features(cutoff: pd.Timestamp, customer_ids):
    """Returns dataframe with m1..m24 monthly counts + a few static features."""
    txn = txn_all[txn_all["TransactionDate"] <= cutoff]
    feat = pd.DataFrame(index=customer_ids); feat.index.name = "UniqueID"

    # 24 monthly counts
    for m in range(1, SEQ_LEN + 1):
        mo_end   = cutoff - pd.DateOffset(months=m - 1)
        mo_start = cutoff - pd.DateOffset(months=m)
        c = txn[(txn["TransactionDate"] > mo_start) & (txn["TransactionDate"] <= mo_end)] \
                .groupby("UniqueID").size()
        feat[f"m{m}"]     = c
        feat[f"log_m{m}"] = np.log1p(c)

    # Static features
    W = {"3m":  cutoff - pd.DateOffset(months=3),
         "6m":  cutoff - pd.DateOffset(months=6),
         "12m": cutoff - pd.DateOffset(months=12)}
    for label, start in W.items():
        c = txn[txn["TransactionDate"] > start].groupby("UniqueID").size()
        feat[f"log_txn_{label}"] = np.log1p(c)

    last_txn  = txn.groupby("UniqueID")["TransactionDate"].max()
    first_txn = txn.groupby("UniqueID")["TransactionDate"].min()
    feat["log_days_since_last"] = np.log1p(((cutoff - last_txn).dt.days).clip(lower=0))
    feat["log_vintage"]         = np.log1p((cutoff - first_txn).dt.days)
    feat["log_n_accounts"]      = np.log1p(txn.groupby("UniqueID")["AccountID"].nunique())

    # Monthly aggregates
    txn_m = (
        txn[txn["TransactionDate"] > W["12m"]]
        .assign(ym=lambda d: d["TransactionDate"].dt.to_period("M"))
        .groupby(["UniqueID", "ym"]).size().reset_index(name="cnt")
    )
    ms = txn_m.groupby("UniqueID")["cnt"].agg(active_months="count", monthly_mean="mean")
    feat = feat.join(ms)
    feat["log_monthly_mean"]       = np.log1p(feat["monthly_mean"])
    feat["frac_months_active_12m"] = feat["active_months"].clip(upper=12) / 12
    return feat.reset_index()


def build_demographics(cutoff):
    demo = demo_raw.copy()
    demo["age"] = ((cutoff - demo["BirthDate"]).dt.days / 365.25).clip(0, 120)
    demo = demo.drop(columns=["BirthDate"])
    INCOME_ORDER = {"No Income":0,"Not Disclosed / Unknown":0,"Low Income":1,
                    "Lower-Middle Income":2,"Middle Income":3,"Upper-Middle Income":4,
                    "High Income":5,"Very High Income":6}
    demo["income_ordinal"] = demo["IncomeCategory"].map(INCOME_ORDER).fillna(0)
    return demo[["UniqueID", "age", "income_ordinal"]]


# Same augmentation windows as v18 — needed so NN sees the same training mix
HIST_WINDOWS = [
    (pd.Timestamp("2013-04-30"), pd.Timestamp("2013-05-01"), pd.Timestamp("2013-07-31"), 0.3),
    (pd.Timestamp("2013-07-31"), pd.Timestamp("2013-08-01"), pd.Timestamp("2013-10-31"), 0.4),
    (pd.Timestamp("2013-10-31"), pd.Timestamp("2013-11-01"), pd.Timestamp("2014-01-31"), 0.6),
    (pd.Timestamp("2014-04-30"), pd.Timestamp("2014-05-01"), pd.Timestamp("2014-07-31"), 0.4),
    (pd.Timestamp("2014-07-31"), pd.Timestamp("2014-08-01"), pd.Timestamp("2014-10-31"), 0.6),
    (pd.Timestamp("2014-10-31"), pd.Timestamp("2014-11-01"), pd.Timestamp("2015-01-31"), 0.9),
    (pd.Timestamp("2015-01-31"), pd.Timestamp("2015-02-01"), pd.Timestamp("2015-04-30"), 0.5),
    (pd.Timestamp("2015-04-30"), pd.Timestamp("2015-05-01"), pd.Timestamp("2015-07-31"), 0.5),
    (pd.Timestamp("2015-07-31"), pd.Timestamp("2015-08-01"), pd.Timestamp("2015-10-31"), 0.85),
]
REAL_CUTOFF = pd.Timestamp("2015-10-31")

print("\nBuilding real features (2015-10-31) ...")
real_feat  = build_features(REAL_CUTOFF, all_customer_ids)
real_demo  = build_demographics(REAL_CUTOFF)
real_features = real_feat.merge(real_demo, on="UniqueID", how="left")

test_df    = real_features[real_features["UniqueID"].isin(all_test_ids)].copy().reset_index(drop=True)
orig_train = (
    real_features[real_features["UniqueID"].isin(all_train_ids)]
    .merge(train_labels, on="UniqueID", how="left").copy().reset_index(drop=True)
)
orig_train["sample_weight"] = 1.0
print(f"  Original train: {len(orig_train):,} rows")

aug_frames = [orig_train]
for hist_cutoff, t_start, t_end, weight in HIST_WINDOWS:
    print(f"  hist {hist_cutoff.date()} ...")
    hist_feat  = build_features(hist_cutoff, all_customer_ids)
    hist_demo  = build_demographics(hist_cutoff)
    hist_feats = hist_feat.merge(hist_demo, on="UniqueID", how="left")

    hist_targets = (
        txn_all[(txn_all["TransactionDate"] >= t_start) & (txn_all["TransactionDate"] <= t_end)]
        .groupby("UniqueID").size().reset_index(name="next_3m_txn_count")
    )
    active = txn_all[txn_all["TransactionDate"] <= hist_cutoff]["UniqueID"].unique()
    hist_targets = hist_targets[hist_targets["UniqueID"].isin(active)]

    aug = hist_feats.merge(hist_targets, on="UniqueID", how="inner")
    aug["sample_weight"] = weight
    aug_frames.append(aug)

combined = pd.concat(aug_frames, ignore_index=True)
print(f"\nCombined: {len(combined):,} rows")

SEQ_COLS    = [f"log_m{i}" for i in range(SEQ_LEN, 0, -1)]   # m24 (oldest) -> m1 (newest)
STATIC_COLS = ["log_txn_3m","log_txn_6m","log_txn_12m","log_days_since_last",
               "log_n_accounts","log_vintage","log_monthly_mean",
               "frac_months_active_12m","age","income_ordinal"]


def to_arrays(df):
    seq    = df[SEQ_COLS].fillna(0.0).values.astype(np.float32)
    static = df[STATIC_COLS].fillna(0.0).values.astype(np.float32)
    return seq, static


# ══════════════════════════════════════════════════════════════════════════════
# 3. GRU MODEL
# ══════════════════════════════════════════════════════════════════════════════

class GruRegressor(nn.Module):
    def __init__(self, n_static, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, num_layers=num_layers,
                          dropout=dropout, batch_first=True)
        self.bn = nn.BatchNorm1d(n_static)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, seq, static):
        # seq: (B, SEQ_LEN, 1), static: (B, F)
        _, h = self.gru(seq)
        h_last = h[-1]
        s = self.bn(static)
        return self.head(torch.cat([h_last, s], dim=-1)).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# 4. K-FOLD TRAINING (matches v18 CV split — StratifiedKFold on log_y deciles)
# ══════════════════════════════════════════════════════════════════════════════

y_orig     = orig_train["next_3m_txn_count"].values
y_log_orig = np.log1p(y_orig)
y_bins     = pd.qcut(y_log_orig, q=10, labels=False, duplicates="drop")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_nn  = np.zeros(len(orig_train))
test_nn = np.zeros(len(test_df))
sc_nn   = []

print(f"\nTraining {N_FOLDS}-fold GRU ...\n{'─'*70}")

for fold, (tr_idx, va_idx) in enumerate(skf.split(orig_train, y_bins), 1):
    tr_orig = orig_train.iloc[tr_idx]
    tr_aug  = combined[combined["sample_weight"] != 1.0]
    tr_fold = pd.concat([tr_orig, tr_aug], ignore_index=True)
    va_orig = orig_train.iloc[va_idx]

    seq_tr, st_tr = to_arrays(tr_fold)
    seq_va, st_va = to_arrays(va_orig)
    seq_te, st_te = to_arrays(test_df)

    sc = StandardScaler().fit(st_tr)
    st_tr = sc.transform(st_tr).astype(np.float32)
    st_va = sc.transform(st_va).astype(np.float32)
    st_te = sc.transform(st_te).astype(np.float32)

    seq_tr_t = torch.from_numpy(seq_tr).unsqueeze(-1).to(DEVICE)
    st_tr_t  = torch.from_numpy(st_tr).to(DEVICE)
    y_tr_t   = torch.from_numpy(np.log1p(tr_fold["next_3m_txn_count"].values).astype(np.float32)).to(DEVICE)
    w_tr_t   = torch.from_numpy(tr_fold["sample_weight"].values.astype(np.float32)).to(DEVICE)

    seq_va_t = torch.from_numpy(seq_va).unsqueeze(-1).to(DEVICE)
    st_va_t  = torch.from_numpy(st_va).to(DEVICE)
    y_va     = y_orig[va_idx]

    seq_te_t = torch.from_numpy(seq_te).unsqueeze(-1).to(DEVICE)
    st_te_t  = torch.from_numpy(st_te).to(DEVICE)

    model = GruRegressor(n_static=len(STATIC_COLS), hidden=64, num_layers=2, dropout=0.2).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)

    best_rmsle = 1e9
    best_state = None
    patience   = 0
    PATIENCE_LIMIT = 8
    BATCH = 512

    for epoch in range(60):
        model.train()
        perm = torch.randperm(seq_tr_t.size(0), device=DEVICE)
        for i in range(0, perm.size(0), BATCH):
            idx = perm[i:i+BATCH]
            pred = model(seq_tr_t[idx], st_tr_t[idx])
            loss = (w_tr_t[idx] * (pred - y_tr_t[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            p_va = model(seq_va_t, st_va_t).cpu().numpy()
        score = rmsle(y_va, np.expm1(np.clip(p_va, 0, None)))
        if score < best_rmsle - 1e-5:
            best_rmsle = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE_LIMIT:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_va = model(seq_va_t, st_va_t).cpu().numpy()
        p_te = model(seq_te_t, st_te_t).cpu().numpy()
    o_va = np.clip(np.expm1(np.clip(p_va, 0, None)), 0, None)
    o_te = np.clip(np.expm1(np.clip(p_te, 0, None)), 0, None)
    oof_nn[va_idx] = o_va
    test_nn      += o_te / N_FOLDS
    sc_nn.append(best_rmsle)
    print(f"  Fold {fold}  NN rmsle={best_rmsle:.5f}  (epoch_best capped at 60)")

cv_nn = rmsle(y_orig, oof_nn)
print(f"{'─'*70}")
print(f"  CV  NN  = {cv_nn:.5f}  ({np.mean(sc_nn):.5f}±{np.std(sc_nn):.5f})")


# ══════════════════════════════════════════════════════════════════════════════
# 5. BLEND WITH v18 (calibrated blend as a single base predictor)
# ══════════════════════════════════════════════════════════════════════════════

print("\nLoading v18 OOF and test predictions ...")
oof_v18 = pd.read_csv(DATA_DIR / "oof_v18.csv")
sub_v18 = pd.read_csv(DATA_DIR / "submission_v18.csv")

# Align by UniqueID order with orig_train and test_df
oof_v18 = orig_train[["UniqueID"]].merge(oof_v18[["UniqueID","blend_cal"]], on="UniqueID", how="left")
sub_v18 = test_df[["UniqueID"]].merge(sub_v18[["UniqueID","next_3m_txn_count"]], on="UniqueID", how="left")
v18_oof_count  = oof_v18["blend_cal"].values
v18_test_count = np.expm1(sub_v18["next_3m_txn_count"].values)  # submission is log1p

cv_v18 = rmsle(y_orig, v18_oof_count)
print(f"  v18 (blend_cal) OOF = {cv_v18:.5f}")
print(f"  NN              OOF = {cv_nn:.5f}")

# Correlation in log space
v18_log = np.log1p(np.clip(v18_oof_count, 0, None))
nn_log  = np.log1p(np.clip(oof_nn, 0, None))
corr = np.corrcoef(v18_log, nn_log)[0, 1]
print(f"  Corr(v18, NN) in log space = {corr:.5f}")

# NNLS over [v18, nn] in log space
P_oof  = np.column_stack([v18_log, nn_log])
P_test = np.column_stack([
    np.log1p(np.clip(v18_test_count, 0, None)),
    np.log1p(np.clip(test_nn,         0, None)),
])
w, _ = nnls(P_oof, y_log_orig)
w_norm = w / (w.sum() + 1e-12)
print(f"  NNLS weights (normalized): v18={w_norm[0]:.3f}  NN={w_norm[1]:.3f}  (sum_raw={w.sum():.3f})")

oof_blend_log  = P_oof  @ w
test_blend_log = P_test @ w
oof_blend  = np.expm1(np.clip(oof_blend_log,  0, None))
test_blend = np.clip(np.expm1(np.clip(test_blend_log, 0, None)), 0, None)

cv_blend = rmsle(y_orig, oof_blend)

# Linear cal in log space (2 params, safe)
lin_b, lin_a = np.polyfit(oof_blend_log, y_log_orig, 1)
oof_cal_log  = lin_a + lin_b * oof_blend_log
test_cal_log = lin_a + lin_b * test_blend_log
oof_cal  = np.expm1(np.clip(oof_cal_log,  0, None))
test_cal = np.clip(np.expm1(np.clip(test_cal_log, 0, None)), 0, None)
cv_cal = rmsle(y_orig, oof_cal)

print(f"\n  v18 alone     CV = {cv_v18:.5f}")
print(f"  NN alone      CV = {cv_nn:.5f}")
print(f"  Blend NNLS    CV = {cv_blend:.5f}")
print(f"  Linear cal    CV = {cv_cal:.5f}  (b={lin_b:.4f}, a={lin_a:+.4f})")


# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════════════════

# Pick the final test predictions: prefer linear-calibrated blend, but if NN
# was zero-weighted, fall back to v18 (no point recalibrating an unchanged blend)
if w_norm[1] > 0.001:
    final_test = test_cal
    final_cv   = cv_cal
    label = f"NN+v18 NNLS+lincal (NN weight={w_norm[1]:.3f})"
else:
    final_test = v18_test_count
    final_cv   = cv_v18
    label = "NN zeroed by NNLS — falling back to v18"

sub = test_df[["UniqueID"]].copy()
sub["next_3m_txn_count"] = np.log1p(np.clip(final_test, 0, None))
sub.to_csv(DATA_DIR / "submission_v19.csv", index=False)
print(f"\n  Saved  submission_v19.csv  |  CV={final_cv:.5f}  ({label})")

oof_out = orig_train[["UniqueID","next_3m_txn_count"]].copy()
oof_out["v18"]       = np.clip(v18_oof_count, 0, None)
oof_out["nn"]        = np.clip(oof_nn,        0, None)
oof_out["blend"]     = np.clip(oof_blend,     0, None)
oof_out["blend_cal"] = np.clip(oof_cal,       0, None)
oof_out.to_csv(DATA_DIR / "oof_v19.csv", index=False)
print(f"  Saved  oof_v19.csv\nDone.")
