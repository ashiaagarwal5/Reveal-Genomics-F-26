"""
Clean baseline: PCA(500) on raw gene expression -> Cox PH, patient-level train/test split.

Inputs (expected in this directory):
  - adata_tcga_preprocessed.h5ad   (log1p-normalized TCGA pan-cancer expression, sample x gene)
  - Survival_SupplementalTable_S1_20171025_xena_sp.txt  (TCGA pan-cancer survival table)

Outputs:
  - baseline_scaler.joblib
  - baseline_pca.joblib
  - baseline_coxph.joblib
  - baseline_patient_ids.joblib   (train/test patient id lists, for reproducibility)
"""

import warnings

import h5py
import numpy as np
import pandas as pd
import joblib

# macOS Accelerate BLAS backend raises spurious divide-by-zero/overflow/invalid
# RuntimeWarnings during matmul-heavy routines (e.g. PCA's randomized SVD) even
# when inputs/outputs contain no NaN/Inf. Silence just these during PCA fit.
warnings.filterwarnings(
    "ignore", category=RuntimeWarning,
    message="(divide by zero|overflow|invalid value) encountered in matmul",
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError

H5AD_PATH = "adata_tcga_preprocessed.h5ad"
SURVIVAL_PATH = "Survival_SupplementalTable_S1_20171025_xena_sp.txt"
N_COMPONENTS = 500
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load expression matrix (sample-level), aggregate to one row per patient
# ---------------------------------------------------------------------------
print("Loading expression matrix...")
with h5py.File(H5AD_PATH, "r") as f:
    X = f["X"][:].astype(np.float32)
    sample_barcodes = np.array(
        [b.decode() if isinstance(b, bytes) else b for b in f["obs/_index"][:]]
    )

print(f"  expression matrix: {X.shape[0]} samples x {X.shape[1]} genes")

patient_id = pd.Series(sample_barcodes).str[:12].values
expr_df = pd.DataFrame(X, index=sample_barcodes)
expr_df["patient_id"] = patient_id

print("Aggregating to one row per patient (mean over samples)...")
patient_expr_df = expr_df.groupby("patient_id").mean()
print(f"  patient-level expression: {patient_expr_df.shape[0]} patients x {patient_expr_df.shape[1]} genes")

n_before = patient_expr_df.shape[1]
patient_expr_df = patient_expr_df.dropna(axis=1)
print(f"  dropped {n_before - patient_expr_df.shape[1]} genes with missing values (matches autoencoder notebook's convention)")

# ---------------------------------------------------------------------------
# 2. Load survival data, merge
# ---------------------------------------------------------------------------
print("Loading survival data...")
survival_df = pd.read_csv(SURVIVAL_PATH, sep="\t")
survival_df = survival_df.rename(columns={"_PATIENT": "patient_id"})
survival_df = survival_df[["patient_id", "cancer type abbreviation", "OS.time", "OS"]]
survival_df = survival_df.drop_duplicates(subset="patient_id")
survival_df = survival_df.dropna(subset=["OS.time", "OS"])

merged = patient_expr_df.merge(survival_df.set_index("patient_id"), left_index=True, right_index=True, how="inner")
merged = merged.dropna(subset=["OS.time", "OS"])
print(f"  merged patient-level dataset: {merged.shape[0]} patients")

gene_cols = [c for c in merged.columns if c not in ("cancer type abbreviation", "OS.time", "OS")]
assert len(gene_cols) == patient_expr_df.shape[1]

# ---------------------------------------------------------------------------
# 3. Train/test split by patient ID (80/20)
# ---------------------------------------------------------------------------
all_patient_ids = merged.index.values
train_ids, test_ids = train_test_split(
    all_patient_ids, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"Train patients: {len(train_ids)}  Test patients: {len(test_ids)}")

train_df = merged.loc[train_ids]
test_df = merged.loc[test_ids]

# ---------------------------------------------------------------------------
# 4. StandardScaler + PCA(500), fit on train only
# ---------------------------------------------------------------------------
print(f"Fitting StandardScaler + PCA({N_COMPONENTS}) on train patients only...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[gene_cols].values)
X_test_scaled = scaler.transform(test_df[gene_cols].values)

pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
X_train_pcs = pca.fit_transform(X_train_scaled)
X_test_pcs = pca.transform(X_test_scaled)
assert not np.isnan(X_train_pcs).any() and not np.isnan(X_test_pcs).any(), "NaN in PCA output"
print(f"  explained variance (cumulative, {N_COMPONENTS} PCs): {pca.explained_variance_ratio_.sum():.4f}")

pc_cols = [f"PC{i}" for i in range(N_COMPONENTS)]
train_pcs_df = pd.DataFrame(X_train_pcs, columns=pc_cols, index=train_df.index)
train_pcs_df["OS.time"] = train_df["OS.time"].values
train_pcs_df["OS"] = train_df["OS"].values

test_pcs_df = pd.DataFrame(X_test_pcs, columns=pc_cols, index=test_df.index)
test_pcs_df["OS.time"] = test_df["OS.time"].values
test_pcs_df["OS"] = test_df["OS"].values

# ---------------------------------------------------------------------------
# 5. Fit Cox model on train PCs; fall back to a small penalizer if it won't converge
# ---------------------------------------------------------------------------
print("Fitting Cox model on train PCs...")
penalizer = 0.0
try:
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_pcs_df, duration_col="OS.time", event_col="OS")
except ConvergenceError:
    penalizer = 0.1
    print(f"  convergence failed at penalizer=0.0, retrying with penalizer={penalizer}")
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train_pcs_df, duration_col="OS.time", event_col="OS")
print(f"  fit with penalizer={penalizer}")

# ---------------------------------------------------------------------------
# 6. Evaluate C-index on test set
# ---------------------------------------------------------------------------
c_index = cph.score(test_pcs_df, scoring_method="concordance_index")
print(f"\nTest C-index: {c_index:.4f}")

# ---------------------------------------------------------------------------
# 7. Save fitted objects
# ---------------------------------------------------------------------------
joblib.dump(scaler, "baseline_scaler.joblib")
joblib.dump(pca, "baseline_pca.joblib")
joblib.dump(cph, "baseline_coxph.joblib")
joblib.dump({"train_ids": list(train_ids), "test_ids": list(test_ids)}, "baseline_patient_ids.joblib")
print("\nSaved: baseline_scaler.joblib, baseline_pca.joblib, baseline_coxph.joblib, baseline_patient_ids.joblib")
