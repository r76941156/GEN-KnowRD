# Databricks notebook source
pip install tqdm

# COMMAND ----------

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# 1. CONFIG
# ===============================

model_files = {
    "claude": "/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/all_top3_predictions_claude.csv",
    "gemini": "/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/all_top3_predictions_gemini.csv",
    "ds":     "/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/all_top3_predictions_ds.csv",
    "o3":     "/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/all_top3_predictions_o3.csv",
}

# ↘ tie-breaker priority. Different settings are similar in results.
model_priority = ["claude", "gemini", "ds", "o3"]
output_csv = "all_top3_predictions_gated.csv"

# ===============================
# 2. Load all CSV
# ===============================

dfs = {}
for model, file in model_files.items():
    df = pd.read_csv(file, dtype=str)
    df["rank"] = df["rank"].astype(int)
    dfs[model] = df

# merge keys
all_pairs = dfs["claude"][["patient_id", "dataset"]].drop_duplicates()

# ===============================
# 3. Function: build ranking vector per model
# ===============================

def build_rank_vector(model_df, patient_id, dataset, disease_universe):
    rows = model_df[(model_df["patient_id"] == patient_id) & 
                    (model_df["dataset"] == dataset)]

    rank_map = {row["predicted_disease"].strip().lower(): row["rank"]
                for _, row in rows.iterrows()}

    max_rank = len(disease_universe) + 1
    ranks = [rank_map.get(d.lower(), max_rank) for d in disease_universe]
    return ranks

# ===============================
# 4. Spearman Consensus Gating
# ===============================

results = []
gated_model_counts = Counter()

print("🔍 Running Spearman consensus gating ...")

for _, row in tqdm(all_pairs.iterrows(), total=len(all_pairs), desc="Processing patients"):

    patient_id = row["patient_id"]
    dataset = row["dataset"]

    # --- get disease universe
    disease_universe = set()
    for model, df in dfs.items():
        subset = df[(df["patient_id"] == patient_id) & (df["dataset"] == dataset)]
        disease_universe.update(subset["predicted_disease"].astype(str).tolist())

    disease_universe = list(disease_universe)

    # --- build rank vectors
    model_ranks = {
        model: build_rank_vector(df, patient_id, dataset, disease_universe)
        for model, df in dfs.items()
    }

    # --- compute consensus scores
    consensus_scores = {m: 0.0 for m in model_files.keys()}

    for m1 in model_files.keys():
        for m2 in model_files.keys():
            if m1 == m2:
                continue
            rho, _ = spearmanr(model_ranks[m1], model_ranks[m2])
            if np.isnan(rho):
                rho = 0.0
            consensus_scores[m1] += rho

    # --- pick best gated model (tie-breaker)
    best_score = max(consensus_scores.values())
    tied_models = [m for m, s in consensus_scores.items() if s == best_score]

    for pref in model_priority:
        if pref in tied_models:
            gated_model = pref
            break

    gated_model_counts[gated_model] += 1

    # --- add 3 predicted rows
    model_df = dfs[gated_model]
    subset = model_df[(model_df["patient_id"] == patient_id) & 
                      (model_df["dataset"] == dataset)]

    for _, r in subset.iterrows():
        results.append({
            "patient_id": patient_id,
            "dataset": dataset,
            "rank": r["rank"],
            "disease": r["predicted_disease"],
            "gated_model": gated_model
        })

# ===============================
# 5. Save output
# ===============================

out_df = pd.DataFrame(results)
out_df.to_csv(output_csv, index=False)
print(f"\n✅ Gated results saved to {output_csv}")

# ===============================
# 6. Print model selection stats
# ===============================

print("\n📊 Gated Model Selection Count (by patient_id + dataset):")
for model in model_priority:
    count = gated_model_counts[model]
    print(f"  {model:7s} → {count}")