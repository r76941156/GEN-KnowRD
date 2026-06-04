# Databricks notebook source
# MAGIC %md
# MAGIC ### Model Summary

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, explode, lit, when, countDistinct, lower
)

spark = SparkSession.builder.getOrCreate()

# ==============================================
# 1) Define semantic groups
# ==============================================
grouped_mappings = {
    "Drug or Procedure": [
        "Pharmacologic Substance", "Clinical Drug",
        "Therapeutic or Preventive Procedure", "Organic Chemical"
    ],
    "Genetics & Molecular": [
        "Gene or Genome", "Genetic Function",
        "Amino Acid, Peptide, or Protein", "Enzyme",
        "Cell or Molecular Dysfunction"
    ],
    "Diagnostics & Lab": [
        "Diagnostic Procedure", "Laboratory Procedure",
        "Laboratory or Test Result"
    ],
    "Symptoms & Conditions": [
        "Disease or Syndrome", "Sign or Symptom", "Finding",
        "Pathologic Function", "Neoplastic Process",
        "Mental or Behavioral Dysfunction",
        "Congenital Abnormality", "Anatomical Abnormality"
    ]
}

# flatten semantic type list
target_types = [st for group in grouped_mappings.values() for st in group]

# ==============================================
# 2) Function to load + filter + group
# ==============================================
def with_group_mapping(model):
    df = (
        spark.table(f"wei_lab_sander.llm_{model}_umls_matched_results")
            .filter(col("similarity") >= 0.8)
            .withColumn("model", lit(model))
            .withColumn("semtypes", split(col("semantic_type_name"), "\\|"))
            .withColumn("semtype", explode(col("semtypes")))
            .filter(col("semtype").isin(target_types))
    )

    # === Remove negated UMLS names ===
    df = df.filter(
        ~(
            lower(col("UMLS_name")).like("not %") |
            lower(col("UMLS_name")).like("no %") |
            lower(col("UMLS_name")).like("without %")
        )
    )

    # map semantic group
    df = (
        df.withColumn(
            "group_name",
            when(col("semtype").isin(grouped_mappings["Drug or Procedure"]), "Drug or Procedure")
            .when(col("semtype").isin(grouped_mappings["Genetics & Molecular"]), "Genetics & Molecular")
            .when(col("semtype").isin(grouped_mappings["Diagnostics & Lab"]), "Diagnostics & Lab")
            .when(col("semtype").isin(grouped_mappings["Symptoms & Conditions"]), "Symptoms & Conditions")
        )
    )

    return df.select("model", "group_name", "UMLS_name", "CUI", "disease").dropDuplicates()

# ==============================================
# 3) Load all models
# ==============================================
models = ["ro", "ds", "o3", "claude","gemini"]

combined_df = None
for m in models:
    model_df = with_group_mapping(m)
    combined_df = model_df if combined_df is None else combined_df.unionByName(model_df)

# ==============================================
# 4) Count distinct CUI + distinct diseases
# ==============================================
summary = (
    combined_df.groupBy("group_name", "model")
    .agg(
        countDistinct("CUI").alias("distinct_UMLS_CUI_count"),
        countDistinct("disease").alias("distinct_disease_count")
    )
)

display(summary)

# ==============================================
# Convert Spark DF → Pandas for plotting
# ==============================================
pdf = summary.toPandas()

group_order = [
    "Symptoms & Conditions",
    "Genetics & Molecular",
    "Drug or Procedure",
    "Diagnostics & Lab"
]

pdf["group_name"] = pd.Categorical(pdf["group_name"], categories=group_order, ordered=True)
pdf = pdf.sort_values(["group_name", "model"])

# ==============================================
# Plot: Horizontal Grouped Bar Chart (bright colors + fixed order)
# ==============================================
import matplotlib.pyplot as plt
import numpy as np

# Brightened color palette
color_map = {
    "ro":     "#2D8CDA",   # brighter blue
    "ds":     "#FFBB33",   # brighter orange
    "o3":     "#38DFA2",   # brighter green
    "claude": "#FF7F33",   # brighter red-orange
    "gemini": "#7FCBFF"    # brighter sky blue
}

# fixed model order (top→bottom inside each group)
models = ["claude", "gemini","ds","o3","ro"]

label_map = {
    "ro": "Rarediseases.org",
    "ds": "DeepSeek R1",
    "o3": "OpenAI o3",
    "claude": "Claude-Sonnet-4",
    "gemini": "Gemini-2.5 Pro"
}



fig, ax = plt.subplots(figsize=(18, 8))

bar_height = 0.15
y_positions = np.arange(len(group_order))

for i, model in enumerate(models):
    sub = pdf[pdf["model"] == model]

    counts = sub["distinct_UMLS_CUI_count"].values
    disease_counts = sub["distinct_disease_count"].values

    # Draw bars
  

    ax.barh(
    y_positions + i * bar_height,
    counts,
    height=bar_height,
    color=color_map[model],
    label=label_map.get(model, model)   # use pretty legend name
    )

    # Labels (CUI / disease)
    for y, c, d in zip(y_positions, counts, disease_counts):
        ax.text(
            c + 200,
            y + i * bar_height,
            f"{c:,} / {d:,}",
            va="center",
            fontsize=10
        )

# Fix Y-axis positions to place groups in correct vertical order
ax.set_yticks(y_positions + bar_height * (len(models)-1)/2)
ax.set_yticklabels(group_order)
ax.invert_yaxis()  # ensure order: Symptoms, Genetics, Drug, Diagnostics (top → bottom)

ax.set_xlabel("Distinct UMLS CUI / Disease Count")
ax.set_title("Distinct UMLS CUIs and Disease Coverage by Model and UMLS Group")

ax.legend(
    loc="lower right",        # bottom-right position
    #bbox_to_anchor=(1.0, -0.05),   # slight shift downward
    title="Model",
    #ncol=5,                   # put all model labels in one row
    #frameon=False             # cleaner style (optional)
)


plt.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# COMMAND ----------

# ============================================================
# STEP 1 — PySpark: Load + Filter + Group by Semantic Group
# ============================================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, explode, lit, when, countDistinct, count, avg, stddev, lower
)

spark = SparkSession.builder.getOrCreate()

# ============================================================
# Semantic group definitions
# ============================================================
grouped_mappings = {
    "Drug or Procedure": [
        "Pharmacologic Substance", "Clinical Drug",
        "Therapeutic or Preventive Procedure", "Organic Chemical"
    ],
    "Genetics & Molecular": [
        "Gene or Genome", "Genetic Function",
        "Amino Acid, Peptide, or Protein", "Enzyme",
        "Cell or Molecular Dysfunction"
    ],
    "Diagnostics & Lab": [
        "Diagnostic Procedure", "Laboratory Procedure",
        "Laboratory or Test Result"
    ],
    "Symptoms & Conditions": [
        "Disease or Syndrome", "Sign or Symptom", "Finding",
        "Pathologic Function", "Neoplastic Process",
        "Mental or Behavioral Dysfunction",
        "Congenital Abnormality", "Anatomical Abnormality"
    ]
}

target_types = [st for group in grouped_mappings.values() for st in group]

# ============================================================
# Function to load and tag each row with semantic group
# ============================================================
def with_group_mapping(model):
    df = (
        spark.table(f"wei_lab_sander.llm_{model}_umls_matched_results")
        .filter(col("similarity") >= 0.8)
        .withColumn("model", lit(model))
        .withColumn("semtypes", split(col("semantic_type_name"), "\\|"))
        .withColumn("semtype", explode(col("semtypes")))
        .filter(col("semtype").isin(target_types))
        .withColumn(
            "group_name",
            when(col("semtype").isin(grouped_mappings["Drug or Procedure"]), "Drug or Procedure")
            .when(col("semtype").isin(grouped_mappings["Genetics & Molecular"]), "Genetics & Molecular")
            .when(col("semtype").isin(grouped_mappings["Diagnostics & Lab"]), "Diagnostics & Lab")
            .when(col("semtype").isin(grouped_mappings["Symptoms & Conditions"]), "Symptoms & Conditions")
        )
    )
    
    # remove negated terms
    df = df.filter(
        ~(
            lower(col("UMLS_name")).like("not %") |
            lower(col("UMLS_name")).like("no %") |
            lower(col("UMLS_name")).like("without %")
        )
    )
    
    return df.select("model", "group_name", "UMLS_name", "CUI", "disease").dropDuplicates()

# ============================================================
# Load all models
# ============================================================
models = ["ro", "ds", "o3", "claude", "gemini"]

combined_df = None
for m in models:
    model_df = with_group_mapping(m)
    combined_df = model_df if combined_df is None else combined_df.unionByName(model_df)

# ============================================================
# STEP 2 — Count UMLS per disease for each model × group
# ============================================================
disease_level_df = combined_df.groupBy("model", "group_name", "disease").agg(
    countDistinct("CUI").alias("umls_count")
)

# ============================================================
# STEP 3 — Summary stats per model × group
# ============================================================
disease_summary_stats = disease_level_df.groupBy("model", "group_name").agg(
    count("disease").alias("n_diseases"),
    avg("umls_count").alias("avg_umls_per_disease"),
    stddev("umls_count").alias("std_umls_per_disease")
)

# ============================================================
# STEP 4 — Convert to Pandas for plotting
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pdf = disease_summary_stats.toPandas()

# enforce semantic group order
group_order = [
    "Symptoms & Conditions",
    "Genetics & Molecular",
    "Drug or Procedure",
    "Diagnostics & Lab"
]

pdf["group_name"] = pd.Categorical(
    pdf["group_name"], categories=group_order, ordered=True
)

# sort within each group by average
pdf.sort_values(["group_name", "avg_umls_per_disease"], ascending=[True, False], inplace=True)

# ============================================================
# Label map for x-axis
# ============================================================
label_map = {
    "ro": "Rarediseases.org",
    "ds": "DeepSeek R1",
    "o3": "OpenAI o3",
    "claude": "Claude-Sonnet-4",
    "gemini": "Gemini-2.5 Pro"
}

# fixed model order
model_order = ["ro", "ds", "o3", "claude", "gemini"]

# ============================================================
# STEP 5 — Plot: 2×2 grid of semantic groups
# ============================================================

groups = pdf["group_name"].unique()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()  # flatten to index easily

for idx, group in enumerate(groups):

    ax = axes[idx]

    # filter this semantic group
    sub = pdf[pdf["group_name"] == group]

    # enforce consistent model order
    sub = sub.set_index("model").loc[model_order].reset_index()

    x = np.arange(len(sub))

    bars = ax.bar(
        x,
        sub["avg_umls_per_disease"],
        yerr=sub["std_umls_per_disease"],
        capsize=5,
        color="skyblue"
    )

    # Pretty x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels([label_map[m] for m in sub["model"]], rotation=45, ha="right")

    ax.set_ylabel("Avg UMLS per Disease")

    # Title with (n=xxx)
    title_text = (
        f"{group} — Avg UMLS per Disease"
        # + ", ".join(
        #     [f"{label_map[row['model']]} (n={int(row['n_diseases'])})"
        #      for _, row in sub.iterrows()]
        #)
    )
    ax.set_title(title_text, fontsize=12)

    # label each bar — shifted to the right
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width()/2, h),
            xytext=(5, 3),   # shifted right & up
            textcoords="offset points",
            ha="left",
            fontsize=9
        )

# Remove unused subplot if fewer than 4 groups (not needed but safe)
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, split, explode, countDistinct, lower
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ✅ Spark session
spark = SparkSession.builder.getOrCreate()

# ✅ Target semantic types
target_types = [
    "Disease or Syndrome", "Gene or Genome", "Therapeutic or Preventive Procedure",
    "Pathologic Function", "Diagnostic Procedure", "Sign or Symptom", "Neoplastic Process",
    "Organic Chemical", "Laboratory Procedure", "Congenital Abnormality",
    "Mental or Behavioral Dysfunction", "Pharmacologic Substance", "Genetic Function",
    "Amino Acid, Peptide, or Protein", "Anatomical Abnormality", "Laboratory or Test Result",
    "Clinical Drug", "Enzyme", "Cell or Molecular Dysfunction", "Finding"
]

# ================================================================
# Label map for final heatmap columns
# ================================================================
label_map = {
    "ro": "Rarediseases.org",
    "ds": "DeepSeek R1",
    "o3": "OpenAI o3",
    "claude": "Claude-Sonnet-4",
    "gemini": "Gemini-2.5 Pro"
}

# ==========================================================================================
# Load and clean each model
# ==========================================================================================
def load_filtered(model_name, table_name):
    df = spark.table(table_name).filter(col("similarity") >= 0.8)

    # remove negated UMLS names
    df = df.filter(
        ~(
            lower(col("UMLS_name")).like("not %") |
            lower(col("UMLS_name")).like("no %") |
            lower(col("UMLS_name")).like("without %")
        )
    )

    df = df.withColumn("semantic_type_array", split(col("semantic_type_name"), "\\|"))
    df = df.withColumn("model", lit(model_name))

    # USE CUI instead of UMLS_name
    return df.select("CUI", "model", explode(col("semantic_type_array")).alias("semantic_type"))

# ==========================================================================================
# Combine the 5 models
# ==========================================================================================
df_all = (
    load_filtered("claude", "wei_lab_sander.llm_claude_umls_matched_results")
    .unionByName(load_filtered("ds", "wei_lab_sander.llm_ds_umls_matched_results"))
    .unionByName(load_filtered("o3", "wei_lab_sander.llm_o3_umls_matched_results"))
    .unionByName(load_filtered("ro", "wei_lab_sander.llm_ro_umls_matched_results"))
    .unionByName(load_filtered("gemini", "wei_lab_sander.llm_gemini_umls_matched_results"))
)

# ==========================================================================================
# Count distinct CUIs per model × semantic type
# ==========================================================================================
agg_df = (
    df_all.filter(col("semantic_type").isin(target_types))
          .groupBy("model", "semantic_type")
          .agg(countDistinct("CUI").alias("cui_count"))
)

# ==========================================================================================
# Convert to Pandas and pivot
# ==========================================================================================
pdf = agg_df.toPandas()

pivot_df = (
    pdf
    .pivot(index="semantic_type", columns="model", values="cui_count")
    .fillna(0)
)

# === APPLY label_map to the column names ===
pivot_df = pivot_df.rename(columns=label_map)

# ==========================================================================================
# Heatmap
# ==========================================================================================
plt.figure(figsize=(10, 8))
sns.heatmap(
    pivot_df,
    annot=True, fmt=".0f", cmap="YlGnBu",
    cbar_kws={'label': 'Number of Unique CUIs'}
)

plt.title("CUI Coverage by Model × Semantic Type (All Diseases)")
plt.xlabel("Model")
plt.ylabel("Semantic Type")
plt.tight_layout()
plt.show()


# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, split, explode, lower
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# ------------------------------------------------------------
# Spark session
# ------------------------------------------------------------
spark = SparkSession.builder.getOrCreate()

# ------------------------------------------------------------
# Label map for nicer names
# ------------------------------------------------------------
label_map = {
    "ro": "Rarediseases.org",
    "ds": "DeepSeek R1",
    "o3": "OpenAI o3",
    "claude": "Claude-Sonnet-4",
    "gemini": "Gemini-2.5 Pro"
}

# ------------------------------------------------------------
# Category definitions
# ------------------------------------------------------------
category_map = {
    "Symptoms & Conditions": [
        "Disease or Syndrome", "Sign or Symptom", "Finding",
        "Pathologic Function", "Neoplastic Process",
        "Mental or Behavioral Dysfunction", "Congenital Abnormality",
        "Anatomical Abnormality"
    ],
    "Diagnostics & Lab": [
        "Diagnostic Procedure", "Laboratory Procedure", "Laboratory or Test Result"
    ],
    "Drug or Procedure": [
        "Pharmacologic Substance", "Clinical Drug",
        "Therapeutic or Preventive Procedure", "Organic Chemical"
    ],
    "Genetics & Molecular": [
        "Gene or Genome", "Genetic Function",
        "Amino Acid, Peptide, or Protein", "Enzyme",
        "Cell or Molecular Dysfunction"
    ]
}

# ------------------------------------------------------------
# Load + filter + explode semantic types (USES CUI + NEGATION FILTERING)
# ------------------------------------------------------------
def load_with_type(model_name, table_name):

    df = spark.table(table_name).filter(col("similarity") >= 0.8)

    # ❗ Remove negated UMLS names
    df = df.filter(
        ~(
            lower(col("UMLS_name")).like("not %")
            | lower(col("UMLS_name")).like("no %")
            | lower(col("UMLS_name")).like("without %")
        )
    )

    df = df.withColumn("model", lit(model_name))
    df = df.withColumn("semantic_type_array", split(col("semantic_type_name"), "\\|"))

    return df.select(
        "CUI",
        "model",
        explode(col("semantic_type_array")).alias("semantic_type")
    ).dropDuplicates()

# ------------------------------------------------------------
# Load all models
# ------------------------------------------------------------
df_all = (
    load_with_type("claude", "wei_lab_sander.llm_claude_umls_matched_results")
    .unionByName(load_with_type("ds", "wei_lab_sander.llm_ds_umls_matched_results"))
    .unionByName(load_with_type("o3", "wei_lab_sander.llm_o3_umls_matched_results"))
    .unionByName(load_with_type("ro", "wei_lab_sander.llm_ro_umls_matched_results"))
    .unionByName(load_with_type("gemini", "wei_lab_sander.llm_gemini_umls_matched_results"))
)

# ------------------------------------------------------------
# Compute Jaccard similarity (using CUI sets)
# ------------------------------------------------------------
def compute_jaccard(df):

    pdf = df.toPandas()

    # Convert to sets of CUIs
    model_sets = {
        m: set(pdf.loc[pdf["model"] == m, "CUI"])
        for m in pdf["model"].unique()
    }

    models = list(model_sets.keys())
    jmat = pd.DataFrame(index=models, columns=models, dtype=float)

    for m1, m2 in combinations(models, 2):
        s1, s2 = model_sets[m1], model_sets[m2]
        inter, union = len(s1 & s2), len(s1 | s2)
        j = inter / union if union > 0 else 0
        jmat.loc[m1, m2] = j
        jmat.loc[m2, m1] = j

    for m in models:
        jmat.loc[m, m] = 1.0

    # Replace indices and columns using label_map
    jmat = jmat.rename(index=label_map, columns=label_map)

    return jmat.astype(float)


# ============================================================
# dataframe to store ALL jaccard similarities
# ============================================================
similarity_records = []


# ------------------------------------------------------------
# Plot 2×2 heatmaps
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, (cat, types) in zip(axes.flat, category_map.items()):

    df_sub = (
        df_all.filter(col("semantic_type").isin(types))
              .select("CUI", "model")
              .dropDuplicates()
    )

    if df_sub.count() == 0:
        ax.set_visible(False)
        continue

    jmat = compute_jaccard(df_sub)

    # ----------------------------
    # Save similarity into list
    # ----------------------------
    for m1 in jmat.index:
        for m2 in jmat.columns:
            similarity_records.append({
                "semantic_group": cat,
                "model_a": m1,
                "model_b": m2,
                "jaccard": float(jmat.loc[m1, m2])
            })

    sns.heatmap(
        jmat,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "Jaccard Similarity"},
        ax=ax
    )

    ax.set_title(f"Jaccard Similarity Between Models ({cat})")
    ax.set_xlabel("Model")
    ax.set_ylabel("Model")

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Convert similarity records → dataframe
# ------------------------------------------------------------
similarity_df = pd.DataFrame(similarity_records)

# ============================================================
# Compute Mean and SD of Jaccard Similarity (EXCLUDING diagonal)
# ============================================================

# Remove diagonal entries (self = 1.00)
similarity_offdiag = similarity_df[similarity_df["model_a"] != similarity_df["model_b"]]

# Compute stats
group_stats = (
    spark.createDataFrame(similarity_offdiag)
    .groupBy("semantic_group")
    .agg(
        avg("jaccard").alias("mean_jaccard"),
        stddev("jaccard").alias("std_jaccard")
    )
)

group_stats_pdf = group_stats.toPandas()

print("=== Mean & SD of Jaccard Similarity (Off-Diagonal Only) ===")
display(group_stats_pdf)

# ============================================================
# Pairwise similarity *by semantic group*
# ============================================================

# Remove model=self diagonal entries
similarity_nondiag = similarity_df[
    similarity_df["model_a"] != similarity_df["model_b"]
].copy()

# Compute mean Jaccard per pair *within each semantic group*
group_pair_summary = (
    similarity_nondiag
    .groupby(["semantic_group", "model_a", "model_b"])["jaccard"]
    .mean()
    .reset_index()
)

# Remove duplicate unordered pairs (A,B) and (B,A)
group_pair_summary["pair"] = group_pair_summary.apply(
    lambda x: " & ".join(sorted([x["model_a"], x["model_b"]])),
    axis=1
)

group_pair_summary = (
    group_pair_summary
    .groupby(["semantic_group", "pair"])["jaccard"]
    .mean()
    .reset_index()
    .sort_values(["semantic_group", "jaccard"], ascending=[True, False])
)

print("\n=== Pairwise Jaccard Similarity Per Semantic Group ===")
display(group_pair_summary)
