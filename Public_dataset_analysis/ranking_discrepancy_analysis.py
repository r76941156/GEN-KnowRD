# Databricks notebook source
# MAGIC %md
# MAGIC ### Gap Analysis between top1/top2

# COMMAND ----------

df = spark.read.option("header", "true").csv("/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/rank_top2_eval_summary_gemma.csv")
df.createOrReplaceTempView("rank_top2_eval_summary_view")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from rank_top2_eval_summary_view

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct disease_1,disease_2 from rank_top2_eval_summary_view where relation_type='broader-narrower'

# COMMAND ----------

from pyspark.sql import functions as F
import matplotlib.pyplot as plt

# === Step 1: Normalize relation_type and count ===
df = spark.table("rank_top2_eval_summary_view")

df_grouped = (
    df.withColumn(
        "normalized_relation_type",
        F.when(
            F.col("relation_type").isin("etiologic-related", "etiological-related"),
            "etiologic/etiological-related"
        ).otherwise(F.col("relation_type"))
    )
    .groupBy("normalized_relation_type")
    .agg(F.count("*").alias("case_count"))
)

# === Step 2: Convert to pandas ===
pdf = df_grouped.toPandas()
total = pdf["case_count"].sum()
pdf["percentage"] = pdf["case_count"] / total * 100
pdf["label"] = pdf.apply(lambda row: f"{row['normalized_relation_type']}\n{row['percentage']:.1f}%\nn={row['case_count']}", axis=1)

# === Step 3: Plot Pie Chart ===
plt.figure(figsize=(8, 8))
plt.pie(
    pdf["case_count"],
    labels=pdf["label"],
    autopct=None,
    startangle=140,
    textprops={"fontsize": 10}
)
plt.title("Relation Type Distribution (Top-2 Hits)", fontsize=14)
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
plt.tight_layout()
plt.show()


# COMMAND ----------

df2 = spark.read.option("header", "true").csv("/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/topk_eval_all_models.csv")
df2.createOrReplaceTempView("topk_eval_all_models_view")

# COMMAND ----------

# MAGIC %sql
# MAGIC select model,topk,accuracy as recall from topk_eval_all_models_view 
# MAGIC where group = 'all' and topk in ('@1','@2','@3')
# MAGIC order by topk

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Data ===
data = [
    ("claude", "@1", 0.8563),
    ("gemini", "@1", 0.8444),
    ("ds", "@1", 0.8428),
    ("o3", "@1", 0.8429),
    ("claude", "@2", 0.9103),
    ("gemini", "@2", 0.9028),
    ("ds", "@2", 0.9048),
    ("o3", "@2", 0.9105),
    ("claude", "@3", 0.9239),
    ("gemini", "@3", 0.9196),
    ("ds", "@3", 0.9217),
    ("o3", "@3", 0.9252),
]

df = pd.DataFrame(data, columns=["Model", "TopK", "Recall"])

# === Plotting setup ===
sns.set(style="whitegrid")
models = ["claude", "gemini", "ds", "o3"]
topk_order = ["@1", "@2", "@3"]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, model in enumerate(models):
    ax = axes[i]
    subset = df[df["Model"] == model]
    sns.barplot(data=subset, x="TopK", y="Recall", order=topk_order, ax=ax)

    # Annotate values
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2, height),
                    ha='center', va='bottom', fontsize=10)

    ax.set_ylim(0.83, 0.93)
    ax.set_title(f"{model.capitalize()} Recall@1/2/3", fontsize=14)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Top-K")

fig.suptitle("Recall@1/2/3 Per LLM", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
