# Databricks notebook source
# MAGIC %md
# MAGIC ### FT embedding model ranking result

# COMMAND ----------


models = {
    "claude": "top10_predictions_claude.csv",
    "o3": "top10_predictions_o3.csv",
    "gemini": "top10_predictions_gemini.csv",
    "ds": "top10_predictions_ds.csv",
    "ro": "top10_predictions_ro.csv"
}

base_path = "/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset"

for model, filename in models.items():
    df = (
        spark.read
        .option("header", True)
        .csv(f"{base_path}/{filename}")
    )
    
    view_name = f"top10_{model}_embedding_patients"
    df.createOrReplaceTempView(view_name)
    
    print(f"Created temp view: {view_name}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### RRF calculation for Dense and Sparse ranks

# COMMAND ----------

from pyspark.sql.functions import col, lit, lower, trim, row_number, when
from pyspark.sql.window import Window


dataset_list = ["pubmed", "hms", "mme", "mygene2", "lirical", "ramedis"]
model_list = ["claude", "o3", "gemini", "ds","ro"]

k_rrf = 60  # RRF smoothing parameter

for dataset in dataset_list:
    for model in model_list:
        print(f"🔁 Processing fusion for model: {model}, dataset: {dataset}")

        # === Sparse / BM25 ===
        sparse_tbl = f"wei_lab_sander_mlflow.llm_{model}_{dataset}_top10_similar_diseases_bm25_02"
        sparse_df = (
            spark.table(sparse_tbl)
            .select("patient_id", "rare_disease_name", "disease", "rank",
                    "num_true_classifications", "final_classification_string")
            .withColumnRenamed("rank", "sparse_rank")
            .withColumn("disease_norm", lower(trim(col("disease"))))
            .withColumn("dataset", lit(dataset))
            .withColumn("model", lit(model))
        )

        # === ground truth ===
        truth_df = sparse_df.select(
            "patient_id", "dataset", "rare_disease_name",
            "num_true_classifications", "final_classification_string"
        ).dropDuplicates(["patient_id", "dataset"])

        sparse_df = sparse_df.drop("rare_disease_name", "num_true_classifications", "final_classification_string")

        # === Dense / Embedding ===
        dense_df = (
            spark.table(f"top10_{model}_embedding_patients")
            .filter(col("dataset") == dataset)
            .select("patient_id", "predicted_disease", "rerank_position")
            .withColumnRenamed("predicted_disease", "disease")
            .withColumnRenamed("rerank_position", "dense_rank")
            .withColumn("disease_norm", lower(trim(col("disease"))))
        )

        # === Join dense and sparse ===
        joined_df = (
            sparse_df.join(dense_df, on=["patient_id", "disease_norm"], how="outer")
            .withColumn("dataset", lit(dataset))
            .withColumn("model", lit(model))
        )

        # === RRF Fusion ===
        rrf_expr = (
            when(col("dense_rank").isNotNull(), 1 / (col("dense_rank") + k_rrf)).otherwise(0) +
            when(col("sparse_rank").isNotNull(), 1 / (col("sparse_rank") + k_rrf)).otherwise(0)
        )
        joined_df = joined_df.withColumn("rrf_score", rrf_expr)

      

        window_unique_rank = Window.partitionBy("patient_id").orderBy(
            col("rrf_score").desc(),                            # keep RRF coverage (top3/5)
            when(col("dense_rank").isNotNull(), 0).otherwise(1),# prefer dense-supported
            col("dense_rank").asc_nulls_last(),                 # 🔑 protect top1
            col("sparse_rank").asc_nulls_last()
        )

        final_df = joined_df.withColumn("final_rank", row_number().over(window_unique_rank))

        # === Join with ground truth ===
        final_df = final_df.join(truth_df, on=["patient_id", "dataset"], how="left")

        # === special case: mygene2 patient_id=4 ===
        if dataset == "mygene2":
            final_df = final_df \
                .withColumn(
                    "rare_disease_name",
                    when(col("patient_id") == "4",
                         lit("Arthrogryposis Renal Dysfunction Cholestasis Syndrome")
                    ).otherwise(col("rare_disease_name"))
                ).withColumn(
                    "num_true_classifications",
                    when(col("patient_id") == "4", lit(6)).otherwise(col("num_true_classifications"))
                ).withColumn(
                    "final_classification_string",
                    when(col("patient_id") == "4",
                         lit("developmental anomalies during embryogenesis, genetic diseases, hepatic diseases, inborn errors of metabolism, renal diseases, skin diseases, transplant-related disorders")
                    ).otherwise(col("final_classification_string"))
                )

        # === save result ===
        output_tbl = f"wei_lab_sander_mlflow.llm_{model}_{dataset}_top10_fusion_dense_sparse_rrf"
        spark.sql(f"DROP TABLE IF EXISTS {output_tbl}")
        (
            final_df.select(
                "patient_id", "dataset", "model", "rare_disease_name", "disease_norm",
                "dense_rank", "sparse_rank", "rrf_score", "final_rank",
                "num_true_classifications", "final_classification_string"
            )
            .write.mode("overwrite").format("delta")
            .saveAsTable(output_tbl)
        )

        print(f"✅ Saved to {output_tbl}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Combine datasets for Models

# COMMAND ----------

from pyspark.sql.utils import AnalysisException
from functools import reduce
from pyspark.sql.functions import lit


model_list = ["claude", "o3", "gemini", "ds","ro"]
dataset_list = ["pubmed", "hms", "mme", "mygene2", "lirical", "ramedis"]


# === Expected Schema (for consistency) ===
expected_cols = [
    "patient_id", "dataset", "model", "rare_disease_name", "disease_norm",
    "final_rank",
    "num_true_classifications", "final_classification_string"
]

def normalize_columns(df):
    """Add any missing expected columns as null, preserve order"""
    for c in expected_cols:
        if c not in df.columns:
            df = df.withColumn(c, lit(None))
    return df.select(expected_cols)

# === Main Loop ===
for model in model_list:
    print(f"🔁 Combining datasets for model: {model}")
    all_dfs = []

    for dataset in dataset_list:
        table_name = f"wei_lab_sander_mlflow.llm_{model}_{dataset}_top10_fusion_dense_sparse_rrf"
        try:
            df = spark.table(table_name)
            df = normalize_columns(df)
            all_dfs.append(df)
            print(f"✅ Loaded and normalized: {table_name}")
        except AnalysisException:
            print(f"⚠️ Skipped missing table: {table_name}")
            continue

    if not all_dfs:
        print(f"❌ No data found for model {model}, skipping...")
        continue

    # === Combine all datasets ===
    combined_df = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), all_dfs)

    # === Save result ===
    output_tbl = f"wei_lab_sander_mlflow.{model}_top10_fusion"
    spark.sql(f"DROP TABLE IF EXISTS {output_tbl}")
    combined_df.write.mode("overwrite").format("delta").saveAsTable(output_tbl)

    print(f"🎉 Saved combined table for model '{model}' to: {output_tbl}")



# COMMAND ----------

# MAGIC %md
# MAGIC ### Top1/3/5 accuracy calculations

# COMMAND ----------

from pyspark.sql.functions import col, lower, trim, when, max as F_max, lit
import pandas as pd

model_list = ["claude", "o3", "gemini", "ds","ro"]

topks = [1, 3, 5]
group_keys = ["pubmed", "non-pubmed", "all"]

summary_rows = []

for model in model_list:
    table_name = f"wei_lab_sander_mlflow.{model}_top10_fusion"
    try:
        print(f"📊 Evaluating top-k accuracy for: {model}")
        df = (
            spark.table(table_name)
            .withColumn("disease_norm", lower(trim(col("disease_norm"))))
            .withColumn("rare_disease_name", lower(trim(col("rare_disease_name"))))
            .withColumn("is_match", col("disease_norm") == col("rare_disease_name"))
            .withColumn("group", when(col("dataset") == "pubmed", "pubmed").otherwise("non-pubmed"))
        )
    except:
        print(f"⚠️ Skipping missing table: {table_name}")
        continue

    for k in topks:
        filtered_df = df.filter(col("final_rank") <= k)
        hit_df = filtered_df.groupBy("patient_id", "dataset", "group") \
            .agg(F_max(when(col("is_match"), 1).otherwise(0)).alias("hit"))

        # Add 'all' group by duplicating
        all_df = hit_df.withColumn("group", lit("all"))
        union_df = hit_df.unionByName(all_df)

        for group in group_keys:
            sub = union_df.filter(col("group") == group)
            total = sub.count()
            hits = sub.filter(col("hit") == 1).count()
            acc = hits / total if total > 0 else 0.0

            summary_rows.append({
                "model": model,
                "topk": f"@{k}",
                "group": group,
                "total": total,
                "hits": hits,
                "accuracy": round(acc, 4)
            })

# === Show result ===
summary_df = pd.DataFrame(summary_rows).sort_values(by=["model", "group", "topk"])
display(summary_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Ground truth rank data for Mean Reciprocal Rank (MRR) for stage 1

# COMMAND ----------

from pyspark.sql.functions import (
    col, lower, trim, when,
    min as F_min, lit
)

model_list = ["claude", "o3", "gemini", "ds", "ro"]

output_table = "wei_lab_sander_mlflow.patient_level_ground_truth_rank"

all_models_df = []

for model in model_list:
    table_name = f"wei_lab_sander_mlflow.{model}_top10_fusion"
    print(f"📊 Processing model: {model}")

    try:
        df = (
            spark.table(table_name)
            .withColumn("disease_norm", lower(trim(col("disease_norm"))))
            .withColumn("rare_disease_name", lower(trim(col("rare_disease_name"))))
            .withColumn("is_match", col("disease_norm") == col("rare_disease_name"))
        )
    except:
        print(f"⚠️ Skipping missing table: {table_name}")
        continue

    # 1️⃣ Get the rank where the ground-truth disease appears (if any)
    gt_rank_df = (
        df.filter(col("is_match"))
          .groupBy("patient_id", "dataset")
          .agg(F_min("final_rank").alias("ground_truth_disease_rank"))
    )

    # 2️⃣ Patient-level baseline (ensure patients without a hit are kept)
    base_df = (
        df.select("patient_id", "dataset")
          .distinct()
    )

    # 3️⃣ Left join + assign rank = 11 for patients with no ground-truth hit
    final_df = (
        base_df.join(gt_rank_df, ["patient_id", "dataset"], how="left")
        .withColumn(
            "ground_truth_disease_rank",
            when(col("ground_truth_disease_rank").isNull(), lit(11))
            .otherwise(col("ground_truth_disease_rank"))
        )
        .withColumn("model", lit(model))
        .select(
            "patient_id",
            "dataset",
            "model",
            "ground_truth_disease_rank"
        )
    )

    all_models_df.append(final_df)

# 4️⃣ Union results from all models
result_df = all_models_df[0]
for df in all_models_df[1:]:
    result_df = result_df.unionByName(df)

# 5️⃣ Save as a new table
(
    result_df
    .write
    .mode("overwrite")
    .saveAsTable(output_table)
)

print(f"✅ Saved patient-level ground-truth rank table to: {output_table}")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_mlflow.patient_level_ground_truth_rank 
# MAGIC order by patient_id, dataset