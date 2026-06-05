# Databricks notebook source
# MAGIC %md
# MAGIC ### BM25 main codes

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col, split, array_contains, lower

# ✅ Interested semantic types
target_types = [
    "Disease or Syndrome", "Gene or Genome", "Therapeutic or Preventive Procedure",
    "Pathologic Function", "Diagnostic Procedure", "Sign or Symptom", "Neoplastic Process",
    "Organic Chemical", "Laboratory Procedure", "Congenital Abnormality",
    "Mental or Behavioral Dysfunction", "Pharmacologic Substance", "Genetic Function",
    "Amino Acid, Peptide, or Protein", "Anatomical Abnormality", "Laboratory or Test Result",
    "Clinical Drug", "Enzyme", "Cell or Molecular Dysfunction", "Finding"
]

models = ["claude", "o3", "ro", "ds", "gemini"]

# === BM25 hyperparameters ===
k1 = 1.5
b = 0.75

for model in models:
    print(f"🚀 Processing model: {model}")

    # ✅ Load and filter
    df = spark.table(f"wei_lab_sander.llm_{model}_umls_matched_results") \
        .filter(col("similarity") >= 0.8) \
        .withColumn("semtypes", split(col("semantic_type_name"), "\\|"))

    # ✅ Filter rows with matching semantic type
    filtered = None
    for t in target_types:
        temp = df.filter(array_contains(col("semtypes"), t))
        filtered = temp if filtered is None else filtered.unionByName(temp)

    # ✅ Exclude negated concepts
    filtered = filtered.filter(
        ~(
            lower(col("UMLS_name")).like("not %") |
            lower(col("UMLS_name")).like("no %") |
            lower(col("UMLS_name")).like("without %")
        )
    )

    # ✅ Keep relevant columns
    filtered = filtered.select("disease", "chunk_index", "UMLS_name", "CUI", "semantic_type_name")

    # ✅ Calculate document length (CUI count per disease)
    df_doclen = filtered.groupBy("disease").agg(F.count("*").alias("doc_len"))
    avg_doc_len = df_doclen.select(F.avg("doc_len")).first()[0]

    # ✅ TF
    df_tf = filtered.groupBy("disease", "CUI").agg(F.count("*").alias("tf"))

    # ✅ DF
    df_df = filtered.select("disease", "CUI").distinct() \
        .groupBy("CUI").agg(F.countDistinct("disease").alias("df"))

    # ✅ Total diseases
    total_diseases = filtered.select("disease").distinct().count()

    # ✅ Merge + BM25
    df_bm25 = df_tf.join(df_df, on="CUI", how="inner") \
        .join(df_doclen, on="disease", how="left") \
        .withColumn("idf", F.log((F.lit(total_diseases) + F.lit(1)) / (F.col("df") + F.lit(0.5)))) \
        .withColumn(
            "bm25",
            F.col("idf") * (
                (F.col("tf") * (k1 + 1)) / (
                    F.col("tf") + k1 * (1 - b + b * (F.col("doc_len") / avg_doc_len))
                )
            )
        )

    # ✅ Add back UMLS name
    df_umls_name = filtered.select("CUI", "UMLS_name", "semantic_type_name").distinct()
    df_bm25 = df_bm25.join(df_umls_name, on="CUI", how="left")

    # ✅ Save
    df_bm25.select(
        "disease", "semantic_type_name", "CUI", "UMLS_name", "tf", "df", "idf", "bm25"
    ).write.mode("overwrite").saveAsTable(
        f"wei_lab_sander_mlflow.llm_{model}_filtered_bm25_results_02"
    )

    print(f"✅ Finished: llm_{model}_filtered_bm25_results_02 (with b={b})")


# COMMAND ----------

# MAGIC %md
# MAGIC ### TOP-10 diseases for different LLMs based on BM25 reranking

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, split, explode, trim, lower, row_number
from pyspark.sql.window import Window


spark = SparkSession.builder.getOrCreate()

# ✅ model and dataset lists
model_list = ["o3", "gemini", "claude", "ds", "ro"]
dataset_list = ["pubmed", "hms", "mme", "mygene2", "lirical", "ramedis"]


# ✅ read rare disease categories
category_df = (
    spark.table("wei_lab_sander.rare_disease_cui_category_final")
         .withColumn("rare_disease_name_lower", lower(col("rare_disease_name")))
         .select("rare_disease_name_lower", "num_true_classifications", "final_classification_string")
)

for dataset in dataset_list:
    for model in model_list:
        print(f"🔁 Processing dataset: {dataset}, model: {model}")

        # === Step 1: Load source tables ===
        if dataset == "pubmed":
            mygene_df = spark.table(f"wei_lab_sander_umls_mapping.{dataset}_dataset_update")
        else:
            mygene_df = spark.table(f"wei_lab_sander_umls_mapping.{dataset}_dataset")

        # ✅ Read BM25
        bm25_df = spark.table(f"wei_lab_sander_mlflow.llm_{model}_filtered_bm25_results_02")

        # === Step 2: Explode phenotype CUIs
        exploded = (
            mygene_df
            .withColumn("cui", explode(split(col("phenotype_umls_cuis"), ",")))
            .withColumn("cui", trim(col("cui")))
            .withColumn("query_disease", lower(col("rare_disease_name")))
        )

        bm25_df = bm25_df.withColumn("disease_norm", lower(col("disease")))

        # === Step 3: Join on CUI (cross-disease matching)
        joined = (
            exploded.alias("q")
            .join(
                bm25_df.alias("t"),
                on="cui",
                how="inner"
            )
        )

        # === Step 4: Aggregate per patient × matched_disease
        agg_df = (
            joined.groupBy("q.patient_id", "q.rare_disease_name", "t.disease")
                  .agg(
                      F.countDistinct("cui").alias("shared_cui_count"),
                      F.sum("bm25").alias("bm25_sum")
                  )
        )

        # === Step 5: Get top 10 matches by BM25 sum
        w = Window.partitionBy("patient_id").orderBy(F.desc("bm25_sum"))
        top10 = agg_df.withColumn("rank", row_number().over(w)).filter(col("rank") <= 10)

        # === Step 5.5: Add classification columns
        top10_with_class = (
            top10.withColumn("rare_disease_name_lower", lower(col("rare_disease_name")))
                 .join(category_df, on="rare_disease_name_lower", how="left")
                 .drop("rare_disease_name_lower")
        )

        # ✅ Step 6: Save results with _bm25 suffix
        output_table = f"wei_lab_sander_mlflow.llm_{model}_{dataset}_top10_similar_diseases_bm25_02"

        spark.sql(f"DROP TABLE IF EXISTS {output_table}")

        top10_with_class.write.mode("overwrite").saveAsTable(output_table)

        print(f"✅ Finished: {output_table}")

print("🏁 All done!")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Combine all datasets for each LLM

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

# Start Spark session
spark = SparkSession.builder.getOrCreate()

# Define models and datasets
model_list = ["claude", "o3", "gemini", "ds","ro"]
dataset_list = ["pubmed", "hms", "mme", "mygene2", "lirical", "ramedis"]

# Loop through each model
for model in model_list:
    combined_df = None

    for dataset in dataset_list:
        table_name = f"wei_lab_sander_mlflow.llm_{model}_{dataset}_top10_similar_diseases_bm25_02"

        try:
            df = (
                spark.table(table_name)
                .select("patient_id", "rare_disease_name", "disease", "rank","shared_cui_count","bm25_sum")
                .withColumn("dataset", lit(dataset))
            )

            if combined_df is None:
                combined_df = df
            else:
                combined_df = combined_df.unionByName(df)

        except Exception as e:
            print(f"❌ Skipped {table_name}: {e}")

    if combined_df is not None:
        output_table = f"wei_lab_sander_mlflow.llm_{model}_all_datasets_top10_bm25_02"
        

        spark.sql(f"DROP TABLE IF EXISTS {output_table}")
        combined_df.write \
        .mode("overwrite") \
        .option("overwriteSchema", "true") \
        .saveAsTable(output_table)

    else:
        print(f"⚠️ No data found for model: {model}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Recall calcuations for all models and datasets

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, max as spark_max, when
import builtins
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === Step 1: Spark session ===
spark = SparkSession.builder.getOrCreate()

# === Step 2: LLM list ===
llm_list = ["claude", "gemini", "o3", "ds","ro"] 

# === Name mapping ===
label_map = {
    "claude": "Claude-Sonnet-4",
    "gemini": "Gemini-2.5 Pro",
    "o3": "OpenAI o3",
    "ds": "DeepSeek R1",
    "ro": "Rarediseases.org"
}

# === Fixed denominators ===
TOTAL_ALL = 9290
TOTAL_PUBMED = 8601
TOTAL_NONPUB = 689

# === Step 3: Compute top-k recall with FIXED TOTALS ===
def compute_topk_for_llm(llm_name):
    try:
        table_name = f"wei_lab_sander_mlflow.llm_{llm_name}_all_datasets_top10_bm25_02"
        print(f"\n🔍 Processing: {table_name}")

        df = spark.table(table_name)

        df = df.withColumn("disease", lower(col("disease")))
        df = df.withColumn("rare_disease_name", lower(col("rare_disease_name")))
        df = df.withColumn("is_match", (col("disease") == col("rare_disease_name")).cast("int"))

        df = df.withColumn("data_group",
                           when(col("dataset") == "pubmed", "pubmed").otherwise("non_pubmed"))

        results = {"llm_model": llm_name}

        for group in ["all", "pubmed", "non_pubmed"]:

            df_group = df if group == "all" else df.filter(col("data_group") == group)

            # choose denominator
            if group == "all":
                denominator = TOTAL_ALL
            elif group == "pubmed":
                denominator = TOTAL_PUBMED
            else:
                denominator = TOTAL_NONPUB

            for k in [1, 3, 5]:

                df_topk = (
                    df_group.filter(col("rank") <= k)
                            .groupBy("patient_id", "dataset")
                            .agg(spark_max("is_match").alias(f"hit@{k}"))
                )

                hits = df_topk.filter(col(f"hit@{k}") == 1).count()
                recall = float(builtins.round(hits / denominator, 4))

                results[f"{group}_hits@{k}"] = hits
                results[f"{group}_hit_rate@{k}"] = recall

                print(f"📊 [{llm_name}] {group.upper()} @Top-{k}: hits={hits}, recall={recall} (denominator={denominator})")

        return results

    except Exception as e:
        print(f"❌ Error processing {llm_name}: {e}")
        return {"llm_model": llm_name}


# === Step 4: Run for all LLMs ===
all_results = [compute_topk_for_llm(llm) for llm in llm_list]

# === Step 5: Convert results to Spark DataFrame ===
summary_df = spark.createDataFrame(all_results)
summary_df.createOrReplaceTempView("llm_hit_rate_summary_temp")

# === Step 6: Safely convert to Pandas ===
summary_spark_df = spark.sql("SELECT * FROM llm_hit_rate_summary_temp ORDER BY llm_model")
summary_pdf = summary_spark_df.toPandas()

models = summary_pdf["llm_model"].tolist()
mapped_models = [label_map[m] for m in models]

# === Step 7: Plot setup ===
topks = ["hit_rate@1", "hit_rate@3", "hit_rate@5"]
x = np.arange(len(topks))
bar_width = 0.15

# === Uniform number labels ===
def annotate_numbers(ax, x_positions, y_values, bar_width):
    for j in range(len(x_positions)):
        for i, model_vals in enumerate(y_values):
            val = model_vals[j]
            x_pos = x_positions[j] + i * bar_width
            ax.text(
                x_pos, val + 0.015,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9, color="black"
            )

# ============================================================
# === Chart 1: All Patients ===
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
y_values = []

for i, model in enumerate(models):
    y = [summary_pdf.loc[i, f"all_{k}"] for k in topks]
    y_values.append(y)
    ax.bar(x + i * bar_width, y, width=bar_width, label=label_map[model])

annotate_numbers(ax, x, y_values, bar_width)

ax.set_xticks(x + bar_width * (len(models)-1) / 2)
ax.set_xticklabels(["Top-1", "Top-3", "Top-5"])
ax.set_ylim(0.2, 0.9)
ax.set_ylabel("Recall")
ax.set_title("Top-K Recall Across Models (All Patients)")

ax.legend(title="Model", loc="upper left")  

ax.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ============================================================
# === Chart 2: PubMed Patients ===
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
y_values = []

for i, model in enumerate(models):
    y = [summary_pdf.loc[i, f"pubmed_{k}"] for k in topks]
    y_values.append(y)
    ax.bar(x + i * bar_width, y, width=bar_width, label=label_map[model])

annotate_numbers(ax, x, y_values, bar_width)

ax.set_xticks(x + bar_width * (len(models)-1) / 2)
ax.set_xticklabels(["Top-1", "Top-3", "Top-5"])
ax.set_ylim(0.2, 0.9)
ax.set_ylabel("Recall")
ax.set_title("Top-K Recall Across Models (PubMed Patients)")

ax.legend(title="Model", loc="upper left")

ax.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# ============================================================
# === Chart 3: Non-PubMed Patients ===
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))
y_values = []

for i, model in enumerate(models):
    y = [summary_pdf.loc[i, f"non_pubmed_{k}"] for k in topks]
    y_values.append(y)
    ax.bar(x + i * bar_width, y, width=bar_width, label=label_map[model])

annotate_numbers(ax, x, y_values, bar_width)

ax.set_xticks(x + bar_width * (len(models)-1) / 2)
ax.set_xticklabels(["Top-1", "Top-3", "Top-5"])
ax.set_ylim(0.0, 1)
ax.set_ylabel("Recall")
ax.set_title("Top-K Recall Across Models (Non-PubMed Patients)")

ax.legend(title="Model", loc="upper left") 

ax.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
