# Databricks notebook source
# MAGIC %md
# MAGIC ### Embedding ranking result: Notes before/after initial diagnosis_date (half year)

# COMMAND ----------

#claude
df = spark.read.option("header", True).csv("/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/structured_cases_reranked_results_claude_all.csv")
df.createOrReplaceTempView("emb_claude_view")

#ds
df = spark.read.option("header", True).csv("/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/structured_cases_reranked_results_ds_all.csv")
df.createOrReplaceTempView("emb_ds_view")

#gemini
df = spark.read.option("header", True).csv("/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/structured_cases_reranked_results_gemini_all.csv")
df.createOrReplaceTempView("emb_gemini_view")

#o3
df = spark.read.option("header", True).csv("/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/structured_cases_reranked_results_o3_all.csv")
df.createOrReplaceTempView("emb_o3_view")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Top 1 note match

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

def evaluate_top1_recall_with_details(pred_df, score_col, source_name):
    # Define valid note_type and note_title pairs
    valid_note_pairs = [
        ("Outpatient note", "Progress Notes"),
        ("Inpatient note", "Progress Notes"),
        ("Inpatient note", "Assessment & Plan Note"),
        ("Outpatient note", "Assessment & Plan Note"),
        ("Inpatient note", "Subjective & Objective"),
        ("Admission note", "H&P"),
        ("Discharge summary", "Discharge Summary"),
        ("Emergency department note", "ED Provider Notes"),
        ("Outpatient note", "Clinic Note")
    ]

    note_condition = None
    for note_type, note_title in valid_note_pairs:
        cond = (F.col("note_type_name") == note_type) & (F.col("note_title") == note_title)
        note_condition = cond if note_condition is None else note_condition | cond

    # Load ground-truth note data
    truth_df = (
        spark.table("wei_lab_sander.emb_test_records_notes")
        # .filter(F.col("note_date") >= F.col("earliest_diagnosis_date"))
        .filter(note_condition)
        .select(
            "person_id",
            "row_index",
            "disease_name",
            "note_type_name",
            "note_title",
            "earliest_diagnosis_date",
            "note_date",
            "note_text"
        )
        .withColumnRenamed("disease_name", "true_disease_name")
        .dropDuplicates(["person_id", "row_index", "true_disease_name"])
    )

    # Normalize casing and whitespace
    pred_df = pred_df.withColumn("disease", F.lower(F.trim(F.col("disease"))))
    truth_df = truth_df.withColumn(
        "true_disease_name",
        F.lower(F.trim(F.col("true_disease_name")))
    )

    # Join predictions with ground truth and label hits
    merged = (
        pred_df.join(truth_df, on=["person_id", "row_index"], how="inner")
        .withColumn(
            "is_hit",
            F.when(F.col("disease") == F.col("true_disease_name"), 1).otherwise(0)
        )
    )

    # Define ranking window: match_score → note_date → row_index
    w_score = (
        Window.partitionBy("person_id", "true_disease_name")
        .orderBy(
            F.desc(score_col),
            F.desc("note_date"),
            F.asc("row_index")
        )
    )

    # Keep only the Top-1 prediction per group
    top1 = (
        merged.withColumn("rank_by_score", F.row_number().over(w_score))
        .filter(F.col("rank_by_score") == 1)
        .withColumnRenamed("is_hit", "top1_hit")
        .withColumn("predicted_disease_name", F.col("disease"))
        .select(
            "person_id",
            "row_index",
            "true_disease_name",
            "predicted_disease_name",
            "top1_hit",
            "note_type_name",
            "note_title",
            "note_text",
            "earliest_diagnosis_date",
            "note_date",
            score_col
        )
        .withColumn("model", F.lit(source_name))
    )

    return top1


# Model name (switchable: "ds", "gemini", "o3")
llm = "gemini"   # 🔁 Change model name here
input_view = f"emb_{llm}_view"
output_table = f"wei_lab_sander.top1_details_{llm}_emb_tiebreak"

# Load reranked cosine output for the selected model
emb_input = (
    spark.table(input_view)
    .select(
        F.col("patient_id").alias("person_id"),
        "row_index",
        "disease",
        F.col("match_score")
    )
)

# ✅ Compute Top-1 recall details
top1_details = evaluate_top1_recall_with_details(
    pred_df=emb_input,
    score_col="match_score",
    source_name=f"{llm}_emb"
)

# ✅ Display and save results
display(top1_details.orderBy(F.desc("match_score")))
spark.sql(f"DROP TABLE IF EXISTS {output_table}")
top1_details.write.mode("overwrite").saveAsTable(output_table)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Score distributions

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, floor, avg, stddev, percentile_approx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ✅ Create SparkSession
spark = SparkSession.builder.getOrCreate()

# ✅ Load data
df = (
    spark.table("wei_lab_sander.top1_details_gemini_emb_tiebreak")
         .select("row_index", "top1_hit", "match_score")
)

# ✅ Create score bins (bin size = 0.1)
df = df.withColumn(
    "score_bin",
    (floor(col("match_score") * 10) / 10).cast("double")
)

# ✅ Group by top1_hit and score_bin, then count
grouped_df = df.groupBy("top1_hit", "score_bin").count()

# ✅ Summary statistics: mean, median, and standard deviation
stat_df = df.groupBy("top1_hit").agg(
    avg("match_score").alias("avg_score"),
    stddev("match_score").alias("stddev_score"),
    percentile_approx("match_score", 0.5).alias("median_score")
)

# ✅ Convert to Pandas and display statistics
stat_pdf = stat_df.toPandas().round(2)
print("\n🎯 Match Score Statistics (mean, stddev, median):")
stat_pdf.display()

# ✅ Convert grouped data to Pandas for plotting
pdf = grouped_df.toPandas()

# ✅ Pivot: index = top1_hit (0 or 1), columns = score_bin (stacked), values = count
pivot_df = (
    pdf.pivot(index="top1_hit", columns="score_bin", values="count")
       .fillna(0)
       .sort_index(axis=1)   # Ensure bins are ordered from low to high
)

# ✅ Plot stacked bar chart with value labels
fig, ax = plt.subplots(figsize=(10, 6))

bottom = [0] * len(pivot_df)
colors = plt.cm.viridis(np.linspace(0, 1, pivot_df.shape[1]))

for score_bin, color in zip(pivot_df.columns, colors):
    values = pivot_df[score_bin].values
    bars = ax.bar(
        pivot_df.index.astype(str),
        values,
        bottom=bottom,
        label=f"{score_bin:.1f}–{score_bin + 0.1:.1f}",
        color=color
    )

    # Add count labels inside bars
    for bar, value in zip(bars, values):
        if value > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white"
            )

    bottom = [b + v for b, v in zip(bottom, values)]

# === Titles and formatting ===
ax.set_title("Similarity Score Distribution by Top-1 Hit vs. Miss (Gemini Embedding Model)")
ax.set_xlabel("Top-1 Prediction (0 = Miss, 1 = Hit)")
ax.set_ylabel("Number of Predictions")
ax.legend(
    title="Similarity Score Bin",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Miss (0)", "Hit (1)"])

plt.tight_layout()
plt.show()


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Step 1: Load data
df = spark.table("wei_lab_sander.top1_details_gemini_emb_tiebreak")

# Step 2: Count number of patients per disease and hit status
grouped_df = (
    df.groupBy("true_disease_name", "top1_hit")
      .agg(F.countDistinct("person_id").alias("patient_count"))
)

# Step 3: Add total patient count per disease
window_spec = Window.partitionBy("true_disease_name")
grouped_df = grouped_df.withColumn(
    "total_patient",
    F.sum("patient_count").over(window_spec)
)

# Step 4: Add percentage column
grouped_df = grouped_df.withColumn(
    "percent",
    F.round(F.col("patient_count") / F.col("total_patient") * 100, 2)
)

# Step 5: Pivot results so hit = 1 and 0 become columns
pivot_df = (
    grouped_df
    .groupBy("true_disease_name", "total_patient")
    .pivot("top1_hit", [1, 0])
    .agg(
        F.first("patient_count").alias("count"),
        F.first("percent").alias("percent")
    )
)

# Step 6: Fill null values with 0
pivot_df = pivot_df.fillna(
    0,
    subset=["1_count", "1_percent", "0_count", "0_percent"]
)

# Step 7: Format final output
final_df = pivot_df.select(
    "true_disease_name",
    "total_patient",
    F.col("1_count").alias("hit_1_count"),
    F.col("1_percent").alias("hit_1_percent"),
    F.col("0_count").alias("hit_0_count"),
    F.col("0_percent").alias("hit_0_percent"),
)

# Step 8: Display results (sorted by total patient count)
final_df.orderBy(F.desc("total_patient")).display()


import pandas as pd
import matplotlib.pyplot as plt

# === Convert Spark DataFrame to Pandas DataFrame ===
pdf = final_df.toPandas()

# === Keep only top N diseases (too many would be crowded) ===
pdf = pdf.sort_values("total_patient", ascending=False).head(20)

# === Append (patient = xxx) label to disease name ===
pdf["label_with_count"] = pdf.apply(
    lambda row: f"{row['true_disease_name']} \n(patient = {row['total_patient']})",
    axis=1
)

# === Set new index: disease name + patient count ===
stack_df = pdf.set_index("label_with_count")[["hit_1_percent", "hit_0_percent"]]

# === Plot stacked bar chart ===
ax = stack_df.plot(
    kind="bar",
    stacked=True,
    figsize=(14, 6),
    color=["#4CAF50", "#F44336"],
    edgecolor="black"
)

# === Add percentage labels ===
for i, (idx, row) in enumerate(stack_df.iterrows()):
    # Hit block
    hit1 = row["hit_1_percent"]
    if hit1 > 0:
        ax.text(
            i,
            hit1 / 2,
            f"{hit1:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

    # Miss block (on top of hit block)
    hit0 = row["hit_0_percent"]
    if hit0 > 0:
        ax.text(
            i,
            hit1 + hit0 / 2,
            f"{hit0:.1f}%",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold"
        )

# === Title and formatting ===
plt.ylabel("Percentage (%)")
plt.xlabel("True Disease Name")
plt.title("Top-1 Prediction Accuracy per Disease (Gemini)")
plt.xticks(rotation=45, ha="right")
plt.legend(["Top-1 Hit", "Top-1 Miss"], loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

# === Show plot ===
plt.show()


# COMMAND ----------

# MAGIC %sql
# MAGIC select sum(total_pair) from ( 
# MAGIC   --833 without llm kb ,claude:857/850, gemini:864/832, o3:847/844, ds:856/814. total:984 -- last token pooling
# MAGIC select note_type_name, note_title, count(*) as total_pair,count(person_id,true_disease_name) from 
# MAGIC wei_lab_sander.top1_details_claude_emb_tiebreak
# MAGIC where true_disease_name = predicted_disease_name and top1_hit='1'
# MAGIC group by note_type_name, note_title) 
# MAGIC --llama-source:700~0.71 ,llama-gemini:856, llama-claude:854,llama-o3:848, llama-ds:854; similar since llama use avg pooling

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quality indicatror

# COMMAND ----------

from pyspark.sql import functions as F

# -------------------------------------------
# Name mapping
# -------------------------------------------
llm_map = {
    "ds": "DeepSeek R1",
    "claude": "Claude-Sonnet-4",
    "gemini": "Gemini-2.5 Pro",
    "o3": "OpenAI o3"
}

llms = list(llm_map.keys())

results = []

for llm in llms:
    table_llama = f"wei_lab_sander.top1_details_{llm}_emb_tiebreak_llama"
    table_qwen  = f"wei_lab_sander.top1_details_{llm}_emb_tiebreak"

    query = f"""
    WITH llama AS (
      SELECT person_id, row_index, match_score
      FROM {table_llama}
      WHERE true_disease_name = predicted_disease_name AND top1_hit = '1'
    ),
    qwen AS (
      SELECT person_id, row_index, match_score
      FROM {table_qwen}
      WHERE true_disease_name = predicted_disease_name AND top1_hit = '1'
    ),
    overlap AS (
      SELECT 
        l.person_id, 
        l.row_index,
        l.match_score AS llama_score,
        q.match_score AS qwen_score
      FROM llama l
      INNER JOIN qwen q 
        ON l.person_id = q.person_id AND l.row_index = q.row_index
    )
    SELECT 
      '{llm_map[llm]}' AS base_llm,
      COUNT(*) AS overlap_count,
      ROUND(AVG(llama_score), 2) AS llama_avg,
      ROUND(STDDEV(llama_score), 2) AS llama_stddev,
      ROUND(PERCENTILE(llama_score, 0.5), 2) AS llama_median,
      ROUND(AVG(qwen_score), 2) AS qwen_avg,
      ROUND(STDDEV(qwen_score), 2) AS qwen_stddev,
      ROUND(PERCENTILE(qwen_score, 0.5), 2) AS qwen_median
    FROM overlap
    """

    df = spark.sql(query)
    results.append(df)

# Combine results
final_df = results[0]
for df in results[1:]:
    final_df = final_df.unionByName(df)

# Display comparison table
final_df.orderBy("overlap_count", ascending=False).display()

####
import pandas as pd
import matplotlib.pyplot as plt

# Convert Spark DF → Pandas DF
pdf = final_df.toPandas()

# Plotting variables
models = pdf["base_llm"]
llama_avg = pdf["llama_avg"]
llama_std = pdf["llama_stddev"]
qwen_avg = pdf["qwen_avg"]
qwen_std = pdf["qwen_stddev"]
overlap_count = pdf["overlap_count"]

x = range(len(models))
width = 0.35

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# LLaMA bar
bars1 = ax.bar([i - width/2 for i in x], llama_avg, width, yerr=llama_std, 
               label='Llama', color='blue', capsize=5)

# Qwen bar
bars2 = ax.bar([i + width/2 for i in x], qwen_avg, width, yerr=qwen_std, 
               label='Qwen', color='orange', capsize=5)

# X-axis
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylabel("Cosine Similarity")
ax.set_title("Score Comparison (Overlap Predictions): Llama vs Qwen")
ax.legend()

# Add overlap counts
for i, count in enumerate(overlap_count):
    ax.text(i, max(llama_avg[i], qwen_avg[i]) + 0.02, f'n={count}', 
            ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()
