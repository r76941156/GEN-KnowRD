# Databricks notebook source
pip install openpyxl upsetplot

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reranking performance comparision with(out) knowledge boost

# COMMAND ----------

import pandas as pd
# Read the Excel file into a Pandas DataFrame

comp_df = pd.read_excel(
    "/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/rerank_summary_table_kb_vs_nokb_all_models.xlsx")

                                                                          
# Convert the Pandas DataFrame to a Spark DataFrame
spark_df = spark.createDataFrame(comp_df)

# Create or replace a temporary view
spark_df.createOrReplaceTempView("comp_view")



# COMMAND ----------

# ============================================================
# Imports
# ============================================================
from pyspark.sql.functions import col, count, when, countDistinct
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import from_memberships, UpSet

# ============================================================
# Model name mapping (for plots / tables)
# ============================================================
name_map = {
    "claude": "Qwen/Claude_KB (Fusion)",
    "o3": "Qwen/O3_KB (Fusion)",
    "gemini": "Qwen/Gemini_KB (Fusion)",
    "ds": "Qwen/DS_KB (Fusion)",
    "ro": "Qwen/RO_KB (Fusion)"
}

models = ["claude", "o3", "gemini", "ds", "ro"]

# ============================================================
# Load base table (LONG FORMAT)
# ============================================================
base_df = (
    spark.table("comp_view")
    .filter(col("outcome_type").isin("🟢 Top-1", "🟡 Improved", "🔴 Worse", "🟡 Same"))
)

# ============================================================
# PART 1 — Summary table (groupBy model)  
# ============================================================
summary_df = (
    base_df
    .groupBy("model")
    .agg(
        count(when(col("kb_vs_base") == "better", True)).alias("better"),
        count(when(col("kb_vs_base") == "worse", True)).alias("worse"),
        count(when(col("kb_vs_base") == "same", True)).alias("same")
    )
    .withColumn("total", col("better") + col("worse") + col("same"))
    .withColumn("better_pct", col("better") / col("total") * 100)
    .withColumn("worse_pct", col("worse") / col("total") * 100)
    .withColumn("same_pct", col("same") / col("total") * 100)
)

summary_df = summary_df.replace(name_map, subset=["model"])
summary_df.orderBy("model").display()


# ============================================================
# PART 1b — Bar chart
#   ✔ Better / Worse / Same → Top1 / NonTop1
#   ✔ White text INSIDE bars
# ============================================================

# -------- base counts --------
base_counts = (
    base_df
    .groupBy("model")
    .agg(count("*").alias("total"))
)

# -------- split by status + Top1 --------
split_df = (
    base_df
    .groupBy("model")
    .agg(
        # BETTER
        count(
            when((col("kb_vs_base") == "better") & (col("kb_rank") == 1), True)
        ).alias("better_top1"),
        count(
            when((col("kb_vs_base") == "better") & (col("kb_rank") != 1), True)
        ).alias("better_nontop1"),

        # WORSE
        count(
            when((col("kb_vs_base") == "worse") & (col("base_rank_nokb") == 1), True)
        ).alias("worse_top1"),
        count(
            when((col("kb_vs_base") == "worse") & (col("base_rank_nokb") != 1), True)
        ).alias("worse_nontop1"),

        # SAME
        count(
            when(
                (col("kb_vs_base") == "same") &
                (col("kb_rank") == 1) &
                (col("base_rank_nokb") == 1),
                True
            )
        ).alias("same_top1"),
        count(
            when(
                (col("kb_vs_base") == "same") &
                ~((col("kb_rank") == 1) & (col("base_rank_nokb") == 1)),
                True
            )
        ).alias("same_nontop1"),
    )
)

plot_df = (
    split_df
    .join(base_counts, on="model")
    .withColumn("better_top1_pct", col("better_top1") / col("total") * 100)
    .withColumn("better_nontop1_pct", col("better_nontop1") / col("total") * 100)
    .withColumn("worse_top1_pct", col("worse_top1") / col("total") * 100)
    .withColumn("worse_nontop1_pct", col("worse_nontop1") / col("total") * 100)
    .withColumn("same_top1_pct", col("same_top1") / col("total") * 100)
    .withColumn("same_nontop1_pct", col("same_nontop1") / col("total") * 100)
    .replace(name_map, subset=["model"])
    .orderBy("model")
)

pdf = plot_df.toPandas()

# ============================================================
# Plot
# ============================================================
models_plot = pdf["model"].tolist()
x = list(range(len(models_plot)))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6))

# ---- BETTER ----
b1a = ax.bar(
    [i - width for i in x],
    pdf["better_top1_pct"],
    width,
    label="Better (Top-1)",
    edgecolor="black"
)
b1b = ax.bar(
    [i - width for i in x],
    pdf["better_nontop1_pct"],
    width,
    bottom=pdf["better_top1_pct"],
    label="Better (Non-Top1)",
    edgecolor="black"
)

# ---- WORSE ----
b2a = ax.bar(
    x,
    pdf["worse_top1_pct"],
    width,
    label="Worse (Top-1)",
    edgecolor="black"
)
b2b = ax.bar(
    x,
    pdf["worse_nontop1_pct"],
    width,
    bottom=pdf["worse_top1_pct"],
    label="Worse (Non-Top1)",
    edgecolor="black"
)

# ---- SAME ----
b3a = ax.bar(
    [i + width for i in x],
    pdf["same_top1_pct"],
    width,
    label="Same (Top-1)",
    edgecolor="black"
)
b3b = ax.bar(
    [i + width for i in x],
    pdf["same_nontop1_pct"],
    width,
    bottom=pdf["same_top1_pct"],
    label="Same (Non-Top1)",
    edgecolor="black"
)

# ---- axes ----
ax.set_title("KB Impact on Ranking Outcomes Across Models", fontsize=14)
ax.set_ylabel("Percentage (%)")
ax.set_xlabel("Model")
ax.set_xticks(x)
ax.set_xticklabels(models_plot, rotation=20)

ax.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    frameon=True
)

# ============================================================
# White text INSIDE each bar segment
# ============================================================
def label_stack(x_pos, bottom, height, value):
    if value > 0:
        ax.text(
            x_pos,
            bottom + height / 2,
            f"{value:.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="white"
        )

for i in range(len(models_plot)):
    # Better
    label_stack(i - width, 0, pdf.loc[i, "better_top1_pct"], pdf.loc[i, "better_top1_pct"])
    label_stack(
        i - width,
        pdf.loc[i, "better_top1_pct"],
        pdf.loc[i, "better_nontop1_pct"],
        pdf.loc[i, "better_nontop1_pct"]
    )

    # Worse
    label_stack(i, 0, pdf.loc[i, "worse_top1_pct"], pdf.loc[i, "worse_top1_pct"])
    label_stack(
        i,
        pdf.loc[i, "worse_top1_pct"],
        pdf.loc[i, "worse_nontop1_pct"],
        pdf.loc[i, "worse_nontop1_pct"]
    )

    # Same
    label_stack(i + width, 0, pdf.loc[i, "same_top1_pct"], pdf.loc[i, "same_top1_pct"])
    label_stack(
        i + width,
        pdf.loc[i, "same_top1_pct"],
        pdf.loc[i, "same_nontop1_pct"],
        pdf.loc[i, "same_nontop1_pct"]
    )

plt.tight_layout()
plt.show()

# ============================================================
# PART 2 — UpSet plots 
# ============================================================
pdf = (
    base_df
    .select("patient_id", "correct_disease", "model", "kb_vs_base")
    .toPandas()
)

pdf["correct_disease"] = pdf["correct_disease"].str.lower().str.strip()

def upset_by_status(status):
    membership = {}

    for m in models:
        label = name_map[m]
        subset = pdf[
            (pdf["model"] == m) &
            (pdf["kb_vs_base"] == status)
        ][["patient_id", "correct_disease"]]

        membership[label] = set(map(tuple, subset.values))

    all_pairs = set().union(*membership.values())

    memberships = [
        [model for model, s in membership.items() if pair in s]
        for pair in all_pairs
    ]

    upset_data = from_memberships(memberships)
    UpSet(
        upset_data,
        subset_size="count",
        show_counts=True
    ).plot()

    plt.suptitle(
        f"KB Impact ({status.upper()}) Across Models",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

upset_by_status("better")
upset_by_status("worse")
upset_by_status("same")

# ============================================================
# PART 3 — Diseases where ALL 4 models are worse 
# ============================================================
df_all_worse = (
    base_df
    .filter(col("kb_vs_base") == "worse")
    .groupBy("patient_id", "correct_disease")
    .agg(countDistinct("model").alias("num_models"))
    .filter(col("num_models") == 4)   # claude + o3 + gemini + ds
)

(
    df_all_worse
    .groupBy("correct_disease")
    .count()
    .withColumnRenamed("count", "patient_count")
    .orderBy(col("patient_count").desc())
    .display()
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Better/Worse/Same summary charts

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH base AS (
# MAGIC   SELECT
# MAGIC     model,
# MAGIC     base_rank_nokb,
# MAGIC     kb_vs_base,
# MAGIC     top1_transition,
# MAGIC
# MAGIC     -- 🔑 three cohorts
# MAGIC     CASE
# MAGIC       WHEN LOWER(dataset) = 'pubmed' THEN 'pubmed'
# MAGIC       ELSE 'non_pubmed'
# MAGIC     END AS dataset_group
# MAGIC   FROM comp_view
# MAGIC ),
# MAGIC
# MAGIC -- =========================
# MAGIC -- expand to ALL / PubMed / Non-PubMed
# MAGIC -- =========================
# MAGIC expanded AS (
# MAGIC   SELECT 'all' AS cohort, * FROM base
# MAGIC   UNION ALL
# MAGIC   SELECT 'pubmed' AS cohort, * FROM base WHERE dataset_group = 'pubmed'
# MAGIC   UNION ALL
# MAGIC   SELECT 'non_pubmed' AS cohort, * FROM base WHERE dataset_group = 'non_pubmed'
# MAGIC ),
# MAGIC
# MAGIC -- =========================
# MAGIC -- Top-1 population
# MAGIC -- =========================
# MAGIC top1 AS (
# MAGIC   SELECT
# MAGIC     cohort,
# MAGIC     model,
# MAGIC     COUNT(*) AS top1_total,
# MAGIC     SUM(CASE WHEN top1_transition = 'stable_top1' THEN 1 ELSE 0 END) AS top1_same,
# MAGIC     SUM(CASE WHEN top1_transition = 'lost_top1'   THEN 1 ELSE 0 END) AS top1_worse
# MAGIC   FROM expanded
# MAGIC   WHERE base_rank_nokb = 1
# MAGIC   GROUP BY cohort, model
# MAGIC ),
# MAGIC
# MAGIC -- =========================
# MAGIC -- Non-Top-1 population
# MAGIC -- =========================
# MAGIC non_top1 AS (
# MAGIC   SELECT
# MAGIC     cohort,
# MAGIC     model,
# MAGIC     COUNT(*) AS non_top1_total,
# MAGIC     SUM(CASE WHEN kb_vs_base = 'better' THEN 1 ELSE 0 END) AS non_top1_better,
# MAGIC     SUM(CASE WHEN kb_vs_base = 'worse'  THEN 1 ELSE 0 END) AS non_top1_worse,
# MAGIC     SUM(CASE WHEN kb_vs_base = 'same'   THEN 1 ELSE 0 END) AS non_top1_same
# MAGIC   FROM expanded
# MAGIC   WHERE base_rank_nokb <> 1
# MAGIC   GROUP BY cohort, model
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC   t.cohort,
# MAGIC   t.model,
# MAGIC
# MAGIC   -- Top-1 stability
# MAGIC   ROUND(t.top1_same  / t.top1_total * 100, 1) AS top1_same_pct,
# MAGIC   ROUND(t.top1_worse / t.top1_total * 100, 1) AS top1_worse_pct,
# MAGIC
# MAGIC   -- Non-Top-1 direction
# MAGIC   ROUND(n.non_top1_better / n.non_top1_total * 100, 1) AS non_top1_better_pct,
# MAGIC   ROUND(n.non_top1_worse  / n.non_top1_total * 100, 1) AS non_top1_worse_pct,
# MAGIC   ROUND(n.non_top1_same   / n.non_top1_total * 100, 1) AS non_top1_same_pct
# MAGIC
# MAGIC FROM top1 t
# MAGIC JOIN non_top1 n
# MAGIC   ON t.cohort = n.cohort
# MAGIC  AND t.model  = n.model
# MAGIC ORDER BY cohort, model;
# MAGIC

# COMMAND ----------

# =====================================================
# A. Spark SQL: Non-Top1 → Top-1 promotion summary
# =====================================================
promotion_sql = """
WITH base AS (
  SELECT
    model,
    CASE
      WHEN LOWER(dataset) = 'pubmed' THEN 'PUBMED'
      ELSE 'NON_PUBMED'
    END AS dataset_group,
    base_rank_nokb,
    kb_rank
  FROM comp_view
),

expanded AS (
  SELECT 'ALL' AS cohort, * FROM base
  UNION ALL
  SELECT 'PUBMED' AS cohort, * FROM base WHERE dataset_group = 'PUBMED'
  UNION ALL
  SELECT 'NON_PUBMED' AS cohort, * FROM base WHERE dataset_group = 'NON_PUBMED'
)

SELECT
  cohort,
  model,
  COUNT(*) AS non_top1_total,
  SUM(CASE WHEN kb_rank = 1 THEN 1 ELSE 0 END) AS non_top1_rank1_count,
  ROUND(
    SUM(CASE WHEN kb_rank = 1 THEN 1 ELSE 0 END) / COUNT(*) * 100,
    1
  ) AS non_top1_rank1_pct
FROM expanded
WHERE base_rank_nokb <> 1
GROUP BY cohort, model
ORDER BY cohort, model
"""

promotion_sdf = spark.sql(promotion_sql)

promotion_sdf.write.mode("overwrite").saveAsTable(
    "wei_lab_sander.non_top1_promotion_summary"
)


# COMMAND ----------

# =====================================================
# B. Build results_chart2 (Pandas)
# =====================================================
import pandas as pd

summary_df = spark.table(
    "wei_lab_sander.non_top1_promotion_summary"
).toPandas()

# ---- model name mappings ----
name_map = {
    "claude": "Qwen/Claude_KB (Fusion)",
    "ds": "Qwen/DS_KB (Fusion)",
    "gemini": "Qwen/Gemini_KB (Fusion)",
    "o3": "Qwen/O3_KB (Fusion)",
    "ro": "Qwen/RO_KB (Fusion)"
}

summary_df["model"] = summary_df["model"].map(name_map)

# ---- results_chart2 ----
results_chart2 = {
    cohort: summary_df.loc[
        summary_df["cohort"] == cohort,
        ["model", "non_top1_rank1_pct"]
    ].reset_index(drop=True)
    for cohort in ["ALL", "PUBMED", "NON_PUBMED"]
}

# COMMAND ----------

# =====================================================
# C. Plot: Non-Top1 → Top-1 Promotion
# =====================================================
import numpy as np
import matplotlib.pyplot as plt

def plot_non_top1_to_top1_promotion(results_chart2):

    models = [
        "Qwen/Claude_KB (Fusion)",
        "Qwen/DS_KB (Fusion)",
        "Qwen/Gemini_KB (Fusion)",
        "Qwen/O3_KB (Fusion)",
        "Qwen/RO_KB (Fusion)"
    ]

    cohorts_display = ["ALL", "PUBMED", "NON-PUBMED"]
    cohorts_lookup  = ["ALL", "PUBMED", "NON_PUBMED"]

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2","#CCB974"]

    # ---- build value matrix ----
    vals = {model: [] for model in models}

    for cohort in cohorts_lookup:
        df = (
            results_chart2[cohort]
            .set_index("model")
            .reindex(models)
        )
        for model in models:
            vals[model].append(df.loc[model, "non_top1_rank1_pct"])

    x = np.arange(len(cohorts_display))
    width = 0.18

    plt.figure(figsize=(12, 6))

    for i, model in enumerate(models):
        plt.bar(
            x + i * width,
            vals[model],
            width,
            label=model,
            color=colors[i],
            edgecolor="black"
        )

        for j, v in enumerate(vals[model]):
            plt.text(
                x[j] + i * width,
                v + 0.6,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.xticks(x + width * 1.5, cohorts_display, fontsize=11)
    plt.ylabel("Non-Top-1 → Top-1 Promotion (%)", fontsize=12)

    plt.title(
        "Promotion of Non-Top-1 Predictions to Top-1 After KB Reranking",
        fontsize=14,
        fontweight="bold"
    )

    plt.ylim(10, 70)   # 根据你的数据可微调
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()
    plt.show()


# ---- call ----
plot_non_top1_to_top1_promotion(results_chart2)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Disease complexity for stage 1 (BM25 / Qwen embedding - kB comparisons)

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.getActiveSession()


model_list = ["claude", "o3", "ds", "gemini","ro"]
dataset_list = ["pubmed", "hms", "mme", "mygene2", "lirical", "ramedis"]
topx_list = [1, 3, 5]

for model in model_list:
    for dataset in dataset_list:
       
        input_table = f"wei_lab_sander_mlflow.llm_{model}_{dataset}_top10_fusion_dense_sparse_rrf"
        try:
            base_df = spark.table(input_table)

            for topx in topx_list:
                print(f"📝 Processing model: {model}, dataset: {dataset}, Top{topx}")

                df_out = (
                    base_df
                    .filter(col("final_rank") <= topx)
                    .select(
                        "dataset",
                        "patient_id",
                        "rare_disease_name",
                        "disease_norm",
                        "num_true_classifications",
                        "final_classification_string",
                        "final_rank"
                    )
                    .withColumnRenamed("disease_norm", "disease")
                    .withColumnRenamed("final_rank", "rank")
                    .orderBy("dataset","patient_id", "rare_disease_name", "final_rank")
                )

                out_table = f"wei_lab_sander_mlflow.llm_{model}_{dataset}_top{topx}_similar_diseases"
               
                spark.sql(f"DROP TABLE IF EXISTS {out_table}")


                df_out.write.mode("overwrite").saveAsTable(out_table)

                print(f"✅ Saved to {out_table}")

        except Exception as e:
            print(f"❌ Skipping model: {model}, dataset: {dataset} due to error: {str(e)}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Recall by diff LLMs and datasets

# COMMAND ----------

from pyspark.sql.functions import lower, col, round as spark_round
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

model_list = ["claude", "o3", "ds", "gemini","ro"]
dataset_list = ["pubmed", "hms", "mme", "mygene2", "lirical", "ramedis"]
topx_list = [1, 3, 5, 10]

for topx in topx_list:
    all_results = []

    for dataset in dataset_list:
       
        if dataset == "pubmed":
            tbl = f"wei_lab_sander_umls_mapping.{dataset}_dataset_update"
        else:
            tbl = f"wei_lab_sander_umls_mapping.{dataset}_dataset"

        total_patients = spark.table(tbl).select("patient_id").distinct().count()
        print(f"🧮 {dataset}: Total Patients = {total_patients}")

        for model in model_list:
            print(f"🔍 Processing {dataset} with {model} @Top{topx}")
            try:
                df = spark.table(f"wei_lab_sander_mlflow.llm_{model}_{dataset}_top{topx}_similar_diseases")
                
                match_df = df.filter(lower(col("rare_disease_name")) == lower(col("disease")))
                matched_count = match_df.select("patient_id").distinct().count()

                print(f"✅ {dataset}-{model}: matched {matched_count} / {total_patients}")
                all_results.append((dataset, model, total_patients, matched_count, topx))
            except Exception as e:
                print(f"❌ Error reading table for {model}-{dataset}-Top{topx}: {e}")
                continue

    # ✅ save result into dataframes
    if all_results:
        final_df = spark.createDataFrame(all_results, ["dataset", "model", "total_pts", "match_per"]) \
                        .withColumn("match_rate", spark_round(col("match_per") / col("total_pts"), 4))

        # ✅ Sum results into TopX tables
        output_table = f"wei_lab_sander_mlflow.llm_all_datasets_match_summary_top{topx}"
        final_df.write.mode("overwrite").saveAsTable(output_table)
        print(f"📦 Saved results to: {output_table}")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   model,
# MAGIC   --dataset,
# MAGIC   SUM(match_per) AS total_matches,
# MAGIC   SUM(total_pts) AS total_pts,
# MAGIC   round(SUM(match_per) * 1.0 / SUM(total_pts),3) AS weighted_match_rate
# MAGIC FROM wei_lab_sander_mlflow.llm_all_datasets_match_summary_top1 --1/3/5
# MAGIC GROUP BY model --,dataset
# MAGIC ORDER BY weighted_match_rate DESC;

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col, round, countDistinct, lit

spark = SparkSession.builder.getOrCreate()

# ✅ Define configs

model_list = ["ro", "claude", "o3", "ds", "gemini"]
dataset_list = ["pubmed", "hms", "mme", "mygene2", "lirical", "ramedis"]
topx_list = [1, 3, 5]

# ✅ Loop through each TopX and write a separate result table
for topx in topx_list:
    print(f"\n📊 Processing Top{topx}...\n")
    all_results = []

    for dataset in dataset_list:
        for model in model_list:
            tbl = f"wei_lab_sander_mlflow.llm_{model}_{dataset}_top{topx}_similar_diseases"
            print(f"🔍 Checking: {tbl}")

            try:
                df = spark.table(tbl)
            except:
                print(f"⚠️ Table not found: {tbl}")
                continue

            # Match on rare_disease_name == disease (case insensitive)
            matched = df.filter(
                lower(col("rare_disease_name")) == lower(col("disease"))
            )

            # Total per class
            total_per_group = df.groupBy("num_true_classifications") \
                .agg(
                    countDistinct("patient_id").alias("total_pts"),
                    countDistinct("disease").alias("total_diseases")
                )

            matched_per_group = matched.groupBy("num_true_classifications") \
                .agg(
                    countDistinct("patient_id").alias("matched_pts")
                )

            # Join and compute match rate
            result = total_per_group.join(
                matched_per_group,
                on="num_true_classifications",
                how="left"
            ).fillna(0)

            result = result.withColumn("dataset", lit(dataset)) \
                           .withColumn("model", lit(model)) \
                           .withColumn("match_rate", round(col("matched_pts") / col("total_pts"), 4))

            all_results.append(result)

    # 🔄 Combine and save for this topX
    if all_results:
        final_df = all_results[0]
        for df in all_results[1:]:
            final_df = final_df.unionByName(df)

        output_table = f"wei_lab_sander_mlflow.llm_all_datasets_match_summary_top{topx}_by_class"
        spark.sql(f"DROP TABLE IF EXISTS {output_table}")
        final_df.write.mode("overwrite").saveAsTable(output_table)
        print(f"✅ Done saving: {output_table}")
    else:
        print(f"❌ No results found for Top{topx}.")


# COMMAND ----------

# MAGIC %md
# MAGIC ### LLM accuracy vs. complexity

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.lines as mlines
import seaborn as sns
from matplotlib.transforms import ScaledTranslation

# =====================================================
# Visual style
# =====================================================
sns.set(style="whitegrid", font_scale=1.2, rc={
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# =====================================================
# Name Mapping
# =====================================================
name_map = {
    "claude": "Claude-Sonnet-4",
    "gemini": "Gemini-2.5 Pro",
    "ds": "DeepSeek R1",
    "o3": "OpenAI o3",
    "ro": "Rarediseases.org"
}

bright_palette = sns.color_palette("tab10")
#top_nums = ["1", "3", "5"]
top_nums = ["1"]
order = [str(i) for i in range(1, 9)] + ["9+"]

# =====================================================
# Main loop
# =====================================================
for topx in top_nums:
    print(f"📊 Plotting Top{topx} (bar chart)...")

    query = f"""
    SELECT
      model,
      CASE 
        WHEN num_true_classifications BETWEEN 9 AND 13 THEN '9+'
        ELSE CAST(num_true_classifications AS STRING)
      END AS classification_group,
      SUM(matched_pts) AS matched_sum,
      SUM(total_pts) AS total_sum,
      ROUND(SUM(matched_pts) / SUM(total_pts), 4) AS recall
    FROM
      wei_lab_sander_mlflow.llm_all_datasets_match_summary_top{topx}_by_class
    GROUP BY
      model,
      CASE 
        WHEN num_true_classifications BETWEEN 9 AND 13 THEN '9+'
        ELSE CAST(num_true_classifications AS STRING)
      END
    """
    df = spark.sql(query).toPandas()

    if df.empty or df["total_sum"].fillna(0).sum() == 0:
        print(f"⚠️ No data for Top{topx}, skipping.")
        continue

    # -------------------------------------------------
    # Preprocess
    # -------------------------------------------------
    df["model"] = df["model"].map(name_map)
    df["classification_group"] = pd.Categorical(
        df["classification_group"], categories=order, ordered=True
    )
    df = df.sort_values(["classification_group", "model"])

    # -------------------------------------------------
    # Overall average
    # -------------------------------------------------
    overall_avg = df["matched_sum"].sum() / df["total_sum"].sum()

    # =================================================
    # BAR CHART
    # =================================================
    fig = plt.figure(figsize=(11, 6))
    ax = sns.barplot(
        data=df,
        x="classification_group",
        y="recall",
        hue="model",
        palette=bright_palette,
        edgecolor="black"
    )

    # -------------------------------------------------
    # Overall avg line
    # -------------------------------------------------
    ax.axhline(overall_avg, color="gray", linestyle="--", linewidth=1.2)

    # -------------------------------------------------
    # n annotations (DATA-DRIVEN, SAFE)
    # -------------------------------------------------
    ann_df = (
        df.groupby("classification_group", observed=True)
          .agg(
              recall_max=("recall", "max"),
              total_sum=("total_sum", "first")
          )
          .reset_index()
    )

    xticks = ax.get_xticks()
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    x_lookup = dict(zip(xticklabels, xticks))

    for _, row in ann_df.iterrows():
        cg = row["classification_group"]
        if cg not in x_lookup:
            continue

    
        ax.annotate(
            f"n = {int(row['total_sum'])}",
            xy=(x_lookup[cg], row["recall_max"]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white",
                edgecolor="none",
                alpha=0.8
            )
        )

    # -------------------------------------------------
    # Overall avg label
    # -------------------------------------------------
    x_center = (len(order) - 1) / 2
    if topx == "1":
        x_center += 0.7

    ax.text(
        x_center,
        overall_avg,
        f"Overall avg = {overall_avg:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="gray",
        transform=ax.transData + ScaledTranslation(
            0, 6 / 72, fig.dpi_scale_trans
        ),
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.8
        )
    )

    # -------------------------------------------------
    # Dynamic Y-axis
    # -------------------------------------------------
    y_min = df["recall"].min()
    y_max = df["recall"].max()
    margin = 0.1 * (y_max - y_min if y_max > y_min else 1)
    ax.set_ylim(max(0, y_min - margin), min(1.0, y_max + margin))

    # -------------------------------------------------
    # Labels & title
    # -------------------------------------------------
    ax.set_title(
        f"LLM Recall vs Disease Complexity — Top{topx}",
        fontsize=16,
        fontweight="bold"
    )
    ax.set_xlabel("Disease complexity (# classifications)")
    ax.set_ylabel("Recall")

    # -------------------------------------------------
    # Legend
    # -------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    avg_handle = mlines.Line2D([], [], color="gray", linestyle="--", label="Overall avg")

    ax.legend(
        handles=handles + [avg_handle],
        labels=labels + ["Overall avg"],
        title="Model",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=len(handles) + 1,
        frameon=True
    )

    # -------------------------------------------------
    # Clean look
    # -------------------------------------------------
    ax.grid(False)
    plt.tight_layout()
    plt.show()
