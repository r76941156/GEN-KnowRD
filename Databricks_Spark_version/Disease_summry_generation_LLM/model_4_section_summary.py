# Databricks notebook source
# MAGIC %md
# MAGIC ### All 4 sections for each model
# MAGIC - clinical_presentation_section
# MAGIC - diagnostic_evaluation_section
# MAGIC - subtype_variant_section
# MAGIC - management_standard_therapy_section

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType
import re

# ✅ Start Spark
spark = SparkSession.builder.getOrCreate()

# === 🔧 UDF: Remove links ===
def remove_links(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\((?:\s*[\w\.-]+\.[a-z]{2,}(?:,)?\s*)+\)", "", text)
    return text

remove_links_udf = udf(remove_links, StringType())

# === 🌀 Loop through each model, load + join all sections ===
model_list = ["claude", "o3", "gemini", "ds"]

for m in model_list:
    print(f"🔄 Processing model: {m}")

    # === Step 1: Load 4 section tables
    clinical_tbl = f"wei_lab_sander_mlflow.llm_{m}_clinical_presentation_section"
    diagnostic_tbl = f"wei_lab_sander_mlflow.llm_{m}_diagnostic_evaluation_section"
    subtype_tbl = f"wei_lab_sander_mlflow.llm_{m}_subtype_variant_section"
    management_tbl = f"wei_lab_sander_mlflow.llm_{m}_management_therapy_section"

    clinical_df = spark.table(clinical_tbl).select(
        col("disease_name"),
        col("clinical_presentation_section").alias(f"{m}_clinical_presentation_section")
    )
    diagnostic_df = spark.table(diagnostic_tbl).select(
        col("disease_name"),
        col("diagnostic_evaluation_section").alias(f"{m}_diagnostic_evaluation_section")
    )
    subtype_df = spark.table(subtype_tbl).select(
        col("disease_name"),
        col("subtype_variant_section").alias(f"{m}_subtype_variant_section")
    )
    management_df = spark.table(management_tbl).select(
        col("disease_name"),
        col("management_standard_therapy_section").alias(f"{m}_management_therapy_section")
    )

    # === Step 2: Outer join 4 sections
    model_df = clinical_df \
        .join(diagnostic_df, on="disease_name", how="outer") \
        .join(subtype_df, on="disease_name", how="outer") \
        .join(management_df, on="disease_name", how="outer")

    # === Step 3: Clean markdown if model is o3
    if m == "o3":
        for sec in ["clinical_presentation", "diagnostic_evaluation", "subtype_variant", "management_therapy"]:
            colname = f"{m}_{sec}_section"
            model_df = model_df.withColumn(colname, remove_links_udf(col(colname)))

    # === Step 4: Save output
    output_table = f"wei_lab_sander_mlflow.llm_{m}_all_sections"
    spark.sql(f"DROP TABLE IF EXISTS {output_table}")
    model_df.write.mode("overwrite").saveAsTable(output_table)

    print(f"✅ Saved full section table: {output_table}")


# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from wei_lab_sander_mlflow.llm_claude_all_sections