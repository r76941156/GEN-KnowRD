# Databricks notebook source
# MAGIC %md
# MAGIC ### UMLS preferred term mapping

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ✅ Initialize Spark
spark = SparkSession.builder.getOrCreate()

# ✅ File path
file_path = "/Volumes/workspace_victrsd/wei_lab_sander/phemap/MRCONSO.RRF"

# ✅ Target terminologies
target_sabs = [
    "SNOMEDCT_US", "RXNORM", "LNC", "OMIM",
    "ICD9CM", "ICD10CM", "HPO", "ORPHANET",
    "MSH", "GO"
]

# ✅ Read the RRF file
df_raw = spark.read.option("delimiter", "|").csv(file_path)

# ✅ Select required columns
df_selected = df_raw.select(
    col("_c0").alias("CUI"),
    col("_c11").alias("SAB"),
    col("_c12").alias("TTY"),
    col("_c13").alias("CODE"),
    col("_c14").alias("CUI_Name")
)

# ✅ Filter for target terminologies and TTY = PT
df_filtered = df_selected.filter(
    (col("SAB").isin(target_sabs)) &
    (col("TTY") == "PT")
)

# ✅ Remove duplicates by (CUI, CODE, SAB)
df_distinct = df_filtered.dropDuplicates(["CUI", "CODE", "SAB"])

# ✅ Save as a Hive table
df_distinct.write.mode("overwrite").saveAsTable(
    "wei_lab_sander.umls_cui_terminology_mapping_PT"
)


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander.umls_cui_terminology_mapping_PT