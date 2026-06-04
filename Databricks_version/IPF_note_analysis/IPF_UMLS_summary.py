# Databricks notebook source
pip install openpyxl upsetplot

# COMMAND ----------

# MAGIC %md
# MAGIC ### IPF sign and symptom sections from models (Claude/Gemini/o3/DS/RO)

# COMMAND ----------

# # Read the Excel file into a Pandas DataFrame
sign_symptom_df = pd.read_excel(
    "/Volumes/workspace_victrsd/wei_lab_sander/phemap/UMLS_term_extraction_LLM_text_0728.xlsx"
)

# Convert the Pandas DataFrame to a Spark DataFrame
spark_df = spark.createDataFrame(sign_symptom_df)

# Create or replace a temporary view
spark_df.createOrReplaceTempView("IPF_sign_symptom_view")
display(spark_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### IPF Signs and symptoms UMLS summary

# COMMAND ----------

# MAGIC %sql
# MAGIC select model,semantic_type_name,count(distinct UMLS_name) as total from wei_lab_sander.llm_sign_symptom_matched_results
# MAGIC where semantic_type_name in (
# MAGIC   'Anatomical Abnormality',
# MAGIC   'Daily or Recreational Activity',
# MAGIC   'Disease or Syndrome',
# MAGIC   'Finding',
# MAGIC   'Laboratory or Test Result',
# MAGIC   'Mental or Behavioral Dysfunction',
# MAGIC   'Neoplastic Process',
# MAGIC   'Pathologic Function',
# MAGIC   'Sign or Symptom'
# MAGIC ) and similarity>= 0.9 
# MAGIC and UMLS_name not in (
# MAGIC   'Asymptomatic (finding)',
# MAGIC   'Complication',
# MAGIC   'No Complication',
# MAGIC   'No breathlessness',
# MAGIC   'Rest',
# MAGIC   'Physical findings',
# MAGIC   'Reduced',
# MAGIC   'Abnormal',
# MAGIC   'Disease',
# MAGIC   'Severe (severity modifier)',
# MAGIC   'Progressive cGVHD',
# MAGIC   'Symptoms',
# MAGIC   'Moderate',
# MAGIC   'Allergy Severity - Severe',
# MAGIC   'Death (finding)',
# MAGIC   'No Shortness of Breath',
# MAGIC   'No sputum',
# MAGIC   'Increased (finding)',
# MAGIC   'Discomfort',
# MAGIC   'Progressive Disease',
# MAGIC   'No Weight Loss',
# MAGIC   'Oral Manifestations',
# MAGIC   'No fatigue',
# MAGIC   'Reinfection',
# MAGIC   'No respiratory symptoms',
# MAGIC   'Swelling',
# MAGIC   'Exacerbation',
# MAGIC   'Intensity and Distress 5',
# MAGIC   'Disease Progression',
# MAGIC   'Disability',
# MAGIC   'Worse',
# MAGIC   'Impaired exercise tolerance',
# MAGIC   'Dry skin',
# MAGIC   'Symptom mild',
# MAGIC   'Mental disorders',
# MAGIC   'Ache',
# MAGIC   'Oral Complication',
# MAGIC   'Present',
# MAGIC   'irPD (Immune-Related Response Criteria)',
# MAGIC   'Fasting',
# MAGIC   'Infection',
# MAGIC   'Communicable Diseases',
# MAGIC   'Medical Condition',
# MAGIC   'Sexual intercourse - finding',
# MAGIC   'Exercise',
# MAGIC   'METHOD:*',
# MAGIC   'SYSTEM:*',
# MAGIC   'SCALE:*',
# MAGIC   'Household composition',
# MAGIC   'Living'
# MAGIC )
# MAGIC group by model,semantic_type_name

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table wei_lab_sander.IPF_sign_symptom_matched_results_final as
# MAGIC select * from wei_lab_sander.llm_sign_symptom_matched_results
# MAGIC where semantic_type_name in (
# MAGIC   'Anatomical Abnormality',
# MAGIC   'Daily or Recreational Activity',
# MAGIC   'Disease or Syndrome',
# MAGIC   'Finding',
# MAGIC   'Laboratory or Test Result',
# MAGIC   'Mental or Behavioral Dysfunction',
# MAGIC   'Neoplastic Process',
# MAGIC   'Pathologic Function',
# MAGIC   'Sign or Symptom'
# MAGIC )
# MAGIC and similarity >= 0.9
# MAGIC and UMLS_name not in (
# MAGIC   'Asymptomatic (finding)',
# MAGIC   'Complication',
# MAGIC   'No Complication',
# MAGIC   'No breathlessness',
# MAGIC   'Rest',
# MAGIC   'Physical findings',
# MAGIC   'Reduced',
# MAGIC   'Abnormal',
# MAGIC   'Disease',
# MAGIC   'Severe (severity modifier)',
# MAGIC   'Progressive cGVHD',
# MAGIC   'Symptoms',
# MAGIC   'Moderate',
# MAGIC   'Allergy Severity - Severe',
# MAGIC   'Death (finding)',
# MAGIC   'No Shortness of Breath',
# MAGIC   'No sputum',
# MAGIC   'Increased (finding)',
# MAGIC   'Discomfort',
# MAGIC   'Progressive Disease',
# MAGIC   'No Weight Loss',
# MAGIC   'Oral Manifestations',
# MAGIC   'No fatigue',
# MAGIC   'Reinfection',
# MAGIC   'No respiratory symptoms',
# MAGIC   'Swelling',
# MAGIC   'Exacerbation',
# MAGIC   'Intensity and Distress 5',
# MAGIC   'Disease Progression',
# MAGIC   'Disability',
# MAGIC   'Worse',
# MAGIC   'Impaired exercise tolerance',
# MAGIC   'Dry skin',
# MAGIC   'Symptom mild',
# MAGIC   'Mental disorders',
# MAGIC   'Ache',
# MAGIC   'Oral Complication',
# MAGIC   'Present',
# MAGIC   'irPD (Immune-Related Response Criteria)',
# MAGIC   'Fasting',
# MAGIC   'Infection',
# MAGIC   'Communicable Diseases',
# MAGIC   'Medical Condition',
# MAGIC   'Sexual intercourse - finding',
# MAGIC   'Exercise',
# MAGIC   'METHOD:*',
# MAGIC   'SYSTEM:*',
# MAGIC   'SCALE:*',
# MAGIC   'Household composition',
# MAGIC   'Living',
# MAGIC   'Activity (animal life circumstance)',
# MAGIC   'Cause of Death',
# MAGIC   'Collapse (finding)',
# MAGIC   'Complaint (finding)',
# MAGIC   'Enlargement (morphologic abnormality)',
# MAGIC   'Hospitalization 1',
# MAGIC   'Hospitalization 2',
# MAGIC   'Hospitalization 3',
# MAGIC   'Sign or Symptom',
# MAGIC   'Low Grade Prostatic Intraepithelial Neoplasia',
# MAGIC   'Physical activity',
# MAGIC   'Low Grade Cervical Squamous Intraepithelial Neoplasia',
# MAGIC   'Neoplasm Metastasis',
# MAGIC   'Fluid overload',
# MAGIC   'Finding of pH',
# MAGIC   'Past history of'
# MAGIC ) 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upset plot chart for IPF disease

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, split, explode
import pandas as pd
import matplotlib.pyplot as plt
from upsetplot import from_indicators, UpSet

# ✅ Create Spark session
spark = SparkSession.builder.getOrCreate()

# ✅ Define semantic types of interest
target_types = [
    "Disease or Syndrome", "Gene or Genome", "Therapeutic or Preventive Procedure",
    "Pathologic Function", "Diagnostic Procedure", "Sign or Symptom", "Neoplastic Process",
    "Organic Chemical", "Laboratory Procedure", "Congenital Abnormality",
    "Mental or Behavioral Dysfunction", "Pharmacologic Substance", "Genetic Function",
    "Amino Acid, Peptide, or Protein", "Anatomical Abnormality", "Laboratory or Test Result",
    "Clinical Drug", "Enzyme", "Cell or Molecular Dysfunction", "Finding"
]

# ✅ Load model results and filter by similarity and target semantic types
def load_filtered(model_name, table_name):
    df = spark.table(table_name).filter(col("similarity") >= 0.9)
    df = df.withColumn("semantic_type_array", split(col("semantic_type_name"), "\\|"))
    df = df.withColumn("model", lit(model_name))
    df = df.select(
        "disease",
        "UMLS_name",
        "model",
        explode(col("semantic_type_array")).alias("semantic_type")
    )
    return df.filter(col("semantic_type").isin(target_types))

# ✅ Combine results from all models (including Gemini)
df_all = (
    load_filtered("claude", "wei_lab_sander.llm_claude_umls_matched_results")
    .unionByName(load_filtered("ds", "wei_lab_sander.llm_ds_umls_matched_results"))
    .unionByName(load_filtered("o3", "wei_lab_sander.llm_o3_umls_matched_results"))
    .unionByName(load_filtered("ro", "wei_lab_sander.llm_ro_umls_matched_results"))
    .unionByName(load_filtered("gemini", "wei_lab_sander.llm_gemini_umls_matched_results"))
)

# ✅ Filter for IPF and remove duplicate UMLS concepts per model
target_disease = "Idiopathic Pulmonary Fibrosis"
df_ipf = (
    df_all
    .filter(col("disease") == target_disease)
    .select("UMLS_name", "model")
    .dropDuplicates()
)

# ✅ Convert to pandas and create a boolean pivot table
df_pd = df_ipf.toPandas()
df_pd["value"] = True

pivot_df = df_pd.pivot_table(
    index="UMLS_name",
    columns="model",
    values="value",
    aggfunc="any",
    fill_value=False
).astype(bool)

# ✅ Create UpSet plot
upset_data = from_indicators(pivot_df.columns.tolist(), pivot_df)

plt.figure(figsize=(10, 6))
UpSet(
    upset_data,
    show_counts=True,
    sort_by="cardinality"
).plot()

plt.title(f"UMLS Concept Overlap (Filtered by Semantic Type) – {target_disease}")
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes with parsed with negation/family...

# COMMAND ----------

# MAGIC %md
# MAGIC ### IPF case / IPF control

# COMMAND ----------

# Read the Excel file into a Pandas DataFrame
sign_symptom_df = pd.read_excel(
    "/Volumes/workspace_victrsd/wei_lab_sander/phemap/Case_sign_symptom_171_final.xlsx"
)


# Convert the Pandas DataFrame to a Spark DataFrame
spark_df = spark.createDataFrame(sign_symptom_df)

# Create or replace a temporary view
spark_df.createOrReplaceTempView("IPF_case_group_view")


sign_symptom_df1 = pd.read_excel(
    "/Volumes/workspace_victrsd/wei_lab_sander/phemap/Control_sign_symptom_171_final.xlsx"
)


# Convert the Pandas DataFrame to a Spark DataFrame
spark_df1 = spark.createDataFrame(sign_symptom_df1)

# Create or replace a temporary view
spark_df1.createOrReplaceTempView("IPF_control_group_view")

# COMMAND ----------

# MAGIC %sql
# MAGIC select 'case' as group,count(distinct person_id) as total_pt,count(*) as total_records,count(distinct cui) as unique_UMLS_CUI from IPF_case_group_view
# MAGIC union all
# MAGIC select 'control' as group,count(distinct person_id) as total_pt,count(*) as total_records,count(distinct cui) as unique_UMLS_CUI from IPF_control_group_view

# COMMAND ----------

# MAGIC %md
# MAGIC ### Non_IPF_control

# COMMAND ----------

# Read the Excel file into a Pandas DataFrame
sign_symptom_df = pd.read_excel(
    "/Volumes/workspace_victrsd/wei_lab_sander/phemap/non_IPF_control_0731.xlsx"
)

# Convert the Pandas DataFrame to a Spark DataFrame
spark_df = spark.createDataFrame(sign_symptom_df)

# Create or replace a temporary view
spark_df.createOrReplaceTempView("non_IPF_control_group_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Individual model appraoch

# COMMAND ----------

from pyspark.sql.functions import col

# 1. Load the CUI list and model list
df_cui_model = spark.table("wei_lab_sander.llm_sign_symptom_matched_results_final")
distinct_models = [row["model"] for row in df_cui_model.select("model").distinct().collect()]
cui_set = df_cui_model.select("CUI").distinct()

# 2. Load the original symptom view
df_symptom = (
    spark.table("non_IPF_control_group_view")
    .filter(~col("context_flag").rlike("(FAMILY|NEGATED_EXISTENCE|HISTORICAL|HYPOTHETICAL)"))
)

# 3. Filter and save data for each model
for model_name in distinct_models:
    
    print(f"Processing model: {model_name} ...")
    
    model_cui_set = (
        df_cui_model
        .filter(col("model") == model_name)
        .select("CUI")
        .distinct()
    )
    
    df_filtered = df_symptom.join(model_cui_set, on="CUI", how="inner")
    
    output_table = f"wei_lab_sander_ipf.non_IPF_control_group_{model_name}_sign_symptom_notes"
    print(f"Writing to table: {output_table} ...")
    
    df_filtered.write.mode("overwrite").saveAsTable(output_table)
    
    print(f"Finished writing table: {output_table}")

print("All processing complete.")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Case and non IPF_control note comparsion summary

# COMMAND ----------

sql = """
-- ===================== CASE =====================
select 'Gemini-2.5 Pro' as model, 'case' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.case_group_gemini_sign_symptom_notes
where person_id in (select case_id FROM wei_lab_sander.ipf_case_non_IPF_control_unique_pairs_final)

union all
select 'OpenAI o3' as model, 'case' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.case_group_o3_sign_symptom_notes
where person_id in (select case_id FROM wei_lab_sander.ipf_case_non_IPF_control_unique_pairs_final)

union all
select 'DeepSeek R1' as model, 'case' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.case_group_ds_sign_symptom_notes
where person_id in (select case_id FROM wei_lab_sander.ipf_case_non_IPF_control_unique_pairs_final)

union all
select 'Rarediseases.org' as model, 'case' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.case_group_ro_sign_symptom_notes
where person_id in (select case_id FROM wei_lab_sander.ipf_case_non_IPF_control_unique_pairs_final)

union all
select 'Claude-Sonnet-4' as model, 'case' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.case_group_claude_sign_symptom_notes
where person_id in (select case_id FROM wei_lab_sander.ipf_case_non_IPF_control_unique_pairs_final)

-- ===================== NON-IPF CONTROL =====================
union all
select 'Gemini-2.5 Pro' as model, 'non_IPF_control' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.non_IPF_control_group_gemini_sign_symptom_notes

union all
select 'OpenAI o3' as model, 'non_IPF_control' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.non_IPF_control_group_o3_sign_symptom_notes

union all
select 'DeepSeek R1' as model, 'non_IPF_control' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.non_IPF_control_group_ds_sign_symptom_notes

union all
select 'Rarediseases.org' as model, 'non_IPF_control' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.non_IPF_control_group_ro_sign_symptom_notes

union all
select 'Claude-Sonnet-4' as model, 'non_IPF_control' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.non_IPF_control_group_claude_sign_symptom_notes

"""

# Run SQL
df = spark.sql(sql)
display(df)

# ==========================================
# PIVOT CASE VS CONTROL
# ==========================================
from pyspark.sql.functions import col, round, expr
import matplotlib.pyplot as plt

df_pivot = df.groupBy("model").pivot("group", ["case", "non_IPF_control"]).agg(
    expr("first(umls_count)").alias("umls_count"),
    expr("first(note_count)").alias("note_count")
)

# ==========================================
# DELTA METRICS
# ==========================================
df_result = df_pivot.withColumn(
        "delta_umls",
        col("case_umls_count") - col("non_IPF_control_umls_count")
    ).withColumn(
        "delta_notes",
        col("case_note_count") - col("non_IPF_control_note_count")
    ).withColumn(
        "umls_per_note_case",
        round(col("case_umls_count") / col("case_note_count"), 2)
    ).withColumn(
        "umls_per_note_control",
        round(col("non_IPF_control_umls_count") / col("non_IPF_control_note_count"), 2)
    ).withColumn(
        "delta_umls_per_note",
        round(col("umls_per_note_case") - col("umls_per_note_control"), 2)
    )

# Convert to pandas
pdf = df_result.toPandas().sort_values(by="delta_umls", ascending=False)

# ==========================================
# PLOT: Delta UMLS (Case - Non-IPF-Control)
# ==========================================
plt.figure(figsize=(10, 5))
bars = plt.bar(pdf["model"], pdf["delta_umls"], color='steelblue')

plt.title("Study 1 - Delta Distinct UMLS CUI Count (Case - Non-IPF Control)", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Delta UMLS Count", fontsize=12)
plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{int(height)}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 1),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# COMMAND ----------

from pyspark.sql import functions as F

# ===========================
# MODEL LABELS
# ===========================
label_map = {
    "gemini": "Gemini-2.5 Pro",
    "o3": "OpenAI o3",
    "ds": "DeepSeek R1",
    "rare_disease_org": "Rarediseases.org",
    "claude": "Claude-Sonnet-4"
}

# ===========================
# LOAD FUNCTION
# ===========================
def load_sign_symptom_notes(model, group, table_name, filter_case_only=False):
    df = spark.table(table_name).select("person_id", "note_id")
    if filter_case_only:
        case_ids = [row["case_id"] for row in spark.table(
            "wei_lab_sander.ipf_case_non_IPF_control_unique_pairs_final"
        ).select("case_id").collect()]
        df = df.filter(F.col("person_id").isin(case_ids))

    # Apply human-readable labels here
    return df.withColumn("model", F.lit(label_map[model])) \
             .withColumn("group", F.lit(group))

# ===========================
# TABLE MAPPING
# ===========================
models = {
    "gemini": ("case", "wei_lab_sander_ipf.case_group_gemini_sign_symptom_notes", True),
    "o3": ("case", "wei_lab_sander_ipf.case_group_o3_sign_symptom_notes", True),
    "ds": ("case", "wei_lab_sander_ipf.case_group_ds_sign_symptom_notes", True),
    "rare_disease_org": ("case", "wei_lab_sander_ipf.case_group_ro_sign_symptom_notes", True),
    "claude": ("case", "wei_lab_sander_ipf.case_group_claude_sign_symptom_notes", True),

    "gemini_control": ("non_IPF_control", "wei_lab_sander_ipf.non_IPF_control_group_gemini_sign_symptom_notes", False),
    "o3_control": ("non_IPF_control", "wei_lab_sander_ipf.non_IPF_control_group_o3_sign_symptom_notes", False),
    "ds_control": ("non_IPF_control", "wei_lab_sander_ipf.non_IPF_control_group_ds_sign_symptom_notes", False),
    "rare_disease_org_control": ("non_IPF_control", "wei_lab_sander_ipf.non_IPF_control_group_ro_sign_symptom_notes", False),
    "claude_control": ("non_IPF_control", "wei_lab_sander_ipf.non_IPF_control_group_claude_sign_symptom_notes", False),
}

# ===========================
# UNION ALL MODEL DATA
# ===========================
all_df = None
for key, (group, table, is_case) in models.items():
    model = key.replace("_control", "")  # base model key
    df = load_sign_symptom_notes(model, group, table, is_case)
    all_df = df if all_df is None else all_df.unionByName(df)

# ===========================
# PER-PATIENT METRICS
# ===========================
umls_per_pt = all_df.groupBy("model", "group", "person_id").agg(
    F.count("*").alias("umls_count"),
    F.countDistinct("note_id").alias("note_count")
).withColumn(
    "umls_per_note", 
    F.col("umls_count") / F.col("note_count")
)

# ===========================
# GROUP-LEVEL METRICS
# ===========================
per_group_stats = umls_per_pt.groupBy("model", "group").agg(
    F.round(F.avg("umls_per_note"), 3).alias("mean_umls_per_note"),
    F.round(F.stddev_pop("umls_per_note"), 3).alias("std_umls_per_note"),
    F.count("*").alias("pt_count")
)

# ===========================
# PIVOT INTO CASE / CONTROL
# ===========================
pivot_df = per_group_stats.groupBy("model").pivot(
    "group", ["case", "non_IPF_control"]
).agg(
    F.first("mean_umls_per_note").alias("mean"),
    F.first("std_umls_per_note").alias("std")
)

pdf = pivot_df.toPandas().sort_values(by="case_mean", ascending=False)

# ===========================
# PLOT
# ===========================
# ===========================
# PLOT WITH CLEAN LABELS
# ===========================
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(pdf))
width = 0.35

plt.figure(figsize=(10, 6))

bars_case = plt.bar(
    x - width/2, 
    pdf["case_mean"], 
    yerr=pdf["case_std"], 
    width=width,
    label="Case", 
    capsize=5
)

bars_control = plt.bar(
    x + width/2, 
    pdf["non_IPF_control_mean"], 
    yerr=pdf["non_IPF_control_std"], 
    width=width,
    label="Non-IPF Control", 
    capsize=5
)

plt.xticks(x, pdf["model"], rotation=45)
plt.ylabel("UMLS per Note (per Patient)")
plt.title("Study 1 - UMLS per Note per Patient with SD")
plt.legend()

# ========================================
# Fractional shift label helper for beauty
# ========================================
def add_labels_fractional_shift(bars, x_shift_fraction):
    for bar in bars:
        height = bar.get_height()
        label_x = bar.get_x() + bar.get_width()/2 + x_shift_fraction * bar.get_width()
        plt.annotate(
            f"{height:.2f}",
            xy=(label_x, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9
        )

# Case labels slightly left inside the bar
add_labels_fractional_shift(bars_case, x_shift_fraction=-0.25)

# Control labels slightly right inside the bar
add_labels_fractional_shift(bars_control, x_shift_fraction=0.25)

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Case and control groups umls and note count summary

# COMMAND ----------

# ==========================================
# SQL WITH DISTINCT CUI + HUMAN-READABLE LABELS
# ==========================================
sql = """
-- ===================== CASE =====================
select 'Gemini-2.5 Pro' as model, 'case' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.case_group_gemini_sign_symptom_notes

union all
select 'OpenAI o3' as model, 'case' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.case_group_o3_sign_symptom_notes

union all
select 'DeepSeek R1' as model, 'case' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.case_group_ds_sign_symptom_notes

union all
select 'Rarediseases.org' as model, 'case' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.case_group_ro_sign_symptom_notes

union all
select 'Claude-Sonnet-4' as model, 'case' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.case_group_claude_sign_symptom_notes

-- ===================== CONTROL =====================
union all
select 'Gemini-2.5 Pro' as model, 'control' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.control_group_gemini_sign_symptom_notes

union all
select 'OpenAI o3' as model, 'control' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.control_group_o3_sign_symptom_notes

union all
select 'DeepSeek R1' as model, 'control' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.control_group_ds_sign_symptom_notes

union all
select 'Rarediseases.org' as model, 'control' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.control_group_ro_sign_symptom_notes

union all
select 'Claude-Sonnet-4' as model, 'control' as group,
       count(distinct concat(note_id, '_', cui)) as umls_count,
       count(distinct person_id) as pt_count,
       count(distinct note_id) as note_count
from wei_lab_sander_ipf.control_group_claude_sign_symptom_notes
"""

df = spark.sql(sql)
display(df)

# ==========================================
# PIVOT CASE VS CONTROL
# ==========================================
from pyspark.sql.functions import col, round, expr
import matplotlib.pyplot as plt

df_pivot = df.groupBy("model").pivot("group", ["case", "control"]).agg(
    expr("first(umls_count)").alias("umls_count"),
    expr("first(note_count)").alias("note_count")
)

# ==========================================
# DELTA METRICS
# ==========================================
df_result = df_pivot.withColumn(
        "delta_umls",
        col("case_umls_count") - col("control_umls_count")
    ).withColumn(
        "delta_notes",
        col("case_note_count") - col("control_note_count")
    ).withColumn(
        "umls_per_note_case",
        round(col("case_umls_count") / col("case_note_count"), 2)
    ).withColumn(
        "umls_per_note_control",
        round(col("control_umls_count") / col("control_note_count"), 2)
    ).withColumn(
        "delta_umls_per_note",
        round(col("umls_per_note_case") - col("umls_per_note_control"), 2)
    )

# Convert to pandas
pdf = df_result.toPandas().sort_values(by="delta_umls", ascending=False)

# ==========================================
# PLOT: Delta Distinct UMLS (Case - Control)
# ==========================================
plt.figure(figsize=(10, 5))
bars = plt.bar(pdf["model"], pdf["delta_umls"], color='steelblue')

plt.title("Study 2 - Delta Distinct UMLS CUI Count (Case - Control)", fontsize=14)
plt.xlabel("Model", fontsize=12)
plt.ylabel("Delta UMLS Count", fontsize=12)
plt.xticks(rotation=45)

for bar in bars:
    height = bar.get_height()
    plt.annotate(f'{int(height)}',
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 1),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()


# COMMAND ----------

sql = """
WITH base AS (
  SELECT 'Gemini-2.5 Pro' AS model, 'case' AS group, person_id, note_id
  FROM wei_lab_sander_ipf.case_group_gemini_sign_symptom_notes
  UNION ALL
  SELECT 'OpenAI o3', 'case', person_id, note_id
  FROM wei_lab_sander_ipf.case_group_o3_sign_symptom_notes
  UNION ALL
  SELECT 'DeepSeek R1', 'case', person_id, note_id
  FROM wei_lab_sander_ipf.case_group_ds_sign_symptom_notes
  UNION ALL
  SELECT 'Rarediseases.org', 'case', person_id, note_id
  FROM wei_lab_sander_ipf.case_group_ro_sign_symptom_notes
  UNION ALL
  SELECT 'Claude-Sonnet-4', 'case', person_id, note_id
  FROM wei_lab_sander_ipf.case_group_claude_sign_symptom_notes

  UNION ALL
  SELECT 'Gemini-2.5 Pro', 'control', person_id, note_id
  FROM wei_lab_sander_ipf.control_group_gemini_sign_symptom_notes
  UNION ALL
  SELECT 'OpenAI o3', 'control', person_id, note_id
  FROM wei_lab_sander_ipf.control_group_o3_sign_symptom_notes
  UNION ALL
  SELECT 'DeepSeek R1', 'control', person_id, note_id
  FROM wei_lab_sander_ipf.control_group_ds_sign_symptom_notes
  UNION ALL
  SELECT 'Rarediseases.org', 'control', person_id, note_id
  FROM wei_lab_sander_ipf.control_group_ro_sign_symptom_notes
  UNION ALL
  SELECT 'Claude-Sonnet-4', 'control', person_id, note_id
  FROM wei_lab_sander_ipf.control_group_claude_sign_symptom_notes
),

umls_counts AS (
  SELECT model, group, person_id, COUNT(*) AS umls_count
  FROM base
  GROUP BY model, group, person_id
),

note_counts AS (
  SELECT model, group, person_id, COUNT(DISTINCT note_id) AS note_count
  FROM base
  GROUP BY model, group, person_id
),

per_patient_note_avg AS (
  SELECT u.model,
         u.group,
         u.person_id,
         u.umls_count,
         n.note_count,
         CAST(u.umls_count AS DOUBLE) / n.note_count AS umls_per_note
  FROM umls_counts u
  JOIN note_counts n
    ON u.model = n.model AND u.group = n.group AND u.person_id = n.person_id
)

SELECT
  model,
  group,
  ROUND(AVG(umls_per_note), 3) AS umls_per_note_avg,
  ROUND(STDDEV_POP(umls_per_note), 3) AS umls_per_note_std,
  COUNT(*) AS pt_count
FROM per_patient_note_avg
GROUP BY model, group
ORDER BY model, group
"""

from pyspark.sql.functions import col, expr
import matplotlib.pyplot as plt
import numpy as np

df_stats = spark.sql(sql)

pivot = df_stats.groupBy("model").pivot("group", ["case", "control"]).agg(
    expr("first(umls_per_note_avg)").alias("mean"),
    expr("first(umls_per_note_std)").alias("std")
)

pdf = pivot.toPandas().sort_values(by="case_mean", ascending=False)

x = np.arange(len(pdf))
width = 0.35

plt.figure(figsize=(10, 6))

bars_case = plt.bar(
    x - width/2,
    pdf["case_mean"],
    yerr=pdf["case_std"],
    width=width,
    label="Case",
    capsize=5
)

bars_control = plt.bar(
    x + width/2,
    pdf["control_mean"],
    yerr=pdf["control_std"],
    width=width,
    label="Control",
    capsize=5
)

plt.xticks(x, pdf["model"], rotation=45)
plt.ylabel("UMLS per Note per Patient")
plt.title("Study 2 - UMLS per Note per Patient with SD")
plt.legend()

# ============================
# Clean numeric labels on bars
# ============================
def add_labels_fractional_shift(bars, x_shift_fraction):
    for bar in bars:
        height = bar.get_height()
        # position = bar center + fractional shift * bar width
        label_x = bar.get_x() + bar.get_width()/2 + x_shift_fraction * bar.get_width()
        plt.annotate(
            f"{height:.2f}",
            xy=(label_x, height),
            xytext=(0, 2),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=9
        )

# Case labels slightly left inside bar
add_labels_fractional_shift(bars_case, x_shift_fraction=-0.25)

# Control labels slightly right inside bar
add_labels_fractional_shift(bars_control, x_shift_fraction=0.25)

plt.tight_layout()
plt.show()
