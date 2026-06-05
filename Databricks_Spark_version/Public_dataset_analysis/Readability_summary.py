# Databricks notebook source
# MAGIC %md
# MAGIC ### Readability score summary

# COMMAND ----------

pip install textstat spacy openpyxl

# COMMAND ----------

pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load rare disease txt

# COMMAND ----------

import pandas as pd
rare_disease_txt_df = pd.read_excel(
    "/Volumes/workspace_victrsd/wei_lab_sander/phemap/rare_disease_org_0731.xlsx"
)

# Convert the Pandas DataFrame to a Spark DataFrame
spark_df = spark.createDataFrame(rare_disease_txt_df)

# Create or replace a temporary view
spark_df.createOrReplaceTempView("rare_disease_txt_view")


# COMMAND ----------

# MAGIC %sql
# MAGIC select disease_name, disease_text from rare_disease_txt_view

# COMMAND ----------

# MAGIC %md
# MAGIC ### Main Function

# COMMAND ----------

import spacy
from textstat.textstat import textstatistics
import pandas as pd
import re
from textstat import textstat
legacy_round = textstat._legacy_round

# ✅ Load spaCy model only once
nlp = spacy.load("en_core_sci_md")
textstat = textstatistics()

# === Your Custom Readability Functions ===
def break_sentences(text):
    doc = nlp(text)
    return list(doc.sents)

def word_count(text):
    return sum(len([token for token in sentence]) for sentence in break_sentences(text))

def sentence_count(text):
    return len(break_sentences(text))

def avg_sentence_length(text):
    words = word_count(text)
    sentences = sentence_count(text)
    return float(words / sentences) if sentences else 0.0

def syllables_count(word):
    try:
        return textstat.syllable_count(word)
    except:
        return 0

def avg_syllables_per_word(text):
    syllables = syllables_count(text)
    words = word_count(text)
    return legacy_round(float(syllables) / float(words), 1) if words else 0.0

def poly_syllable_count(text):
    count = 0
    words = []
    for sentence in break_sentences(text):
        words += [token for token in sentence]
    for word in words:
        if syllables_count(word.text) >= 3:
            count += 1
    return count

def smog_index(text):
    if sentence_count(text) >= 3:
        poly_syllab = poly_syllable_count(text)
        SMOG = (1.043 * (30 * (poly_syllab / sentence_count(text))) ** 0.5) + 3.1291
        return round(SMOG, 2)  # ✅ two decimal places
    else:
        return 0.0

# === Optional Cleaner ===
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\*\*.*?\*\*", "", text)
    text = re.sub(r"#+", "", text)
    return text.strip()

def remove_links(text):
    if not isinstance(text, str):
        return text
    # Remove markdown links: [text](url)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", text)
    # Remove bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove parenthetical domains (e.g., (rarediseases.org, pubmed.ncbi.nlm.nih.gov))
    text = re.sub(r"\((?:\s*[\w\.-]+\.[a-z]{2,}(?:,)?\s*)+\)", "", text)
    return text

# === Process via PySpark Driver ===
models = ['ro','o3', 'gemini', 'ds','Claude']
all_records = []

for model in models:
    if model == "ro":
        df = spark.sql ("""select disease_name,disease_text as LLM_text from rare_disease_txt_view 
            
            """)

    elif model =="o3":    
        df = spark.sql(f"""
        SELECT DISTINCT disease_name, disease_text AS LLM_text 
        FROM wei_lab_sander_mlflow.llm_{model}_response_final
        """)    
   
    else:    
        df = spark.sql(f"""
        SELECT DISTINCT disease_name, disease_text AS LLM_text 
        FROM wei_lab_sander_mlflow.llm_{model}_response
        """)
        
    pdf = df.toPandas()

    for _, row in pdf.iterrows():
        text = clean_text(row["LLM_text"])
        if model == "o3":
            text = remove_links(text)
        disease_name = row["disease_name"]

        print(f"Processing ({model}):", text[:100])

        try:
            score = smog_index(text)
        except Exception as e:
            print("Error:", e)
            score = None

        all_records.append({
            "disease_name": disease_name,
            "LLM_text": text,
            "model": model,
            "smog_index": score
        })

# Convert to Spark DataFrame
result_df = spark.createDataFrame(pd.DataFrame(all_records))
result_df.write.mode("overwrite").saveAsTable("wei_lab_sander_mlflow.llm_text_smog_scores")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Avd (SD) SMOG scores between models

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   model,
# MAGIC   ROUND(AVG(smog_index), 2) AS avg_smog,
# MAGIC   ROUND(STDDEV(smog_index), 2) AS stddev_smog
# MAGIC FROM
# MAGIC   wei_lab_sander_mlflow.llm_text_smog_scores
# MAGIC GROUP BY
# MAGIC   model
# MAGIC ORDER BY
# MAGIC   model
# MAGIC