# Databricks notebook source
pip install openpyxl textstat

# COMMAND ----------

# MAGIC %pip install scispacy
# MAGIC %pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
# MAGIC %pip install tiktoken

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %sh TOKENIZERS_PARALLELISM=false

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.autocast.*")


# COMMAND ----------

import pandas as pd
import numpy as np
from pathlib import Path
import spacy
from scispacy.linking import EntityLinker
from transformers import AutoTokenizer
import tiktoken
import os
import warnings
import time
import traceback
from collections import defaultdict

print("🔄 Loading UMLS spaCy pipeline...")

nlp_umls = spacy.load("en_core_sci_scibert")

nlp_umls.add_pipe("scispacy_linker", name="scispacy_umls_linker", config={
    "threshold": 0.8,
    "resolve_abbreviations": True,
    "linker_name": "umls"
})
umls_linker = nlp_umls.get_pipe("scispacy_umls_linker")

if "sentencizer" not in nlp_umls.pipe_names:
    nlp_umls.add_pipe("sentencizer")

hf_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
enc = tiktoken.get_encoding("cl100k_base")

# COMMAND ----------

# MAGIC %md
# MAGIC ### UMLS semantic type mapping

# COMMAND ----------

# # Read the Excel file into a Pandas DataFrame
# rare_disease_txt_df = pd.read_excel(
#     "/Volumes/workspace_victrsd/wei_lab_sander/phemap/rare_disease_org_0731.xlsx"
# )
# # Convert the Pandas DataFrame to a Spark DataFrame
# spark_df = spark.createDataFrame(rare_disease_txt_df)

# # Create or replace a temporary view
# spark_df.createOrReplaceTempView("rare_disease_txt_view")



# COMMAND ----------

# %sql
# select disease_name,disease_text from rare_disease_txt_view 
# where disease_name in (
# select Doc_ID from wei_lab_sander_ipf.rare_disease_org_list where group_num between 1 and 10
# )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Load UMLS mappings (For example: T034 - Laboratory or Test Result)

# COMMAND ----------

sem_type_df = pd.read_excel(
    "/Volumes/workspace_victrsd/wei_lab_sander/phemap/umls_mapping.xlsx",
    header=None,
    names=["TUI", "Type_name"]
)
sem_type_map = dict(zip(sem_type_df["TUI"], sem_type_df["Type_name"]))

display(sem_type_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load dataset (e.g., mygene2)

# COMMAND ----------

# Read the CSV file into a Spark DataFrame
mygene2_df = spark.read.csv("/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/mygene2_processed_output.csv", header=True, inferSchema=True)

# Create or replace a temporary view
mygene2_df.createOrReplaceTempView("mygene2_view")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create table with necessary cols

# COMMAND ----------

# MAGIC %sql
# MAGIC --create or replace table wei_lab_sander_umls_mapping.mygene2_dataset as
# MAGIC --select row_number() over (order by rare_disease_cui) as patient_id, rare_disease_cui,rare_disease_name,phenotype_umls_cuis from mygene2_view order by rare_disease_cui
# MAGIC
# MAGIC select * from wei_lab_sander_umls_mapping.mygene2_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Add disease complexity table

# COMMAND ----------

# cui_category_df = spark.read.csv("/Volumes/workspace_victrsd/wei_lab_sander_umls_mapping/public_dataset/cui_category_review.csv", header=True, inferSchema=True)
# cui_category_df.createOrReplaceTempView("cui_category_review_view")

# COMMAND ----------

# %sql
# select * from cui_category_review_view

# COMMAND ----------

# %sql
# create or replace table wei_lab_sander.rare_disease_cui_category_final as
# select * from wei_lab_sander.rare_disease_cui_category 
# where rare_disease_name not in (
#   'Nonketotic Hyperglycinemia',
#   'Arthrogryposis Renal Dysfunction Cholestasis Syndrome',
#   'Oral-Facial-Digital Syndrome',
#   'Treacher Collins Syndrome',
#   'Hypophosphatasia',
#   'Kleine-Levin Syndrome',
#   'Gaucher Disease',
#   'Marshall Smith Syndrome',
#   'Biotinidase Deficiency',
#   'Leukocyte Adhesion Deficiency Syndromes',
#   'Locked In Syndrome',
#   'Galactosemia',
#   'Adenylosuccinate Lyase Deficiency',
#   'Neonatal Lupus',
#   'Tyrosinemia Type 1',
#   'Phenylketonuria',
#   'Marden Walker Syndrome',
#   'Neurofibromatosis 1',
#   'Common Variable Immune Deficiency',
#   'Respiratory Distress Syndrome Infant',
#   'West Syndrome',
#   'Bosma Arhinia Microphthalmia Syndrome',
#   'Turcot Syndrome',
#   'Stickler Syndrome',
#   'Cockayne Syndrome',
#   'Kleefstra Syndrome',
#   'Primary Hyperoxaluria',
#   'Freeman Sheldon Syndrome',
#   'Riboflavin Transporter Deficiency',
#   'Chromosome 22q112 Deletion Syndrome',
#   'Oculocutaneous Albinism',
#   'Dystonia',
#   'Rubinstein-Taybi Syndrome',
#   'Larsen Syndrome',
#   'KCNQ2 Developmental and Epileptic Encephalopathy',
#   'Hereditary Leiomyomatosis and Renal Cell Carcinoma',
#   'Hypohidrotic Ectodermal Dysplasia',
#   'Nemaline Myopathy',
#   'Craniometaphyseal Dysplasia',
#   'Nonketotic Hyperglycinemia',
#   'Focal Segmental Glomerulosclerosis'
# )
# union 
# select * from cui_category_review_view

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander.rare_disease_cui_category_final

# COMMAND ----------

# MAGIC %md
# MAGIC ### Disease summary (count: 798)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select distinct a.rare_disease_name,b.code as orpha_code,b.rare_disease_cui from (
# MAGIC select distinct rare_disease_name from wei_lab_sander_umls_mapping.hms_dataset
# MAGIC union
# MAGIC select distinct rare_disease_name from wei_lab_sander_umls_mapping.mme_dataset
# MAGIC union
# MAGIC select distinct rare_disease_name from wei_lab_sander_umls_mapping.mygene2_dataset
# MAGIC union
# MAGIC select distinct rare_disease_name from wei_lab_sander_umls_mapping.lirical_dataset
# MAGIC union
# MAGIC select distinct rare_disease_name from wei_lab_sander_umls_mapping.ramedis_dataset
# MAGIC union
# MAGIC select distinct rare_disease_name from wei_lab_sander_umls_mapping.pubmed_dataset_update
# MAGIC ) a 
# MAGIC inner join wei_lab_sander.rare_disease_cui_category_final b
# MAGIC on a.rare_disease_name = b.rare_disease_name
# MAGIC --798 disease
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Main UMLS parser

# COMMAND ----------

def link_umls_entities(nlp_model, text, model, disease_name ,linker, max_tokens=512):
  
    try:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty or invalid text content.")

        wordpiece_len = len(hf_tokenizer.tokenize(text))
        token_chunks = []
        
        if wordpiece_len <= max_tokens:
            doc = nlp_model(text)
            token_chunks.append((0, doc, 0))
        else:
            doc_full = nlp_model.make_doc(text)
            doc_full = nlp_model.get_pipe("sentencizer")(doc_full)

            current_text = ""
            current_start = None
            group_idx = 0
            

            for sent in doc_full.sents:
                sent_text = sent.text.strip()
                if not sent_text:
                    continue

                if current_start is None:
                    current_start = sent.start_char

                combined_text = f"{current_text} {sent_text}".strip() if current_text else sent_text
                wp_len = len(hf_tokenizer.tokenize(combined_text))
                
                if wp_len <= max_tokens:
                    current_text = combined_text
                    
                    
                else:
                    if current_text.strip():
                        chunk_doc = nlp_model(current_text)
                        token_chunks.append((group_idx, chunk_doc, current_start))
                        group_idx += 1
                    current_text = sent_text
                    current_start = sent.start_char

            if current_text.strip():
                chunk_doc = nlp_model(current_text)
                token_chunks.append((group_idx, chunk_doc, current_start))
                

        results = []
        for chunk_index, chunk_doc, chunk_offset in token_chunks:
            for entity in chunk_doc.ents:
                try:
                    for cui, score in entity._.kb_ents:
                        concept = linker.kb.cui_to_entity.get(cui)

                        sentence_span = entity.sent
                        sentence_text = sentence_span.text.strip() if sentence_span else ""
                        sentence_start = chunk_offset + sentence_span.start_char if sentence_span else None
                        sentence_end = chunk_offset + sentence_span.end_char if sentence_span else None

                        tui_list = concept.types  # e.g., ["T109", "T121"]
                        type_names = [sem_type_map.get(tui, tui) for tui in tui_list]

                        results.append({
                            "model": model,
                            'disease': disease_name,
                            "chunk_index": chunk_index,
                            "UMLS_name": concept.canonical_name,
                            "CUI": cui,
                            "similarity": score,
                            "semantic_type": tui_list,
                            "semantic_type_name": "| ".join(type_names),
                            "matched_text": entity.text,
                            "start_char": chunk_offset + entity.start_char,
                            "end_char": chunk_offset + entity.end_char,
                            "sentence": sentence_text,
                            "sentence_start": sentence_start,
                            "sentence_end": sentence_end
                        })
                except Exception:
                    traceback.print_exc()

        return pd.DataFrame(results)

    except Exception as e:
        print(f"Skipping text due to parsing error: {e}")
        return pd.DataFrame()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Datasets:
# MAGIC - IPF notes
# MAGIC - Case reports from public PMC datasets
# MAGIC - NORD 1320 disease summaries
# MAGIC - LLM generated disease summaries

# COMMAND ----------

rare_disease_txt_df = pd.read_excel(
    "/Volumes/workspace_victrsd/wei_lab_sander/phemap/rare_disease_org_0731.xlsx"
)
spark.createDataFrame(rare_disease_txt_df).createOrReplaceTempView("rare_disease_txt_view")
display(rare_disease_txt_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Utilize different datasets to get UMLS CUIs

# COMMAND ----------

import pandas as pd
import re

def remove_links(text):
    if not isinstance(text, str):
        return text
    # Remove markdown links: [text](url)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\)]+)\)", r"\1", text)
    # Remove bare URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove parenthetical domains
    text = re.sub(r"\((?:\s*[\w\.-]+\.[a-z]{2,}(?:,)?\s*)+\)", "", text)
    return text


# 1. Load LLM input file
model_list = ["o3", "claude", "gemini", "ds"]


for model in model_list:
    
    #public datasets
    if model == 'o3':
        tbl = f"wei_lab_sander_mlflow.llm_o3_response_final"
    else:
        tbl = f"wei_lab_sander_mlflow.llm_{model}_response"

    llm_df = spark.sql(f"""
        SELECT disease_name, disease_text AS LLM_text
        FROM {tbl}
    """)

    print(f"🔄 Processing {llm_df.count()} rows...")

    # output table remains the same
    output_table = f"wei_lab_sander.llm_{model}_umls_matched_results_new"

    # 2. Process each LLM entry ONE BY ONE
    for i, row in enumerate(llm_df.collect(), 1):

        text = row.LLM_text
        disease_name = row.disease_name

        # remove link formatting for O3
        if model == "o3":
            text = remove_links(text)

        print(f"📥 Loading LLM text from {model} DB... (row {i})")

        df_linked = link_umls_entities(
            nlp_model=nlp_umls,
            text=text,
            model=model,
            disease_name=disease_name,
            linker=umls_linker
        )

        if df_linked is None or df_linked.empty:
            print(f"⚠️ No UMLS concepts matched in row {i} (model: {model})")
            continue

        # save THIS disease immediately
        spark_df = spark.createDataFrame(df_linked)

        if not spark.catalog.tableExists(output_table):
            spark_df.write.mode("overwrite").saveAsTable(output_table)
        else:
            spark_df.write.mode("append").saveAsTable(output_table)

        print(f"   ✅ Saved {len(df_linked)} UMLS records for: {disease_name}")

print("🎉 Completed all models.")


# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) as record_count from wei_lab_sander.llm_ds_umls_matched_results 
# MAGIC union
# MAGIC select count(*) as record_count from wei_lab_sander.llm_claude_umls_matched_results 
# MAGIC union
# MAGIC select count(*) as record_count from wei_lab_sander.llm_gemini_umls_matched_results 
# MAGIC union
# MAGIC select count(*) as record_count from wei_lab_sander.llm_o3_umls_matched_results 
# MAGIC union
# MAGIC select count(*) as record_count from wei_lab_sander.llm_ro_umls_matched_results 