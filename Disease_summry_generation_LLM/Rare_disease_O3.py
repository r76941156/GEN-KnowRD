# Databricks notebook source
pip install --upgrade "mlflow[databricks]>=3.1" openai

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import os 
import csv
import base64

from openai import OpenAI
import json
import re
import numpy as np
import datetime
import pandas as pd
import time
import ast
from collections import defaultdict
import pandas as pd
from google import genai
from google.genai import types
import anthropic
import mlflow

import json
from pyspark.sql import Row, SparkSession
from pyspark.sql.utils import AnalysisException

# COMMAND ----------

os.environ["OPENAI_API_KEY"]= "xxx"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Function List

# COMMAND ----------


spark = SparkSession.builder.getOrCreate()

# ✅ Extract output_text from response JSON
def extract_output_text(response_json_str):
    try:
        response_dict = json.loads(response_json_str or '{}')
        output_list = response_dict.get("output", [])
        for item in output_list:
            if item.get("type") == "message" and isinstance(item.get("content"), list):
                for content_item in item["content"]:
                    if content_item.get("type") == "output_text" and "text" in content_item:
                        return content_item["text"]
        return None
    except Exception as e:
        print("Failed to extract disease_text:", e)
        return None

# ✅ Flatten any JSON string to a Row (wide format)
def flatten_json_to_row(json_str, extra_cols=None):
    try:
        data = json.loads(json_str or '{}')
    except Exception as e:
        print("Failed to parse JSON:", e)
        data = {}
    flat_data = {
        k: json.dumps(v) if isinstance(v, (dict, list, type(None))) else str(v)
        for k, v in data.items()
    }
    if extra_cols:
        flat_data.update(extra_cols)
    return Row(**flat_data)

# ✅ Append if table exists, otherwise create
def safe_write_to_table(df, table_name):
    try:
        spark.table(table_name)
        df.write.mode("append").saveAsTable(table_name)
    except AnalysisException:
        df.write.mode("overwrite").saveAsTable(table_name)

def save_trace(model,disease_name,total_running_time):
    
    request_tbl = "wei_lab_sander_mlflow.llm_o3_request1"
    response_tbl = "wei_lab_sander_mlflow.llm_o3_response1"
        

    # Get the trace
    trace_id = mlflow.get_last_active_trace_id()
    print(f"Start parsing trace ID: {trace_id}")

    trace = mlflow.get_trace(trace_id)
    # Access request and response
    request_json = trace.data.request
    response_json = trace.data.response

    # --------- Request ---------
  

    request_row = flatten_json_to_row(request_json, extra_cols={
        "disease_name": disease_name,
        "total_running_time": total_running_time
    })
    df_request = spark.createDataFrame([request_row])
    safe_write_to_table(df_request, request_tbl)

    # --------- Response ---------
    output_text = extract_output_text(response_json)
    response_row = flatten_json_to_row(response_json, extra_cols={
        "disease_name": disease_name,
        "disease_text": output_text,
        "total_running_time": total_running_time
    })
    df_response = spark.createDataFrame([response_row])
    safe_write_to_table(df_response, response_tbl)

    
    print("Finished!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt template

# COMMAND ----------

prompt_template_1 = '''You are an expert medical researcher. Your task is to generate a comprehensive and authoritative summary of {disease}
, using the guidance provided within the triple quotation marks. This summary will serve as a reliable Wikipedia entry for use by medical researchers and healthcare professionals. In the "Clinical Presentation" section, you must include exhaustive information to guide clinical assessment. Ensure that all information is well-grounded in verified sources. Only use free-text descriptions with complete sentences. Do not use any other format. Again, you need to be as comprehensive as possible. Otherwise, you will be penalized.

"""
1 Disease Overview
•	Definition: Provide a lay summary description of the disorder.
•	Key Features: List hallmark clinical characteristics (≤ 3 phrases).
•	Disease Category/Class: Identify broad classification (e.g., autoimmune, genetic, fibrotic lung disease).
2 Synonyms & Abbreviations
•	Alternate Names: List all official and historical names.
•	Preferred Acronym(s): Record standard abbreviations.
3 Subtypes / Variants
•	Subdivisions: Enumerate recognized forms or phenotypes; note distinguishing criteria (age, severity, genetic marker, etc.).
•	If none: State “No formal subtypes reported.”
4 Epidemiology
•	Prevalence / Incidence: Quantify (rate per population) and specify data source year/country.
•	Demographics: Capture typical age of onset, sex ratio, ethnic or geographic clustering.
•	Rarity Status: Indicate if classified as “rare” (≤ 200 K in U.S. or regional equivalent).
5 Etiology & Pathogenesis
•	Primary Cause(s): Describe genetic mutation, autoimmune trigger, infection, toxin, or “idiopathic.”
•	Inheritance Pattern (if genetic): State autosomal dominant/recessive, X-linked, mitochondrial, etc.
•	Pathophysiologic Mechanism: Summarize how the cause produces tissue damage or dysfunction.
•	Key Risk Factors: List established environmental or lifestyle contributors.
6 Clinical Presentation
•	Core Signs & Symptoms: Provide a comprehensive list, starting with most common/early and including late signs and symptoms.
•	Progression Pattern: Detail typical progression patterns (acute, relapsing, chronic progressive).
•	Variability between patients: Describe known variability in presentation.
•	Major Complications: Note life-threatening or disabling sequelae.
7 Diagnostic Evaluation
•	Clinical Criteria: Summarize key bedside findings required for diagnosis.
•	Laboratory Tests: Specify biomarkers, antibody assays, enzyme levels, etc.
•	Imaging / Instrumental Tests: List radiology, electrophysiology, biopsies essential for confirmation.
•	Genetic Testing (if applicable): State recommended gene panels or specific variant analysis.
•	Formal Guidelines: Reference any published diagnostic criteria sets.
8 Management & Standard Therapy
•	First-Line Treatments: Name drugs, doses (range), or procedures routinely recommended.
•	Second-Line / Adjunctive: List options for refractory or severe disease.
•	Supportive Care: Include rehabilitation, nutritional guidance, devices, or monitoring protocols.
•	Preventive Measures: Note prophylactic strategies (vaccines, lifestyle modifications).
9 Investigational / Emerging Therapies
•	Therapies in Trials: Summarize novel agents, biologics, gene or cell therapies under clinical investigation.
•	Trial Resources: Provide registry links or identifiers when available.
10 Prognosis
•	Natural History: Describe typical survival or remission expectations without treatment.
•	Impact of Therapy: State how modern treatment alters outcomes.
•	Prognostic Factors: List variables associated with better or worse course.
"""
'''

# COMMAND ----------

# MAGIC %md
# MAGIC ### Handle diseases by diff defined groups

# COMMAND ----------

df = spark.sql("select Doc_ID from wei_lab_sander_ipf.rare_disease_org_list where group_num in ('0')")
diseases = [row['Doc_ID'] for row in df.collect()]
display(diseases)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use O3 to generate text

# COMMAND ----------


# Get already processed diseases
processed_df = spark.sql("select disease_name from wei_lab_sander_mlflow.llm_o3_response")
processed_diseases = set([row['disease_name'] for row in processed_df.collect()])

# Enable auto-tracing for OpenAI
mlflow.openai.autolog()

# Set up MLflow tracking to Databricks
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/openai-tracing-demo")

client = OpenAI()
#diseases = ['Maffucci Syndrome','47 XXY Klinefelter Syndrome'] #test only

# Loop through diseases
for disease in diseases:
    if disease in processed_diseases:
       print(f"Skipping disease: {disease}")
       continue
    
    start_time = time.time()

    # Format prompts
    prompt_1 = prompt_template_1.format(disease=disease)
    
    
    # First response
    response_1 = client.responses.create(
        model="o3",  
        tools=[{"type": "web_search_preview"}],  
        input=[
            {"role": "system", "content": "You are a rare disease expert"},
            {"role": "user", "content": prompt_1}
        ]
    )
    ### wait 10 seconds for next disease
    end_time = time.time()
    total_running_time = round(end_time - start_time, 2)

    time.sleep(10)

    # Get the trace
    save_trace("o3",disease,total_running_time)    

    

print("Done!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review Output: MLFlow change output format for o3, so we have to combine old/newresponses to have a final version

# COMMAND ----------

# MAGIC %sql
# MAGIC -- create or replace table wei_lab_sander_mlflow.llm_o3_response_final as
# MAGIC -- select * from (
# MAGIC -- select model,output,parallel_tool_calls,temperature,tool_choice,tools,top_logprobs
# MAGIC -- ,usage,disease_name,disease_text,total_running_time from wei_lab_sander_mlflow.llm_o3_response
# MAGIC -- union
# MAGIC
# MAGIC -- select model,output,parallel_tool_calls,temperature,tool_choice,tools,top_logprobs
# MAGIC -- ,usage,disease_name,disease_text,total_running_time from wei_lab_sander_mlflow.llm_o3_response1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validation

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_mlflow.llm_o3_response_final