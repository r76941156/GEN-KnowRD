# Databricks notebook source
pip install google-genai


# COMMAND ----------

pip install --upgrade "mlflow[databricks]>=3.1"

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
from pyspark.sql.types import StructType, StructField, StringType

from mlflow.entities import SpanType



# COMMAND ----------

gemini_api = "xxx"


# COMMAND ----------

# MAGIC %md
# MAGIC ## Prompt Template

# COMMAND ----------

prompt_template_1 = '''You are an expert medical researcher. Your task is to generate a comprehensive and authoritative summary of {disease}
, using the guidance provided within the triple quotation marks. This summary will serve as a reliable Wikipedia entry for use by medical researchers and healthcare professionals. In the "Clinical Presentation" section, you must include exhaustive information to guide clinical assessment. Ensure that all information is well-grounded in verified sources. Only use free-text descriptions with complete sentences. Do not use any other format. Again, you need to be as comprehensive as possible. Otherwise, you will be penalized.

The entire output must be written in English.

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
# MAGIC ### Function List

# COMMAND ----------


def flatten_json_to_row(json_obj, extra_cols=None):
    flat = {}
    for k, v in json_obj.items():
        # If nested, store as stringified JSON
        if isinstance(v, (dict, list)):
            flat[k] = json.dumps(v)
        else:
            flat[k] = v
    if extra_cols:
        flat.update(extra_cols)
    return flat

def save_gemini_trace(disease_name,total_running_time):

    request_tbl = "wei_lab_sander_mlflow.llm_gemini_request"
    response_tbl = "wei_lab_sander_mlflow.llm_gemini_response"

    trace_id = mlflow.get_last_active_trace_id()
    print(f"Start parsing trace ID: {trace_id}")
    trace = mlflow.get_trace(trace_id)

    
    # === Step 1: Save request info ===
    spans = trace.search_spans(name="build_prompt")
    if not spans:
        print("No span named 'build_prompt' found.")
        return

    span = spans[0]
    request_prompt = span.outputs

    request_row = {
        "trace_id": trace_id,
        "disease_name": disease_name,
        "request_prompt": request_prompt,
        "total_running_time": total_running_time
    }

    request_row = {k: str(v) if v is not None else None for k, v in request_row.items()}
    request_schema = StructType([StructField(k, StringType(), True) for k in request_row.keys()])
    df_request = spark.createDataFrame([request_row], schema=request_schema)
    df_request.write.mode("append").saveAsTable(request_tbl)

   
    # === Step 2: Save response info ===
    spans = trace.search_spans(name="send_disease_query")
    if not spans:
        print("No span named 'send_disease_query' found.")
        return

    span = spans[0]
    raw_output = span.outputs

    try:
        response_obj = raw_output if isinstance(raw_output, dict) else json.loads(raw_output)
    except Exception as e:
        print(f"Failed to parse span output: {e}")
        response_obj = {}

    # Extract disease text
    try:
        disease_text = (
            response_obj["candidates"][0]["content"]["parts"][0]["text"]
            if "candidates" in response_obj else None
        )
    except Exception as e:
        print(f"Failed to extract disease_text: {e}")
        disease_text = None

    # Flatten full span output
    response_row = flatten_json_to_row(response_obj, extra_cols={
        "trace_id": trace_id,
        "disease_name": disease_name,
        "disease_text": disease_text,
        "total_running_time": total_running_time
    })

    response_row = {k: str(v) if v is not None else None for k, v in response_row.items()}
    response_schema = StructType([StructField(k, StringType(), True) for k in response_row.keys()])
    df_response = spark.createDataFrame([response_row], schema=response_schema)
    df_response.write.mode("append").saveAsTable(response_tbl)

    print(f"Saved request to {request_tbl} and response to {response_tbl} for trace ID: {trace_id}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Gemini to generate text

# COMMAND ----------


from mlflow.entities import SpanType
import time
import json

# Get already processed diseases
processed_df = spark.sql("select disease_name from wei_lab_sander_mlflow.llm_gemini_response")
processed_diseases = set([row['disease_name'] for row in processed_df.collect()])


mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/gemini-demo")

client = genai.Client(api_key=gemini_api)
# grounding tool（Google Search）
grounding_tool = types.Tool(
    google_search=types.GoogleSearch()
)

generation_config = types.GenerateContentConfig(
    tools=[grounding_tool]
)

# === Traced Functions ===

@mlflow.trace
def build_prompt(disease):
    return prompt_template_1.format(disease=disease)

@mlflow.trace
def create_chat():
    return  client.chats.create(model="gemini-2.5-pro")

@mlflow.trace
def send_persona_message(chat):
    chat.send_message("You are a rare disease expert.")

@mlflow.trace
def send_disease_query(chat, prompt):
    return chat.send_message(prompt, config=generation_config)

@mlflow.trace
def parse_response(response):
    if response is None:        return ''
    content = response.text.strip()

    mlflow.set_tag("response_length", len(content))
    return content

@mlflow.trace(span_type=SpanType.CHAIN)
def run_for_disease(disease):
    mlflow.set_tag("disease", disease)
    prompt = build_prompt(disease)
    chat = create_chat()
    send_persona_message(chat)
    response = send_disease_query(chat, prompt)
    content = parse_response(response)
    return content

# === Loop ===

# diseases = [ 'Osteomyelitis',
#     'Anemia of Chronic Disease'
  
#   ] #test

for disease in diseases:
    if disease in processed_diseases:
       print(f"Skipping disease: {disease}")
       continue
    
    start_time = time.time()

    print(f"\nProcessing disease: {disease}")
    content_clean = run_for_disease(disease)
    print(f"  Response received (length: {len(content_clean)})")
    
    end_time = time.time()
    total_running_time = round(end_time - start_time, 2)

    time.sleep(20)
    save_gemini_trace(disease,total_running_time)



# COMMAND ----------

# MAGIC %md
# MAGIC ### Output Review

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT * FROM wei_lab_sander_mlflow.llm_gemini_response
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select *  from wei_lab_sander_mlflow.llm_gemini_request