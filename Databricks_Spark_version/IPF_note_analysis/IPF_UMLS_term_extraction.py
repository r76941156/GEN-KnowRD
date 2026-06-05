# Databricks notebook source
# MAGIC %pip install scispacy
# MAGIC %pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_scibert-0.5.4.tar.gz
# MAGIC %pip install tiktoken

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
    "threshold": 0.9,
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
# MAGIC ### Main UMLS terms parsed functions

# COMMAND ----------

def link_umls_entities(nlp_model, text, note_id, note_date, person_id, linker, max_tokens=512):
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

                        

                        results.append({
                            "person_id": person_id,
                            "note_id": note_id,
                            "note_date": note_date,
                            "chunk_index": chunk_index,
                            "UMLS_name": concept.canonical_name,
                            "CUI": cui,
                            "similarity": score,
                            "semantic_type": concept.types,
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
        print(f"Skipping note {note_id} due to parsing error: {e}")
        return pd.DataFrame()


# COMMAND ----------

# Load clinical notes
print("📥 Loading clinical notes...")
df_summary = spark.table("wei_lab_sander.non_ipf_control_notes_grouped").toPandas()
#wei_lab_sander.ipf_control_notes_grouped_171

# Choose multiple groups
selected_groups = ["group_3","group_4"]
group_df = df_summary[df_summary["group_id"].isin(selected_groups)]


# Process in batches of 10 person_ids
person_ids = group_df["person_id"].unique()
batch_size = 10

for batch_idx in range(0, len(person_ids), batch_size):
    batch_pids = person_ids[batch_idx: batch_idx + batch_size]
    batch_df = group_df[group_df["person_id"].isin(batch_pids)]
    
    df_linked_all = []

    print(f"\n🔄 Processing batch {batch_idx // batch_size + 1} with {len(batch_pids)} person_ids...")

    for i, row in enumerate(batch_df.itertuples(), 1):
        person_id = row.person_id
        note_id = row.note_id
        note_date = row.note_date
        text = row.note_text

        df_linked = link_umls_entities(
            nlp_model=nlp_umls,
            text=text,
            note_id=note_id,
            note_date=note_date,
            person_id=person_id,
            linker=umls_linker
        )

        if not df_linked.empty:
            df_linked_all.append(df_linked)
        else:
            print(f"⚠️  No UMLS concepts matched in note_id: {note_id}")

    if df_linked_all:
        df_final = pd.concat(df_linked_all, ignore_index=True)
        spark_df = spark.createDataFrame(df_final)

        output_table_name = "wei_lab_sander.non_ipf_umls_matched_results_2"
        print(f"💾 Saving batch {batch_idx // batch_size + 1} to Spark table...")

        if not spark.catalog.tableExists(output_table_name):
            spark_df.write.mode("overwrite").saveAsTable(output_table_name)
        else:
            spark_df.write.mode("append").saveAsTable(output_table_name)

        print("✅ Batch saved.")
    else:
        print("⚠️ No matched UMLS concepts found in this batch.")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Check the UMLS extraction output

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*),count(distinct person_id) from wei_lab_sander.non_ipf_umls_matched_results_2
# MAGIC