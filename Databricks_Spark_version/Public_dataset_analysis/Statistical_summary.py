# Databricks notebook source
# MAGIC %md
# MAGIC ### Running time

# COMMAND ----------

from pyspark.sql import functions as F

# =========================================================
# Config
# =========================================================
# 4 Models
models = ["claude", "o3", "gemini", "ds"]

# Table mapping rule:
#   o3 -> wei_lab_sander_mlflow.llm_o3_response_final
#   others -> wei_lab_sander_mlflow.llm_{model}_response
def table_for(model: str) -> str:
    return (
        "wei_lab_sander_mlflow.llm_o3_response_final"
        if model == "o3"
        else f"wei_lab_sander_mlflow.llm_{model}_response"
    )

# =========================================================
# Helper: robust percentiles in Spark
# =========================================================
# Spark/Databricks safest: percentile_approx(col, p, accuracy)
ACCURACY = 10000

def build_stats_df(model: str):
    tbl = table_for(model)

    df = (
        spark.table(tbl)
        .where(F.col("total_running_time").isNotNull())
        .select(F.col("total_running_time").cast("double").alias("total_running_time"))
    )

    return df.agg(
        F.lit(model).alias("model"),

        F.round(F.avg("total_running_time"), 2).alias("avg_total_running_time"),
        F.round(F.stddev_pop("total_running_time"), 2).alias("std_total_running_time"),

        F.round(F.expr(f"percentile_approx(total_running_time, 0.25, {ACCURACY})"), 2).alias("q1_total_running_time"),
        F.round(F.expr(f"percentile_approx(total_running_time, 0.50, {ACCURACY})"), 2).alias("median_total_running_time"),
        F.round(F.expr(f"percentile_approx(total_running_time, 0.75, {ACCURACY})"), 2).alias("q3_total_running_time"),

        F.round(
            F.expr(f"percentile_approx(total_running_time, 0.75, {ACCURACY})")
            - F.expr(f"percentile_approx(total_running_time, 0.25, {ACCURACY})"),
            2
        ).alias("iqr_total_running_time"),

        F.count(F.lit(1)).alias("record_count")
    )

# =========================================================
# Loop + union results
# =========================================================
result = None
for m in models:
    one = build_stats_df(m)
    result = one if result is None else result.unionByName(one)

# Optional: pretty order
result = result.select(
    "model",
    "avg_total_running_time",
    "std_total_running_time",
    "q1_total_running_time",
    "median_total_running_time",
    "q3_total_running_time",
    "iqr_total_running_time",
    "record_count",
).orderBy("model")

display(result)  




# COMMAND ----------

# MAGIC %md
# MAGIC ## Token Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ### o3: wei_lab_sander_mlflow.llm_o3_response_final table

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH token_stats AS (
# MAGIC   SELECT 
# MAGIC     -- ======================
# MAGIC     -- INPUT TOKENS
# MAGIC     -- ======================
# MAGIC     AVG(CAST(get_json_object(usage, '$.input_tokens') AS INT)) AS avg_input_tokens,
# MAGIC     STDDEV(CAST(get_json_object(usage, '$.input_tokens') AS INT)) AS std_input_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.input_tokens') AS INT), 0.25, 10000) AS q1_input_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.input_tokens') AS INT), 0.50, 10000) AS median_input_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.input_tokens') AS INT), 0.75, 10000) AS q3_input_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- OUTPUT TOKENS
# MAGIC     -- ======================
# MAGIC     AVG(CAST(get_json_object(usage, '$.output_tokens') AS INT)) AS avg_output_tokens,
# MAGIC     STDDEV(CAST(get_json_object(usage, '$.output_tokens') AS INT)) AS std_output_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.output_tokens') AS INT), 0.25, 10000) AS q1_output_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.output_tokens') AS INT), 0.50, 10000) AS median_output_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.output_tokens') AS INT), 0.75, 10000) AS q3_output_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- REASONING TOKENS
# MAGIC     -- ======================
# MAGIC     AVG(CAST(get_json_object(usage, '$.output_tokens_details.reasoning_tokens') AS INT)) AS avg_reasoning_tokens,
# MAGIC     STDDEV(CAST(get_json_object(usage, '$.output_tokens_details.reasoning_tokens') AS INT)) AS std_reasoning_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.output_tokens_details.reasoning_tokens') AS INT), 0.25, 10000) AS q1_reasoning_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.output_tokens_details.reasoning_tokens') AS INT), 0.50, 10000) AS median_reasoning_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.output_tokens_details.reasoning_tokens') AS INT), 0.75, 10000) AS q3_reasoning_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- CACHED TOKENS
# MAGIC     -- ======================
# MAGIC     AVG(CAST(get_json_object(usage, '$.input_tokens_details.cached_tokens') AS INT)) AS avg_cached_tokens,
# MAGIC     STDDEV(CAST(get_json_object(usage, '$.input_tokens_details.cached_tokens') AS INT)) AS std_cached_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.input_tokens_details.cached_tokens') AS INT), 0.25, 10000) AS q1_cached_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.input_tokens_details.cached_tokens') AS INT), 0.50, 10000) AS median_cached_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.input_tokens_details.cached_tokens') AS INT), 0.75, 10000) AS q3_cached_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- TOTAL TOKENS
# MAGIC     -- ======================
# MAGIC     AVG(CAST(get_json_object(usage, '$.total_tokens') AS INT)) AS avg_total_tokens,
# MAGIC     STDDEV(CAST(get_json_object(usage, '$.total_tokens') AS INT)) AS std_total_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.total_tokens') AS INT), 0.25, 10000) AS q1_total_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.total_tokens') AS INT), 0.50, 10000) AS median_total_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.total_tokens') AS INT), 0.75, 10000) AS q3_total_tokens
# MAGIC
# MAGIC   FROM wei_lab_sander_mlflow.llm_o3_response_final
# MAGIC   WHERE usage IS NOT NULL
# MAGIC )
# MAGIC
# MAGIC SELECT 
# MAGIC   -- ======================
# MAGIC   -- INPUT TOKENS
# MAGIC   -- ======================
# MAGIC   ROUND(avg_input_tokens, 2) AS avg_input_tokens,
# MAGIC   ROUND(std_input_tokens, 2) AS std_input_tokens,
# MAGIC   ROUND(q1_input_tokens, 2) AS q1_input_tokens,
# MAGIC   ROUND(median_input_tokens, 2) AS median_input_tokens,
# MAGIC   ROUND(q3_input_tokens, 2) AS q3_input_tokens,
# MAGIC   ROUND(q3_input_tokens - q1_input_tokens, 2) AS iqr_input_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- OUTPUT TOKENS
# MAGIC   -- ======================
# MAGIC   ROUND(avg_output_tokens, 2) AS avg_output_tokens,
# MAGIC   ROUND(std_output_tokens, 2) AS std_output_tokens,
# MAGIC   ROUND(q1_output_tokens, 2) AS q1_output_tokens,
# MAGIC   ROUND(median_output_tokens, 2) AS median_output_tokens,
# MAGIC   ROUND(q3_output_tokens, 2) AS q3_output_tokens,
# MAGIC   ROUND(q3_output_tokens - q1_output_tokens, 2) AS iqr_output_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- REASONING TOKENS
# MAGIC   -- ======================
# MAGIC   ROUND(avg_reasoning_tokens, 2) AS avg_reasoning_tokens,
# MAGIC   ROUND(std_reasoning_tokens, 2) AS std_reasoning_tokens,
# MAGIC   ROUND(q1_reasoning_tokens, 2) AS q1_reasoning_tokens,
# MAGIC   ROUND(median_reasoning_tokens, 2) AS median_reasoning_tokens,
# MAGIC   ROUND(q3_reasoning_tokens, 2) AS q3_reasoning_tokens,
# MAGIC   ROUND(q3_reasoning_tokens - q1_reasoning_tokens, 2) AS iqr_reasoning_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- CACHED TOKENS
# MAGIC   -- ======================
# MAGIC   ROUND(avg_cached_tokens, 2) AS avg_cached_tokens,
# MAGIC   ROUND(std_cached_tokens, 2) AS std_cached_tokens,
# MAGIC   ROUND(q1_cached_tokens, 2) AS q1_cached_tokens,
# MAGIC   ROUND(median_cached_tokens, 2) AS median_cached_tokens,
# MAGIC   ROUND(q3_cached_tokens, 2) AS q3_cached_tokens,
# MAGIC   ROUND(q3_cached_tokens - q1_cached_tokens, 2) AS iqr_cached_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- TOTAL TOKENS
# MAGIC   -- ======================
# MAGIC   ROUND(avg_total_tokens, 2) AS avg_total_tokens,
# MAGIC   ROUND(std_total_tokens, 2) AS std_total_tokens,
# MAGIC   ROUND(q1_total_tokens, 2) AS q1_total_tokens,
# MAGIC   ROUND(median_total_tokens, 2) AS median_total_tokens,
# MAGIC   ROUND(q3_total_tokens, 2) AS q3_total_tokens,
# MAGIC   ROUND(q3_total_tokens - q1_total_tokens, 2) AS iqr_total_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- RATIOS (UNCHANGED)
# MAGIC   -- ======================
# MAGIC   ROUND(avg_reasoning_tokens / avg_output_tokens, 4) AS output_reasoning_ratio,
# MAGIC   ROUND(avg_cached_tokens / avg_input_tokens, 4) AS input_cached_ratio
# MAGIC
# MAGIC FROM token_stats;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### DS: wei_lab_sander_mlflow.llm_ds_response table

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC WITH token_stats AS (
# MAGIC   SELECT 
# MAGIC     -- ======================
# MAGIC     -- PROMPT TOKENS
# MAGIC     -- ======================
# MAGIC     AVG(CAST(get_json_object(usage, '$.prompt_tokens') AS INT))        AS avg_prompt_tokens,
# MAGIC     STDDEV(CAST(get_json_object(usage, '$.prompt_tokens') AS INT))     AS std_prompt_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.prompt_tokens') AS INT), 0.25, 10000) AS q1_prompt_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.prompt_tokens') AS INT), 0.50, 10000) AS median_prompt_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.prompt_tokens') AS INT), 0.75, 10000) AS q3_prompt_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- COMPLETION TOKENS
# MAGIC     -- ======================
# MAGIC     AVG(CAST(get_json_object(usage, '$.completion_tokens') AS INT))    AS avg_completion_tokens,
# MAGIC     STDDEV(CAST(get_json_object(usage, '$.completion_tokens') AS INT)) AS std_completion_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.completion_tokens') AS INT), 0.25, 10000) AS q1_completion_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.completion_tokens') AS INT), 0.50, 10000) AS median_completion_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.completion_tokens') AS INT), 0.75, 10000) AS q3_completion_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- REASONING TOKENS
# MAGIC     -- ======================
# MAGIC     AVG(CAST(get_json_object(usage, '$.completion_tokens_details.reasoning_tokens') AS INT)) AS avg_reasoning_tokens,
# MAGIC     STDDEV(CAST(get_json_object(usage, '$.completion_tokens_details.reasoning_tokens') AS INT)) AS std_reasoning_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.completion_tokens_details.reasoning_tokens') AS INT), 0.25, 10000) AS q1_reasoning_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.completion_tokens_details.reasoning_tokens') AS INT), 0.50, 10000) AS median_reasoning_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.completion_tokens_details.reasoning_tokens') AS INT), 0.75, 10000) AS q3_reasoning_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- TOTAL TOKENS
# MAGIC     -- ======================
# MAGIC     AVG(CAST(get_json_object(usage, '$.total_tokens') AS INT))         AS avg_total_tokens,
# MAGIC     STDDEV(CAST(get_json_object(usage, '$.total_tokens') AS INT))      AS std_total_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.total_tokens') AS INT), 0.25, 10000) AS q1_total_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.total_tokens') AS INT), 0.50, 10000) AS median_total_tokens,
# MAGIC     percentile_approx(CAST(get_json_object(usage, '$.total_tokens') AS INT), 0.75, 10000) AS q3_total_tokens
# MAGIC
# MAGIC   FROM wei_lab_sander_mlflow.llm_ds_response
# MAGIC   WHERE usage IS NOT NULL
# MAGIC )
# MAGIC
# MAGIC SELECT 
# MAGIC     -- ======================
# MAGIC     -- PROMPT TOKENS
# MAGIC     -- ======================
# MAGIC     ROUND(avg_prompt_tokens, 2)        AS avg_prompt_tokens,
# MAGIC     ROUND(std_prompt_tokens, 2)        AS std_prompt_tokens,
# MAGIC     ROUND(q1_prompt_tokens, 2)         AS q1_prompt_tokens,
# MAGIC     ROUND(median_prompt_tokens, 2)     AS median_prompt_tokens,
# MAGIC     ROUND(q3_prompt_tokens, 2)         AS q3_prompt_tokens,
# MAGIC     ROUND(q3_prompt_tokens - q1_prompt_tokens, 2) AS iqr_prompt_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- COMPLETION TOKENS
# MAGIC     -- ======================
# MAGIC     ROUND(avg_completion_tokens, 2)    AS avg_completion_tokens,
# MAGIC     ROUND(std_completion_tokens, 2)    AS std_completion_tokens,
# MAGIC     ROUND(q1_completion_tokens, 2)     AS q1_completion_tokens,
# MAGIC     ROUND(median_completion_tokens, 2) AS median_completion_tokens,
# MAGIC     ROUND(q3_completion_tokens, 2)     AS q3_completion_tokens,
# MAGIC     ROUND(q3_completion_tokens - q1_completion_tokens, 2) AS iqr_completion_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- REASONING TOKENS
# MAGIC     -- ======================
# MAGIC     ROUND(avg_reasoning_tokens, 2)     AS avg_reasoning_tokens,
# MAGIC     ROUND(std_reasoning_tokens, 2)     AS std_reasoning_tokens,
# MAGIC     ROUND(q1_reasoning_tokens, 2)      AS q1_reasoning_tokens,
# MAGIC     ROUND(median_reasoning_tokens, 2)  AS median_reasoning_tokens,
# MAGIC     ROUND(q3_reasoning_tokens, 2)      AS q3_reasoning_tokens,
# MAGIC     ROUND(q3_reasoning_tokens - q1_reasoning_tokens, 2) AS iqr_reasoning_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- TOTAL TOKENS
# MAGIC     -- ======================
# MAGIC     ROUND(avg_total_tokens, 2)         AS avg_total_tokens,
# MAGIC     ROUND(std_total_tokens, 2)         AS std_total_tokens,
# MAGIC     ROUND(q1_total_tokens, 2)          AS q1_total_tokens,
# MAGIC     ROUND(median_total_tokens, 2)      AS median_total_tokens,
# MAGIC     ROUND(q3_total_tokens, 2)          AS q3_total_tokens,
# MAGIC     ROUND(q3_total_tokens - q1_total_tokens, 2) AS iqr_total_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- RATIO (UNCHANGED)
# MAGIC     -- ======================
# MAGIC     ROUND(avg_reasoning_tokens / avg_completion_tokens, 2) AS output_reasoning_ratio
# MAGIC
# MAGIC FROM token_stats;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Claude: wei_lab_sander_mlflow.llm_claude_response table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   'SUM' AS disease_name,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- INPUT TOKENS
# MAGIC   -- ======================
# MAGIC   CAST(ROUND(SUM(input_tokens), 2) AS STRING) AS input_tokens,
# MAGIC   CAST(ROUND(AVG(input_tokens), 2) AS STRING) AS avg_input_tokens,
# MAGIC   CAST(ROUND(STDDEV(input_tokens), 2) AS STRING) AS std_input_tokens,
# MAGIC   CAST(ROUND(percentile_approx(input_tokens, 0.25, 10000), 2) AS STRING) AS q1_input_tokens,
# MAGIC   CAST(ROUND(percentile_approx(input_tokens, 0.50, 10000), 2) AS STRING) AS median_input_tokens,
# MAGIC   CAST(ROUND(percentile_approx(input_tokens, 0.75, 10000), 2) AS STRING) AS q3_input_tokens,
# MAGIC   CAST(ROUND(
# MAGIC       percentile_approx(input_tokens, 0.75, 10000)
# MAGIC     - percentile_approx(input_tokens, 0.25, 10000), 2
# MAGIC   ) AS STRING) AS iqr_input_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- OUTPUT TOKENS
# MAGIC   -- ======================
# MAGIC   CAST(ROUND(SUM(output_tokens), 2) AS STRING) AS output_tokens,
# MAGIC   CAST(ROUND(AVG(output_tokens), 2) AS STRING) AS avg_output_tokens,
# MAGIC   CAST(ROUND(STDDEV(output_tokens), 2) AS STRING) AS std_output_tokens,
# MAGIC   CAST(ROUND(percentile_approx(output_tokens, 0.25, 10000), 2) AS STRING) AS q1_output_tokens,
# MAGIC   CAST(ROUND(percentile_approx(output_tokens, 0.50, 10000), 2) AS STRING) AS median_output_tokens,
# MAGIC   CAST(ROUND(percentile_approx(output_tokens, 0.75, 10000), 2) AS STRING) AS q3_output_tokens,
# MAGIC   CAST(ROUND(
# MAGIC       percentile_approx(output_tokens, 0.75, 10000)
# MAGIC     - percentile_approx(output_tokens, 0.25, 10000), 2
# MAGIC   ) AS STRING) AS iqr_output_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- WEB SEARCH REQUESTS
# MAGIC   -- ======================
# MAGIC   CAST(ROUND(SUM(web_search_requests), 2) AS STRING) AS web_search_requests,
# MAGIC   CAST(ROUND(AVG(web_search_requests), 2) AS STRING) AS avg_web_search_requests,
# MAGIC   CAST(ROUND(STDDEV(web_search_requests), 2) AS STRING) AS std_web_search_requests,
# MAGIC   CAST(ROUND(percentile_approx(web_search_requests, 0.25, 10000), 2) AS STRING) AS q1_web_search_requests,
# MAGIC   CAST(ROUND(percentile_approx(web_search_requests, 0.50, 10000), 2) AS STRING) AS median_web_search_requests,
# MAGIC   CAST(ROUND(percentile_approx(web_search_requests, 0.75, 10000), 2) AS STRING) AS q3_web_search_requests,
# MAGIC   CAST(ROUND(
# MAGIC       percentile_approx(web_search_requests, 0.75, 10000)
# MAGIC     - percentile_approx(web_search_requests, 0.25, 10000), 2
# MAGIC   ) AS STRING) AS iqr_web_search_requests,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- COSTS
# MAGIC   -- ======================
# MAGIC   CAST(ROUND(SUM(input_cost), 4) AS STRING) AS total_input_cost_usd,
# MAGIC   CAST(ROUND(SUM(output_cost), 4) AS STRING) AS total_output_cost_usd,
# MAGIC   CAST(ROUND(SUM(input_cost + output_cost), 4) AS STRING) AS total_cost_usd
# MAGIC
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     CAST(get_json_object(usage, '$.input_tokens') AS DOUBLE) AS input_tokens,
# MAGIC     CAST(get_json_object(usage, '$.output_tokens') AS DOUBLE) AS output_tokens,
# MAGIC     CAST(get_json_object(get_json_object(usage, '$.server_tool_use'), '$.web_search_requests') AS DOUBLE) AS web_search_requests,
# MAGIC
# MAGIC     (CAST(get_json_object(usage, '$.input_tokens') AS DOUBLE) / 1000000) * 3  AS input_cost,
# MAGIC     (CAST(get_json_object(usage, '$.output_tokens') AS DOUBLE) / 1000000) * 15 AS output_cost
# MAGIC
# MAGIC   FROM wei_lab_sander_mlflow.llm_claude_response
# MAGIC   WHERE usage IS NOT NULL
# MAGIC ) t;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gemini:  wei_lab_sander_mlflow.llm_gemini_response

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH token_info AS (
# MAGIC   SELECT
# MAGIC     CAST(get_json_object(usage_metadata, '$.prompt_token_count') AS INT)        AS input_tokens,
# MAGIC     CAST(get_json_object(usage_metadata, '$.candidates_token_count') AS INT)    AS output_tokens,
# MAGIC     CAST(get_json_object(usage_metadata, '$.thoughts_token_count') AS INT)      AS thoughts_tokens,
# MAGIC     CAST(get_json_object(usage_metadata, '$.tool_use_prompt_token_count') AS INT) AS tool_use_prompt_tokens
# MAGIC   FROM wei_lab_sander_mlflow.llm_gemini_response
# MAGIC   WHERE usage_metadata IS NOT NULL
# MAGIC ),
# MAGIC
# MAGIC token_costs AS (
# MAGIC   SELECT *,
# MAGIC     CASE
# MAGIC       WHEN input_tokens > 200000 THEN input_tokens * 2.50 / 1000000
# MAGIC       ELSE input_tokens * 1.25 / 1000000
# MAGIC     END AS input_cost,
# MAGIC
# MAGIC     CASE
# MAGIC       WHEN input_tokens > 200000 THEN output_tokens * 15.00 / 1000000
# MAGIC       ELSE output_tokens * 10.00 / 1000000
# MAGIC     END AS output_cost
# MAGIC   FROM token_info
# MAGIC ),
# MAGIC
# MAGIC token_stats AS (
# MAGIC   SELECT
# MAGIC     -- ======================
# MAGIC     -- INPUT TOKENS
# MAGIC     -- ======================
# MAGIC     SUM(input_tokens) AS total_input_tokens,
# MAGIC     AVG(input_tokens) AS avg_input_tokens,
# MAGIC     STDDEV(input_tokens) AS std_input_tokens,
# MAGIC     percentile_approx(input_tokens, 0.25, 10000) AS q1_input_tokens,
# MAGIC     percentile_approx(input_tokens, 0.50, 10000) AS median_input_tokens,
# MAGIC     percentile_approx(input_tokens, 0.75, 10000) AS q3_input_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- OUTPUT TOKENS
# MAGIC     -- ======================
# MAGIC     SUM(output_tokens) AS total_output_tokens,
# MAGIC     AVG(output_tokens) AS avg_output_tokens,
# MAGIC     STDDEV(output_tokens) AS std_output_tokens,
# MAGIC     percentile_approx(output_tokens, 0.25, 10000) AS q1_output_tokens,
# MAGIC     percentile_approx(output_tokens, 0.50, 10000) AS median_output_tokens,
# MAGIC     percentile_approx(output_tokens, 0.75, 10000) AS q3_output_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- THOUGHTS TOKENS
# MAGIC     -- ======================
# MAGIC     SUM(thoughts_tokens) AS total_thoughts_tokens,
# MAGIC     AVG(thoughts_tokens) AS avg_thoughts_tokens,
# MAGIC     STDDEV(thoughts_tokens) AS std_thoughts_tokens,
# MAGIC     percentile_approx(thoughts_tokens, 0.25, 10000) AS q1_thoughts_tokens,
# MAGIC     percentile_approx(thoughts_tokens, 0.50, 10000) AS median_thoughts_tokens,
# MAGIC     percentile_approx(thoughts_tokens, 0.75, 10000) AS q3_thoughts_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- TOOL-USE PROMPT TOKENS
# MAGIC     -- ======================
# MAGIC     AVG(tool_use_prompt_tokens) AS avg_tool_use_prompt_tokens,
# MAGIC     STDDEV(tool_use_prompt_tokens) AS std_tool_use_prompt_tokens,
# MAGIC     percentile_approx(tool_use_prompt_tokens, 0.25, 10000) AS q1_tool_use_prompt_tokens,
# MAGIC     percentile_approx(tool_use_prompt_tokens, 0.50, 10000) AS median_tool_use_prompt_tokens,
# MAGIC     percentile_approx(tool_use_prompt_tokens, 0.75, 10000) AS q3_tool_use_prompt_tokens,
# MAGIC
# MAGIC     -- ======================
# MAGIC     -- COSTS (UNCHANGED)
# MAGIC     -- ======================
# MAGIC     SUM(input_cost) AS total_input_cost_usd,
# MAGIC     SUM(output_cost) AS total_output_cost_usd,
# MAGIC     SUM(input_cost + output_cost) AS total_cost_usd
# MAGIC
# MAGIC   FROM token_costs
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC   -- ======================
# MAGIC   -- INPUT TOKENS
# MAGIC   -- ======================
# MAGIC   total_input_tokens,
# MAGIC   ROUND(avg_input_tokens, 2) AS avg_input_tokens,
# MAGIC   ROUND(std_input_tokens, 2) AS std_input_tokens,
# MAGIC   ROUND(q1_input_tokens, 2) AS q1_input_tokens,
# MAGIC   ROUND(median_input_tokens, 2) AS median_input_tokens,
# MAGIC   ROUND(q3_input_tokens, 2) AS q3_input_tokens,
# MAGIC   ROUND(q3_input_tokens - q1_input_tokens, 2) AS iqr_input_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- OUTPUT TOKENS
# MAGIC   -- ======================
# MAGIC   total_output_tokens,
# MAGIC   ROUND(avg_output_tokens, 2) AS avg_output_tokens,
# MAGIC   ROUND(std_output_tokens, 2) AS std_output_tokens,
# MAGIC   ROUND(q1_output_tokens, 2) AS q1_output_tokens,
# MAGIC   ROUND(median_output_tokens, 2) AS median_output_tokens,
# MAGIC   ROUND(q3_output_tokens, 2) AS q3_output_tokens,
# MAGIC   ROUND(q3_output_tokens - q1_output_tokens, 2) AS iqr_output_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- THOUGHTS TOKENS
# MAGIC   -- ======================
# MAGIC   total_thoughts_tokens,
# MAGIC   ROUND(avg_thoughts_tokens, 2) AS avg_thoughts_tokens,
# MAGIC   ROUND(std_thoughts_tokens, 2) AS std_thoughts_tokens,
# MAGIC   ROUND(q1_thoughts_tokens, 2) AS q1_thoughts_tokens,
# MAGIC   ROUND(median_thoughts_tokens, 2) AS median_thoughts_tokens,
# MAGIC   ROUND(q3_thoughts_tokens, 2) AS q3_thoughts_tokens,
# MAGIC   ROUND(q3_thoughts_tokens - q1_thoughts_tokens, 2) AS iqr_thoughts_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- TOOL-USE PROMPT TOKENS
# MAGIC   -- ======================
# MAGIC   ROUND(avg_tool_use_prompt_tokens, 2) AS avg_tool_use_prompt_tokens,
# MAGIC   ROUND(std_tool_use_prompt_tokens, 2) AS std_tool_use_prompt_tokens,
# MAGIC   ROUND(q1_tool_use_prompt_tokens, 2) AS q1_tool_use_prompt_tokens,
# MAGIC   ROUND(median_tool_use_prompt_tokens, 2) AS median_tool_use_prompt_tokens,
# MAGIC   ROUND(q3_tool_use_prompt_tokens, 2) AS q3_tool_use_prompt_tokens,
# MAGIC   ROUND(q3_tool_use_prompt_tokens - q1_tool_use_prompt_tokens, 2) AS iqr_tool_use_prompt_tokens,
# MAGIC
# MAGIC   -- ======================
# MAGIC   -- COSTS
# MAGIC   -- ======================
# MAGIC   ROUND(total_input_cost_usd, 4) AS total_input_cost_usd,
# MAGIC   ROUND(total_output_cost_usd, 4) AS total_output_cost_usd,
# MAGIC   ROUND(total_cost_usd, 4) AS total_cost_usd
# MAGIC
# MAGIC FROM token_stats;
# MAGIC
# MAGIC
# MAGIC --reasoning token ratio:790/2645~30%

# COMMAND ----------

# MAGIC %md
# MAGIC ## Citation summary

# COMMAND ----------

# MAGIC %md
# MAGIC ### o3

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH url_extraction AS (
# MAGIC   SELECT 
# MAGIC     disease_name,
# MAGIC     explode(
# MAGIC       regexp_extract_all(disease_text, '\\[([^\\]]+)\\]\\((https?://[^\\s)]+)', 0)
# MAGIC     ) AS matched_text
# MAGIC   FROM wei_lab_sander_mlflow.llm_o3_response_final
# MAGIC   WHERE disease_text LIKE '%](http%'
# MAGIC ),
# MAGIC parsed_urls AS (
# MAGIC   SELECT
# MAGIC     disease_name,
# MAGIC     regexp_extract(matched_text, '\\[([^\\]]+)\\]', 1) AS website,
# MAGIC     regexp_extract(matched_text, '\\((https?://[^\\s)]+)', 1) AS url
# MAGIC   FROM url_extraction
# MAGIC ),
# MAGIC
# MAGIC --top 10 websites 
# MAGIC
# MAGIC base AS (
# MAGIC     SELECT 
# MAGIC         website,
# MAGIC
# MAGIC         -- Category mapping
# MAGIC         CASE 
# MAGIC             WHEN website IN (
# MAGIC                 'nih.gov', 'ncbi.nlm.nih.gov', 'pubmed.ncbi.nlm.nih.gov',
# MAGIC                 'pmc.ncbi.nlm.nih.gov', 'medlineplus.gov'
# MAGIC             ) THEN 'Government / Official Databases'
# MAGIC
# MAGIC             WHEN website IN ('rarediseases.org', 'orpha.net')
# MAGIC                 THEN 'Rare Disease Databases'
# MAGIC
# MAGIC             WHEN website IN (
# MAGIC                 'medscape.com', 'emedicine.medscape.com', 
# MAGIC                 'clevelandclinic.org', 'my.clevelandclinic.org', 
# MAGIC                 'mayoclinic.org'
# MAGIC             ) THEN 'Clinical Medical Reference'
# MAGIC
# MAGIC             WHEN website IN (
# MAGIC                 'onlinelibrary.wiley.com', 'mdpi.com', 
# MAGIC                 'researchgate.net', 'sciencedirect.com'
# MAGIC             ) THEN 'Academic Journals / Research'
# MAGIC
# MAGIC             WHEN website IN ('wikipedia.org', 'en.wikipedia.org')
# MAGIC                 THEN 'General Encyclopedia'
# MAGIC
# MAGIC             WHEN website = 'youtube.com'
# MAGIC                 THEN 'Media Platforms'
# MAGIC
# MAGIC             ELSE 'Other / Unknown'
# MAGIC         END AS category,
# MAGIC
# MAGIC         disease_name,
# MAGIC         url
# MAGIC     FROM parsed_urls
# MAGIC ),
# MAGIC
# MAGIC -- (A) Compute disease_count per website
# MAGIC -- ranked AS (
# MAGIC --     SELECT 
# MAGIC --         website,
# MAGIC --         category,
# MAGIC --         COUNT(DISTINCT disease_name) AS disease_count
# MAGIC --     FROM base
# MAGIC --     GROUP BY website, category
# MAGIC -- ),
# MAGIC
# MAGIC -- (B) Select TOP 10 websites by disease_count
# MAGIC -- top10 AS (
# MAGIC --     SELECT website, category
# MAGIC --     FROM ranked
# MAGIC --     ORDER BY disease_count DESC
# MAGIC --     LIMIT 10
# MAGIC -- ),
# MAGIC
# MAGIC -- (C) Filter main table to only these top 10 websites
# MAGIC -- filtered AS (
# MAGIC --     SELECT b.*
# MAGIC --     FROM base b
# MAGIC --     JOIN top10 t USING (website)
# MAGIC -- )
# MAGIC
# MAGIC -- (D) Final per-category UNIQUE disease count + citation %
# MAGIC -- SELECT
# MAGIC --     category,
# MAGIC --     COUNT(DISTINCT disease_name) AS unique_disease_count,
# MAGIC --     ROUND(COUNT(url) * 100.0 / 36214, 2) AS citation_percentage
# MAGIC -- FROM filtered
# MAGIC -- GROUP BY category
# MAGIC -- ORDER BY unique_disease_count DESC
# MAGIC
# MAGIC
# MAGIC --total diseases and total citations
# MAGIC
# MAGIC --  SELECT count(distinct disease_name) as disase_count,count(url) as total_citation
# MAGIC --   FROM parsed_urls
# MAGIC
# MAGIC --avg and std ciations
# MAGIC citation_counts AS (
# MAGIC   SELECT disease_name, count(url) as citation_count
# MAGIC   FROM parsed_urls
# MAGIC   GROUP BY disease_name
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC   SUM(citation_count) AS total_citations,
# MAGIC   round(AVG(citation_count),2) AS avg_citation_count,
# MAGIC   round(STDDEV(citation_count),2) AS std_citation_count,
# MAGIC   count(distinct disease_name) AS total_diseases
# MAGIC FROM citation_counts
# MAGIC
# MAGIC -- find what disease did not have http-based citions
# MAGIC -- select * from wei_lab_sander_mlflow.llm_o3_response_final where disease_name not in (
# MAGIC --   SELECT DISTINCT disease_name
# MAGIC --   FROM parsed_urls)
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gemini citations

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH grounding_chunks AS (
# MAGIC   SELECT 
# MAGIC     disease_name,
# MAGIC     explode(
# MAGIC       from_json(
# MAGIC         get_json_object(Candidates, '$[0].grounding_metadata.grounding_chunks'),
# MAGIC         'array<struct<web:struct<uri:string,title:string>>>'
# MAGIC       )
# MAGIC     ) AS chunk
# MAGIC   FROM wei_lab_sander_mlflow.llm_gemini_response
# MAGIC   WHERE get_json_object(Candidates, '$[0].grounding_metadata.grounding_chunks') IS NOT NULL
# MAGIC ),
# MAGIC
# MAGIC disease_url_counts AS (
# MAGIC   SELECT 
# MAGIC     disease_name,
# MAGIC     count(*) as total
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       disease_name,
# MAGIC       chunk.web.title AS url_title,
# MAGIC       chunk.web.uri AS url
# MAGIC     FROM grounding_chunks
# MAGIC     WHERE chunk.web.uri IS NOT NULL
# MAGIC   )
# MAGIC   GROUP BY disease_name
# MAGIC )
# MAGIC
# MAGIC --avg/std
# MAGIC
# MAGIC SELECT
# MAGIC  count(distinct disease_name) as total_diseases,
# MAGIC  sum(total) as total_citations,
# MAGIC   ROUND(AVG(total), 2)    AS avg_citation_count,
# MAGIC   ROUND(STDDEV(total), 2) AS std_citation_count
# MAGIC FROM disease_url_counts
# MAGIC
# MAGIC
# MAGIC ---find diseases without citions
# MAGIC
# MAGIC -- select candidates FROM wei_lab_sander_mlflow.llm_gemini_response where disease_name not  in (
# MAGIC -- select disease_name from 
# MAGIC -- disease_url_counts )
# MAGIC
# MAGIC
# MAGIC -- base AS (
# MAGIC --     SELECT 
# MAGIC --         chunk.web.title AS website,
# MAGIC --         disease_name,
# MAGIC --         chunk.web.uri AS url,
# MAGIC
# MAGIC --         -- Category mapping (clean version)
# MAGIC --         CASE
# MAGIC --             WHEN chunk.web.title IN (
# MAGIC --                 'nih.gov',
# MAGIC --                 'ncbi.nlm.nih.gov',
# MAGIC --                 'pubmed.ncbi.nlm.nih.gov',
# MAGIC --                 'pmc.ncbi.nlm.nih.gov',
# MAGIC --                 'medlineplus.gov'
# MAGIC --             ) THEN 'Government / Official Databases'
# MAGIC
# MAGIC --             WHEN chunk.web.title IN (
# MAGIC --                 'rarediseases.org',
# MAGIC --                 'orpha.net'
# MAGIC --             ) THEN 'Rare Disease Databases'
# MAGIC
# MAGIC --             WHEN chunk.web.title IN (
# MAGIC --                 'medscape.com',
# MAGIC --                 'emedicine.medscape.com',
# MAGIC --                 'clevelandclinic.org',
# MAGIC --                 'my.clevelandclinic.org',
# MAGIC --                 'mayoclinic.org'
# MAGIC --             ) THEN 'Clinical Medical Reference'
# MAGIC
# MAGIC --             WHEN chunk.web.title IN (
# MAGIC --                 'onlinelibrary.wiley.com',
# MAGIC --                 'mdpi.com',
# MAGIC --                 'researchgate.net',
# MAGIC --                 'sciencedirect.com'
# MAGIC --             ) THEN 'Academic Journals / Research'
# MAGIC
# MAGIC --             WHEN chunk.web.title IN (
# MAGIC --                 'wikipedia.org',
# MAGIC --                 'en.wikipedia.org'
# MAGIC --             ) THEN 'General Encyclopedia'
# MAGIC
# MAGIC --             WHEN chunk.web.title = 'youtube.com'
# MAGIC --                 THEN 'Media Platforms'
# MAGIC
# MAGIC --             ELSE 'Other / Unknown'
# MAGIC --         END AS category
# MAGIC --     FROM grounding_chunks
# MAGIC --     WHERE chunk.web.uri IS NOT NULL
# MAGIC -- ),
# MAGIC
# MAGIC -- (1) Compute disease_count per website
# MAGIC -- ranked AS (
# MAGIC --     SELECT
# MAGIC --         website,
# MAGIC --         category,
# MAGIC --         COUNT(DISTINCT disease_name) AS disease_count
# MAGIC --     FROM base
# MAGIC --     GROUP BY website, category
# MAGIC -- ),
# MAGIC
# MAGIC -- (2) Select TOP 10 websites by disease_count
# MAGIC -- top10 AS (
# MAGIC --     SELECT website, category
# MAGIC --     FROM ranked
# MAGIC --     ORDER BY disease_count DESC
# MAGIC --     LIMIT 10
# MAGIC -- ),
# MAGIC
# MAGIC -- (3) Filter to only top 10 websites
# MAGIC -- filtered AS (
# MAGIC --     SELECT b.*
# MAGIC --     FROM base b
# MAGIC --     JOIN top10 t USING (website)
# MAGIC -- )
# MAGIC
# MAGIC -- (4) Final aggregation: true unique diseases + citation %
# MAGIC -- SELECT
# MAGIC --     category,
# MAGIC --     COUNT(DISTINCT disease_name) AS unique_disease_count,
# MAGIC --     ROUND(COUNT(url) / 27079.0 * 100, 2) AS citation_percentage
# MAGIC -- FROM filtered
# MAGIC -- GROUP BY category
# MAGIC -- ORDER BY unique_disease_count DESC;
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Claude citations

# COMMAND ----------

import re
from pyspark.sql.functions import udf, col, explode
from pyspark.sql.types import ArrayType, StructType, StructField, StringType

# === Step 1: Define regex extraction logic ===
def extract_title_type_url(json_str):
    if not json_str:
        return []
    
    # Convert to string if already dict
    if isinstance(json_str, dict):
        json_str = str(json_str)
    
    try:
        matches = re.findall(
            r'"title"\s*:\s*"([^"]+)"[^}]*?"type"\s*:\s*"([^"]+)"[^}]*?"url"\s*:\s*"([^"]+)"',
            json_str
        )
        return [{"title": m[0], "type": m[1], "url": m[2]} for m in matches]
    except Exception:
        return []

# === Step 2: Register UDF ===
schema = ArrayType(StructType([
    StructField("title", StringType(), True),
    StructField("type", StringType(), True),
    StructField("url", StringType(), True),
]))

extract_udf = udf(extract_title_type_url, schema)

# === Step 3: Apply to response column ===
df = spark.table("wei_lab_sander_mlflow.llm_claude_response")

df_with_matches = df.withColumn("extracted_info", extract_udf(col("content")))

# === Step 4: Explode to flatten each title/type/url, include disease_name ===
df_exploded = df_with_matches.select(
    col("id"),
    col("disease_name"),
    explode("extracted_info").alias("info")
).select(
    "disease_name",
    "info.title",
    "info.type",
    "info.url"
).distinct()

from pyspark.sql.functions import regexp_extract

df_final = df_exploded.withColumn(
    "domain",
    regexp_extract(col("url"), r"https?://([^/]+)", 1)
)

df_final.createOrReplaceTempView("llm_claude_response_extracted_view")

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(url) as citation_count,count(distinct disease_name) as total_disease from llm_claude_response_extracted_view

# COMMAND ----------

# MAGIC %sql
# MAGIC select 
# MAGIC   sum(citation_count ) as total_citations,
# MAGIC   count(distinct disease_name) as total_diseases,
# MAGIC   round(avg(citation_count),2) as avg_citation_count,
# MAGIC   round(stddev(citation_count),2) as std_citation_count
# MAGIC from (
# MAGIC   select disease_name, count(url) as citation_count 
# MAGIC   from llm_claude_response_extracted_view
# MAGIC   group by disease_name
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ===============================
# MAGIC -- Step 1: Base table with category
# MAGIC -- ===============================
# MAGIC WITH base AS (
# MAGIC     SELECT
# MAGIC         domain AS website,
# MAGIC         disease_name,
# MAGIC         url,
# MAGIC
# MAGIC         CASE
# MAGIC             WHEN domain IN (
# MAGIC                 'nih.gov', 
# MAGIC                 'ncbi.nlm.nih.gov', 
# MAGIC                 'pubmed.ncbi.nlm.nih.gov', 
# MAGIC                 'pmc.ncbi.nlm.nih.gov', 
# MAGIC                 'medlineplus.gov',
# MAGIC                 'www.ncbi.nlm.nih.gov'
# MAGIC             ) THEN 'Government / Official Databases'
# MAGIC
# MAGIC             WHEN domain IN (
# MAGIC                 'rarediseases.org',
# MAGIC                 'www.orpha.net'
# MAGIC             ) THEN 'Rare Disease Databases'
# MAGIC
# MAGIC             WHEN domain IN (
# MAGIC                 'medscape.com',
# MAGIC                 'emedicine.medscape.com',
# MAGIC                 'clevelandclinic.org',
# MAGIC                 'my.clevelandclinic.org',
# MAGIC                 'mayoclinic.org'
# MAGIC             ) THEN 'Clinical Medical Reference'
# MAGIC
# MAGIC             WHEN domain IN (
# MAGIC                 'onlinelibrary.wiley.com',
# MAGIC                 'mdpi.com',
# MAGIC                 'researchgate.net',
# MAGIC                 'www.sciencedirect.com'
# MAGIC             ) THEN 'Academic Journals / Research'
# MAGIC
# MAGIC             WHEN domain IN ('wikipedia.org', 'en.wikipedia.org')
# MAGIC                 THEN 'General Encyclopedia'
# MAGIC
# MAGIC             WHEN domain = 'youtube.com'
# MAGIC                 THEN 'Media Platforms'
# MAGIC
# MAGIC             ELSE 'Other / Unknown'
# MAGIC         END AS category
# MAGIC     FROM llm_claude_response_extracted_view
# MAGIC ),
# MAGIC
# MAGIC -- =============================================
# MAGIC -- Step 2: Compute disease_count per website
# MAGIC -- =============================================
# MAGIC ranked AS (
# MAGIC     SELECT
# MAGIC         website,
# MAGIC         category,
# MAGIC         COUNT(DISTINCT disease_name) AS disease_count
# MAGIC     FROM base
# MAGIC     GROUP BY website, category
# MAGIC ),
# MAGIC
# MAGIC -- =============================================
# MAGIC -- Step 3: Select TOP 10 websites
# MAGIC -- =============================================
# MAGIC top10 AS (
# MAGIC     SELECT website, category
# MAGIC     FROM ranked
# MAGIC     ORDER BY disease_count DESC
# MAGIC     LIMIT 10
# MAGIC ),
# MAGIC
# MAGIC -- =============================================
# MAGIC -- Step 4: Filter main dataset to top 10 only
# MAGIC -- =============================================
# MAGIC filtered AS (
# MAGIC     SELECT b.*
# MAGIC     FROM base b
# MAGIC     JOIN top10 t USING (website)
# MAGIC )
# MAGIC
# MAGIC -- =============================================
# MAGIC -- Step 5: Final output per category
# MAGIC -- =============================================
# MAGIC SELECT
# MAGIC     category,
# MAGIC     COUNT(DISTINCT disease_name) AS unique_disease_count,
# MAGIC     ROUND(COUNT(url) * 100.0 / 26848, 2) AS citation_percentage
# MAGIC FROM filtered
# MAGIC GROUP BY category
# MAGIC ORDER BY unique_disease_count DESC;
# MAGIC
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 1. Input Data for Each LLM
# ============================================

o3_data = [
    ("Government / Official Databases", 1298, 55.42),
    ("Rare Disease Databases", 836, 11.26),
    ("Clinical Medical Reference", 268, 2.10),
    ("Academic Journals / Research", 309, 1.63),
    ("General Encyclopedia", 260, 1.24),
]

gemini_data = [
    ("Government / Official Databases", 1292, 26.36),
    ("Clinical Medical Reference", 823, 8.18),
    ("Rare Disease Databases", 1149, 7.81),
    ("General Encyclopedia", 1092, 4.68),
    ("Academic Journals / Research", 436, 2.11),
    ("Media Platforms", 283, 1.40),
]

claude_data = [
    ("Government / Official Databases", 1314, 28.94),
    ("Rare Disease Databases", 1103, 6.93),
    ("Clinical Medical Reference", 835, 6.41),
    ("Academic Journals / Research", 945, 6.25),
    ("General Encyclopedia", 1010, 4.12),
]

schema = ["category", "unique_disease_count", "citation_percentage"]

df_o3 = spark.createDataFrame(o3_data, schema).withColumn("llm", F.lit("O3"))
df_gemini = spark.createDataFrame(gemini_data, schema).withColumn("llm", F.lit("Gemini"))
df_claude = spark.createDataFrame(claude_data, schema).withColumn("llm", F.lit("Claude"))

# ============================================
# 2. Combine All
# ============================================

df_all = df_o3.union(df_gemini).union(df_claude)
pdf = df_all.toPandas()

# ============================================
# 3. Pivot Tables
# ============================================

pivot_count = (
    pdf.pivot(index="category", columns="llm", values="unique_disease_count")
       .fillna(0)
)

pivot_pct = (
    pdf.pivot(index="category", columns="llm", values="citation_percentage")
       .fillna(0)
)

# ============================================
# Helper: Label ALL bars
# ============================================

def label_all_bars(ax, bars, fmt="{:.0f}", fontsize=10):
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize
        )

# ============================================
# 4. Plot 1 — Unique Disease Count
# ============================================

fig1, ax1 = plt.subplots(figsize=(12, 6))

categories = pivot_count.index
x = np.arange(len(categories))
width = 0.25

bars_o3 = ax1.bar(x - width, pivot_count["O3"], width, label="O3")
bars_gemini = ax1.bar(x, pivot_count["Gemini"], width, label="Gemini")
bars_claude = ax1.bar(x + width, pivot_count["Claude"], width, label="Claude")

# Label ALL bars
label_all_bars(ax1, bars_o3)
label_all_bars(ax1, bars_gemini)
label_all_bars(ax1, bars_claude)

ax1.set_xticks(x)
ax1.set_xticklabels(categories, rotation=45, ha="right")
ax1.set_ylabel("Unique Disease Count")
ax1.set_title("Unique Disease Coverage by Category Across LLMs")
ax1.legend()

plt.tight_layout()
plt.show()

# ============================================
# 5. Plot 2 — Citation Percentage
# ============================================

fig2, ax2 = plt.subplots(figsize=(12, 6))

bars_o3 = ax2.bar(x - width, pivot_pct["O3"], width, label="O3")
bars_gemini = ax2.bar(x, pivot_pct["Gemini"], width, label="Gemini")
bars_claude = ax2.bar(x + width, pivot_pct["Claude"], width, label="Claude")

# Label ALL bars (1 decimal place)
label_all_bars(ax2, bars_o3, fmt="{:.1f}")
label_all_bars(ax2, bars_gemini, fmt="{:.1f}")
label_all_bars(ax2, bars_claude, fmt="{:.1f}")

ax2.set_xticks(x)
ax2.set_xticklabels(categories, rotation=45, ha="right")
ax2.set_ylabel("Citation Percentage (%)")
ax2.set_title("Citation Percentage by Category Across LLMs")
ax2.legend()

plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Gemini Web queries -17640

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
import json

# ===============================================================
# 1) read Gemini response table
# ===============================================================
df_raw = spark.sql("""
    SELECT disease_name, candidates
    FROM workspace_victrsd.wei_lab_sander_mlflow.llm_gemini_response
""")

# ===============================================================
# 2) UDF：parse return json queries
# ===============================================================
def extract_queries(candidates_json):
    if candidates_json is None:
        return []
    try:
        data = json.loads(candidates_json)
        # data: list or dict
        queries = []

        if isinstance(data, list):
            for item in data:
                if "grounding_metadata" in item and \
                   "web_search_queries" in item["grounding_metadata"]:
                    qs = item["grounding_metadata"]["web_search_queries"]
                    if isinstance(qs, list):
                        queries.extend(qs)

        elif isinstance(data, dict):
            if "grounding_metadata" in data and \
               "web_search_queries" in data["grounding_metadata"]:
                queries = data["grounding_metadata"]["web_search_queries"]

        return queries
    except:
        return []

extract_udf = F.udf(extract_queries, ArrayType(StringType()))

# ===============================================================
# 3) explode queries
# ===============================================================
df_queries = (
    df_raw
    .withColumn("queries", extract_udf("candidates"))
    .withColumn("web_search_query", F.explode("queries"))
    .select("disease_name", "web_search_query")
)

df_queries.write.mode("overwrite").saveAsTable("wei_lab_sander_mlflow.llm_gemini_queries")


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   ROUND(AVG(query_count), 2) AS avg_query_count,
# MAGIC   ROUND(STDDEV(query_count), 2) AS std_query_count
# MAGIC FROM (
# MAGIC   SELECT disease_name, COUNT(*) AS query_count
# MAGIC   FROM wei_lab_sander_mlflow.llm_gemini_queries
# MAGIC   GROUP BY disease_name
# MAGIC )
# MAGIC
# MAGIC --17640/1303

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(distinct disease_name,web_search_query) from wei_lab_sander_mlflow.llm_gemini_queries

# COMMAND ----------

# MAGIC %md
# MAGIC ### Claude Web Search queries - 8057

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table wei_lab_sander_mlflow.llm_claude_queries as
# MAGIC WITH parsed AS (
# MAGIC     SELECT
# MAGIC         disease_name,
# MAGIC         from_json(
# MAGIC             content,
# MAGIC             'array<
# MAGIC                 struct<
# MAGIC                     id:string,
# MAGIC                     input:struct<query:string>,
# MAGIC                     name:string,
# MAGIC                     type:string,
# MAGIC                     content:array<string>,
# MAGIC                     tool_use_id:string
# MAGIC                 >
# MAGIC             >'
# MAGIC         ) AS arr
# MAGIC     FROM workspace_victrsd.wei_lab_sander_mlflow.llm_claude_response
# MAGIC ),
# MAGIC
# MAGIC exploded AS (
# MAGIC     SELECT
# MAGIC         disease_name,
# MAGIC         explode(arr) AS item
# MAGIC     FROM parsed
# MAGIC )
# MAGIC
# MAGIC SELECT --DISTINCT
# MAGIC     disease_name,
# MAGIC     item.input.query
# MAGIC FROM exploded
# MAGIC WHERE item.type = 'server_tool_use'
# MAGIC   AND item.name = 'web_search'
# MAGIC   AND item.input.query IS NOT NULL

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   ROUND(AVG(query_count), 2) AS avg_query_count,
# MAGIC   ROUND(STDDEV(query_count), 2) AS std_query_count
# MAGIC FROM (
# MAGIC   SELECT disease_name, COUNT(*) AS query_count
# MAGIC   FROM wei_lab_sander_mlflow.llm_claude_queries
# MAGIC   GROUP BY disease_name
# MAGIC )
# MAGIC
# MAGIC
# MAGIC
# MAGIC --1320/8057

# COMMAND ----------

# MAGIC %md
# MAGIC ### O3 Web search queries - 12324

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table wei_lab_sander_mlflow.llm_o3_queries as
# MAGIC WITH parsed AS (
# MAGIC   SELECT
# MAGIC     disease_name,
# MAGIC     from_json(
# MAGIC       output,
# MAGIC       'array<
# MAGIC         struct<
# MAGIC           id:string,
# MAGIC           summary:array<string>,
# MAGIC           type:string,
# MAGIC           content:string,
# MAGIC           encrypted_content:string,
# MAGIC           status:string,
# MAGIC           action:struct<query:string,type:string>
# MAGIC         >
# MAGIC       >'
# MAGIC     ) AS events
# MAGIC   FROM wei_lab_sander_mlflow.llm_o3_response_final
# MAGIC ),
# MAGIC
# MAGIC exploded AS (
# MAGIC   SELECT
# MAGIC     disease_name,
# MAGIC     e AS event
# MAGIC   FROM parsed
# MAGIC   LATERAL VIEW explode(events) t AS e
# MAGIC )
# MAGIC
# MAGIC SELECT DISTINCT
# MAGIC   disease_name,
# MAGIC   event.action.query AS query
# MAGIC FROM exploded
# MAGIC WHERE event.type = 'web_search_call'
# MAGIC   AND event.action.query IS NOT NULL 
# MAGIC   
# MAGIC   order by disease_name
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   ROUND(AVG(query_count), 2) AS avg_query_count,
# MAGIC   ROUND(STDDEV(query_count), 2) AS std_query_count
# MAGIC FROM (
# MAGIC   SELECT disease_name, COUNT(*) AS query_count
# MAGIC   FROM wei_lab_sander_mlflow.llm_o3_queries
# MAGIC   GROUP BY disease_name
# MAGIC ) --1319/12324

# COMMAND ----------

# MAGIC %md
# MAGIC ### Summary by diseases

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE wei_lab_sander_mlflow.llm_query_sum AS
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT disease_name, COUNT(*) AS claude_query_count
# MAGIC     FROM wei_lab_sander_mlflow.llm_claude_queries
# MAGIC     GROUP BY disease_name

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_mlflow.llm_query_sum

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE wei_lab_sander_mlflow.llm_query_sum AS
# MAGIC WITH claude_base AS (
# MAGIC     SELECT disease_name, COUNT(*) AS claude_query_count
# MAGIC     FROM wei_lab_sander_mlflow.llm_claude_queries
# MAGIC     GROUP BY disease_name
# MAGIC ),
# MAGIC o3 AS (
# MAGIC     SELECT disease_name, COUNT(*) AS o3_query_count
# MAGIC     FROM wei_lab_sander_mlflow.llm_o3_queries
# MAGIC     GROUP BY disease_name
# MAGIC ),
# MAGIC gemini AS (
# MAGIC     SELECT disease_name, COUNT(*) AS gemini_query_count
# MAGIC     FROM wei_lab_sander_mlflow.llm_gemini_queries
# MAGIC     GROUP BY disease_name
# MAGIC ),
# MAGIC merged AS (
# MAGIC     SELECT
# MAGIC         c.disease_name,
# MAGIC         c.claude_query_count,
# MAGIC         COALESCE(o.o3_query_count, 0) AS o3_query_count,
# MAGIC         COALESCE(g.gemini_query_count, 0) AS gemini_query_count
# MAGIC     FROM claude_base c
# MAGIC     LEFT JOIN o3 o ON c.disease_name = o.disease_name
# MAGIC     LEFT JOIN gemini g ON c.disease_name = g.disease_name
# MAGIC )
# MAGIC
# MAGIC -- =====================================================
# MAGIC -- Final Output: model-level statistics + IQR (2 digits)
# MAGIC -- =====================================================
# MAGIC SELECT
# MAGIC     'claude' AS model,
# MAGIC
# MAGIC     ROUND(AVG(claude_query_count), 2) AS avg_query_count,
# MAGIC     ROUND(STDDEV(claude_query_count), 2) AS std_query_count,
# MAGIC
# MAGIC     ROUND(percentile_approx(claude_query_count, 0.25, 10000), 2) AS q1_query_count,
# MAGIC     ROUND(percentile_approx(claude_query_count, 0.50, 10000), 2) AS median_query_count,
# MAGIC     ROUND(percentile_approx(claude_query_count, 0.75, 10000), 2) AS q3_query_count,
# MAGIC     ROUND(
# MAGIC         percentile_approx(claude_query_count, 0.75, 10000)
# MAGIC         - percentile_approx(claude_query_count, 0.25, 10000),
# MAGIC         2
# MAGIC     ) AS iqr_query_count,
# MAGIC
# MAGIC     ROUND(MIN(claude_query_count) FILTER (WHERE claude_query_count > 0), 2) AS min_query_count,
# MAGIC     ROUND(MAX(claude_query_count), 2) AS max_query_count,
# MAGIC
# MAGIC     COUNT(*) FILTER (WHERE claude_query_count > 0) AS num_diseases_with_queries
# MAGIC FROM merged
# MAGIC
# MAGIC
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC     'o3' AS model,
# MAGIC
# MAGIC     ROUND(AVG(o3_query_count), 2) AS avg_query_count,
# MAGIC     ROUND(STDDEV(o3_query_count), 2) AS std_query_count,
# MAGIC
# MAGIC     ROUND(percentile_approx(o3_query_count, 0.25, 10000), 2) AS q1_query_count,
# MAGIC     ROUND(percentile_approx(o3_query_count, 0.50, 10000), 2) AS median_query_count,
# MAGIC     ROUND(percentile_approx(o3_query_count, 0.75, 10000), 2) AS q3_query_count,
# MAGIC     ROUND(
# MAGIC         percentile_approx(o3_query_count, 0.75, 10000)
# MAGIC         - percentile_approx(o3_query_count, 0.25, 10000),
# MAGIC         2
# MAGIC     ) AS iqr_query_count,
# MAGIC
# MAGIC     ROUND(MIN(o3_query_count) FILTER (WHERE o3_query_count > 0), 2) AS min_query_count,
# MAGIC     ROUND(MAX(o3_query_count), 2) AS max_query_count,
# MAGIC
# MAGIC     COUNT(*) FILTER (WHERE o3_query_count > 0) AS num_diseases_with_queries
# MAGIC FROM merged
# MAGIC
# MAGIC
# MAGIC UNION ALL
# MAGIC SELECT
# MAGIC     'gemini' AS model,
# MAGIC
# MAGIC     ROUND(AVG(gemini_query_count), 2) AS avg_query_count,
# MAGIC     ROUND(STDDEV(gemini_query_count), 2) AS std_query_count,
# MAGIC
# MAGIC     ROUND(percentile_approx(gemini_query_count, 0.25, 10000), 2) AS q1_query_count,
# MAGIC     ROUND(percentile_approx(gemini_query_count, 0.50, 10000), 2) AS median_query_count,
# MAGIC     ROUND(percentile_approx(gemini_query_count, 0.75, 10000), 2) AS q3_query_count,
# MAGIC     ROUND(
# MAGIC         percentile_approx(gemini_query_count, 0.75, 10000)
# MAGIC         - percentile_approx(gemini_query_count, 0.25, 10000),
# MAGIC         2
# MAGIC     ) AS iqr_query_count,
# MAGIC
# MAGIC     ROUND(MIN(gemini_query_count) FILTER (WHERE gemini_query_count > 0), 2) AS min_query_count,
# MAGIC     ROUND(MAX(gemini_query_count), 2) AS max_query_count,
# MAGIC
# MAGIC     COUNT(*) FILTER (WHERE gemini_query_count > 0) AS num_diseases_with_queries
# MAGIC FROM merged;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_mlflow.llm_query_sum

# COMMAND ----------

# MAGIC %md
# MAGIC ### UMLS terms count (PubMed)

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH expanded AS (
# MAGIC     SELECT
# MAGIC         patient_id,
# MAGIC         rare_disease_name,
# MAGIC         phenotype_umls_cuis,
# MAGIC         size(split(phenotype_umls_cuis, ",")) AS cui_count
# MAGIC     FROM wei_lab_sander_umls_mapping.pubmed_dataset_update
# MAGIC ),
# MAGIC
# MAGIC stats AS (
# MAGIC     SELECT
# MAGIC         AVG(cui_count) AS avg_count,
# MAGIC         STDDEV(cui_count) AS sd_count,
# MAGIC         percentile_approx(cui_count, 0.5) AS median_count
# MAGIC     FROM expanded
# MAGIC )
# MAGIC
# MAGIC SELECT
# MAGIC     ROUND(avg_count, 2) AS avg_count,
# MAGIC     ROUND(sd_count, 2) AS sd_count,
# MAGIC     ROUND(median_count, 2) AS median_count
# MAGIC FROM stats;
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### UMLS terms count (Non_PubMed)

# COMMAND ----------

from pyspark.sql import functions as F

# === Dataset list ===
datasets = ["mme", "lirical", "mygene2", "ramedis", "hms"]

results = []
all_df_list = []

for ds in datasets:
    print(f"🔍 Processing dataset: {ds}")

    # Load dataset
    table = f"wei_lab_sander_mlflow.{ds}_mapped_hpo_with_counts"
    df = spark.table(table).select(
        "patient_id",
        "rare_disease_cui",
        "rare_disease_name",
        "phenotype_umls_cui_names_count"
    )

    # Convert to int
    df = df.withColumn(
        "phenotype_umls_cui_names_count",
        F.col("phenotype_umls_cui_names_count").cast("int")
    )

    # Add dataset name column (patient_id not unique across datasets)
    df = df.withColumn("dataset", F.lit(ds))

    # Save for overall union
    all_df_list.append(df)

    # === Per-dataset statistics ===
    raw_stats = df.agg(
        F.avg("phenotype_umls_cui_names_count").alias("avg_count"),
        F.stddev("phenotype_umls_cui_names_count").alias("sd_count"),
        F.expr("percentile_approx(phenotype_umls_cui_names_count, 0.5)").alias("median_count")
    )

    # Format to .XX
    stats = raw_stats.select(
        F.lit(ds).alias("dataset"),
        F.round(F.col("avg_count"), 2).alias("avg_count"),
        F.round(F.col("sd_count"), 2).alias("sd_count"),
        F.round(F.col("median_count"), 2).alias("median_count")
    )

    results.append(stats)


# === Combine per-dataset summaries ===
summary_df = results[0]
for r in results[1:]:
    summary_df = summary_df.unionByName(r)

print("\n📌 Per-dataset summary:")
display(summary_df)   # <<< replaced show()


# === OVERALL summary ===
all_combined = all_df_list[0]
for df in all_df_list[1:]:
    all_combined = all_combined.unionByName(df)

raw_overall = all_combined.agg(
    F.avg("phenotype_umls_cui_names_count").alias("avg_count"),
    F.stddev("phenotype_umls_cui_names_count").alias("sd_count"),
    F.expr("percentile_approx(phenotype_umls_cui_names_count, 0.5)").alias("median_count")
)

overall_summary = raw_overall.select(
    F.lit("overall").alias("dataset"),
    F.round(F.col("avg_count"), 2).alias("avg_count"),
    F.round(F.col("sd_count"), 2).alias("sd_count"),
    F.round(F.col("median_count"), 2).alias("median_count")
)

print("\n📌 OVERALL summary across all 5 datasets:")
display(overall_summary)   
