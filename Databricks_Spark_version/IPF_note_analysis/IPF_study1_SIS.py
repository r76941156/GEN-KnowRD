# Databricks notebook source
# MAGIC %md
# MAGIC # PFSA-Based IPF Progression Modeling Pipeline
# MAGIC
# MAGIC This pipeline models disease progression in IPF using patient symptom trajectories derived from EHR data. It includes weekly symptom coding, co-occurrence analysis, pairwise transitions, and patient-specific weighting.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟦 Step 0: Justification for Using Symptom Pairs or TF-IDF Weights
# MAGIC
# MAGIC To justify modeling **symptom pairs**, we calculated the number of distinct symptoms appearing per patient per week before diagnosis.
# MAGIC
# MAGIC | Group   | Avg Symptoms | Median Symptoms |
# MAGIC |---------|--------------|-----------------|
# MAGIC | IPF     |     3.04     |       2.0       |
# MAGIC | Control |     2.46     |       2.0       |
# MAGIC
# MAGIC Patients with IPF frequently exhibit multiple concurrent symptoms. This supports the use of joint symptom states (pairs) for modeling disease progression.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟧 Step 1: Weekly UMLS Code Mapping
# MAGIC
# MAGIC - Filter UMLS terms to a curated list.
# MAGIC - Exclude mentions with context: negated, family, historical, hypothetical.
# MAGIC - Convert `note_date` to `week_start`.
# MAGIC - Encode symptoms per patient-week:
# MAGIC   - `code = 1` → symptom appears
# MAGIC   - `code = 2` → other symptoms appear that week
# MAGIC   - `code = 0` → no symptoms
# MAGIC
# MAGIC **Output:** `IPF_case_group_weekly_umls_code_validation`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟨 Step 2: Symptom Co-occurrence Analysis
# MAGIC
# MAGIC - Identify all symptom pairs appearing in the same week.
# MAGIC - Count co-occurrence frequency and patient counts.
# MAGIC - Compute conditional co-occurrence rate:
# MAGIC
# MAGIC $$
# MAGIC P(\text{symptom}_2 \mid \text{symptom}_1) = \frac{\text{pair count}}{\text{symptom}_1 \text{ count}}
# MAGIC $$
# MAGIC
# MAGIC - Select high-confidence pairs: rate ≥ 0.2, present in ≥ 10 patients.
# MAGIC
# MAGIC **Output:** `symptom_pair_cooccurrence`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟩 Step 3: PFSA Pairwise Transition Modeling
# MAGIC
# MAGIC - Assign `week_index` to each patient's timeline.
# MAGIC - Compute current week `joint_state` (e.g., `"1_1"`) from symptom codes.
# MAGIC - Compute `next_state` using the next week.
# MAGIC - Join severity data and next-week symptom lists.
# MAGIC
# MAGIC **Output:** `pfsa_pair_with_symptoms_next_week_case_validation`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟦 Step 4: Global Symptom Importance Scoring
# MAGIC
# MAGIC Compute the importance of each symptom based on its frequency in IPF vs. control groups:
# MAGIC
# MAGIC $$
# MAGIC \text{importance} = \log_2\left( \frac{\text{freq}_\text{IPF} + 5}{\text{freq}_\text{Control} + 5} \right)
# MAGIC $$
# MAGIC
# MAGIC (+5 smoothing avoids division by zero)
# MAGIC
# MAGIC **Output:** `symptom_importance_scores`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟪 Step 5: Per-Patient Symptom Weighting
# MAGIC
# MAGIC - For each patient and symptom:
# MAGIC   - Count number of weeks it appeared: `frequency`
# MAGIC - Join with global importance score.
# MAGIC - Compute:
# MAGIC
# MAGIC $$
# MAGIC \mathrm{final\ weight} = \mathrm{importance} \cdot \log(1 + \mathrm{frequency})
# MAGIC $$
# MAGIC
# MAGIC
# MAGIC **Output:** `patient_symptom_final_weight_case_validation`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟥 Step 6: Add Patient Weights to Symptom Pairs
# MAGIC
# MAGIC - Split `symptom_pair` into `s1` and `s2`
# MAGIC - Join patient-specific `final_weight` for both symptoms
# MAGIC - Compute:
# MAGIC
# MAGIC $$
# MAGIC \mathrm{pair\ final\ weight} = \mathrm{final\ weight}_1 + \mathrm{final\ weight}_2
# MAGIC $$
# MAGIC
# MAGIC
# MAGIC **Output:** `pfsa_pair_with_patient_final_weight_case_validation`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟫 Step 7: Strengthen PFSA Features
# MAGIC
# MAGIC - Focus on `joint_state = "1_1"` pairs
# MAGIC - For each patient:
# MAGIC   - Count number of weeks with active pair → `pair_week_distinct`
# MAGIC   - Count number of "worsen" episodes → `worsen_count`
# MAGIC - Join with `pair_final_weight`
# MAGIC
# MAGIC **Output:** `pfsa_pair_with_strengthened_features_case_validation`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ⬛ Step 8: Aggregate Features Per Patient
# MAGIC
# MAGIC - For each patient, sum:
# MAGIC   - `pair_week_distinct`
# MAGIC   - `worsen_count`
# MAGIC   - `pair_final_weight`
# MAGIC - Compute:
# MAGIC
# MAGIC $$
# MAGIC \mathrm{normalized\ weight} = \frac{\mathrm{pair\ final\ weight}}{\mathrm{pair\ week\ distinct} + 1}
# MAGIC $$
# MAGIC
# MAGIC
# MAGIC **Used for downstream modeling.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data preparation for signs and symptoms
# MAGIC
# MAGIC

# COMMAND ----------

model_list = ["gemini", "claude", "ds","ro","o3"]
group_list = ["case","non_IPF_control"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Group summary

# COMMAND ----------

# MAGIC %sql
# MAGIC --wei_lab_sander.ipf_case_control_unique_pairs_171
# MAGIC --wei_lab_sander.ipf_case_non_IPF_control_unique_pairs_final
# MAGIC -- ============================
# MAGIC -- 1. Gender Summary
# MAGIC -- ============================
# MAGIC -- SELECT 
# MAGIC --     'Case' AS group_type,
# MAGIC --     gender,
# MAGIC --     COUNT(*) AS n
# MAGIC -- FROM wei_lab_sander.ipf_case_control_unique_pairs_171
# MAGIC -- GROUP BY gender
# MAGIC
# MAGIC -- UNION ALL
# MAGIC
# MAGIC -- SELECT 
# MAGIC --     'Control' AS group_type,
# MAGIC --     gender,
# MAGIC --     COUNT(*) AS n
# MAGIC -- FROM wei_lab_sander.ipf_case_control_unique_pairs_171
# MAGIC -- GROUP BY gender
# MAGIC -- ORDER BY group_type, gender;
# MAGIC
# MAGIC
# MAGIC
# MAGIC -- ============================
# MAGIC -- 2. Race Summary
# MAGIC -- ============================
# MAGIC -- SELECT 
# MAGIC --     'Case' AS group_type,
# MAGIC --     race,
# MAGIC --     COUNT(*) AS n
# MAGIC -- FROM wei_lab_sander.ipf_case_control_unique_pairs_171
# MAGIC -- GROUP BY race
# MAGIC
# MAGIC -- UNION ALL
# MAGIC
# MAGIC -- SELECT 
# MAGIC --     'Control' AS group_type,
# MAGIC --     race,
# MAGIC --     COUNT(*) AS n
# MAGIC -- FROM wei_lab_sander.ipf_case_control_unique_pairs_171
# MAGIC -- GROUP BY race
# MAGIC -- ORDER BY group_type, race;
# MAGIC
# MAGIC
# MAGIC
# MAGIC -- ============================
# MAGIC -- 3. Age Summary (Mean ± SD)
# MAGIC -- ============================
# MAGIC -- SELECT 
# MAGIC --     'Case' AS group_type,
# MAGIC --     round(AVG(case_onset_age),2) AS mean_age,
# MAGIC --     round(STDDEV(case_onset_age),2) AS sd_age
# MAGIC -- FROM wei_lab_sander.ipf_case_control_unique_pairs_171
# MAGIC
# MAGIC -- UNION ALL
# MAGIC
# MAGIC -- SELECT 
# MAGIC --     'Control' AS group_type,
# MAGIC --     round(AVG(control_onset_age),2) AS mean_age,
# MAGIC --     round(STDDEV(control_onset_age),2) AS sd_age
# MAGIC -- FROM wei_lab_sander.ipf_case_control_unique_pairs_171
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

for group in group_list:
    for model in model_list:
        print(f"Processing group: {group}, Processing model: {model}")

        # 1–2. Load data (keep note_date and severity)
        if group == "case":
            df = spark.sql(f"""
                SELECT person_id, umls_name, note_date, severity
                FROM wei_lab_sander_ipf.{group}_group_{model}_sign_symptom_notes
                WHERE person_id IN (
                    SELECT case_id
                    FROM wei_lab_sander.ipf_case_non_IPF_control_unique_pairs_final
                )
            """)
        else:
            df = spark.sql(f"""
                SELECT person_id, umls_name, note_date, severity
                FROM wei_lab_sander_ipf.{group}_group_{model}_sign_symptom_notes
            """)

        # 3. Add week_start and deduplicate by priority
        df = df.withColumn("week_start", F.date_trunc("week", F.col("note_date")))

        # Define priority: severity != 'unknown' gets higher priority (0),
        # otherwise lower priority (1); then sort by note_date (newest first)
        priority_col = F.when(F.col("severity") != "unknown", 0).otherwise(1)
        w = (
            Window
            .partitionBy("person_id", "umls_name", "week_start")
            .orderBy(priority_col, F.col("note_date").desc())
        )

        df_dedup = (
            df.withColumn("rn", F.row_number().over(w))
              .filter(F.col("rn") == 1)   # keep only the top-ranked record
              .drop("rn")
        )

        # 4. Continue weekly / flag / code logic using df_dedup as input
        df_weekly = df_dedup.select(
            "person_id", "umls_name", "week_start", "severity"
        )

        # ----------------------------
        # 5. Build presence flags (code)
        # ----------------------------
        # Flag weeks where the symptom is present (has_umls = 1)
        df_present = df_weekly.withColumn("has_umls", F.lit(1))

        # All combinations: all persons × all weeks × all symptoms
        all_weeks = df_weekly.select("person_id", "week_start").distinct()
        all_symptoms = df_weekly.select("person_id", "umls_name").distinct()
        all_combos = all_weeks.join(all_symptoms, on="person_id")

        # Left join presence records
        df_flags = (
            all_combos
            .join(
                df_present.select(
                    "person_id", "week_start", "umls_name", "has_umls", "severity"
                ),
                on=["person_id", "week_start", "umls_name"],
                how="left"
            )
            .fillna({"has_umls": 0, "severity": "unknown"})
        )

        # Add flag indicating whether any symptom appears in that week
        df_other = (
            df_weekly
            .select("person_id", "week_start")
            .distinct()
            .withColumn("has_any", F.lit(1))
        )

        df_flags = (
            df_flags
            .join(df_other, on=["person_id", "week_start"], how="left")
            .fillna({"has_any": 0})
        )

        # Final code encoding: 0 / 1 / 2
        df_final = df_flags.withColumn(
            "code",
            F.when(F.col("has_umls") == 1, 1)
             .when(F.col("has_any") == 1, 2)
             .otherwise(0)
        )

        # ----------------------------
        # 0. Load IPF diagnosis dates
        # ----------------------------
        if group == "case":
            sql = f"""
                SELECT
                    case_id AS person_id,
                    {group}_dx_date AS dx_date
                FROM wei_lab_sander.ipf_case_non_IPF_control_unique_pairs_final
            """
        else:
            sql = f"""
                SELECT
                    control_id AS person_id,
                    {group}_dx_date AS dx_date
                FROM wei_lab_sander.ipf_case_non_IPF_control_unique_pairs_final
            """

        diagnosis_df = spark.sql(sql)

        # ----------------------------
        # 6. Add before/after diagnosis label and save
        # ----------------------------
        output_df = (
            df_final
            .join(diagnosis_df, on="person_id", how="left")
            .withColumn(
                "before_after_dx",
                F.when(F.col("week_start") < F.col("dx_date"), "before")
                 .otherwise("after")
            )
            .drop("dx_date")
            .select(
                "person_id",
                "umls_name",
                "week_start",
                "code",
                "severity",
                "before_after_dx"
            )
        )

        if group == "case":
            output_df.write.mode("overwrite").saveAsTable(
                f"wei_lab_sander_IPF.{group}_group_for_non_IPF_control_{model}_weekly_ss_umls_code"
            )
        else:
            output_df.write.mode("overwrite").saveAsTable(
                f"wei_lab_sander_IPF.non_IPF_control_group_{model}_weekly_ss_umls_code"
            )

        print("Done!")


# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select 'case' as group,count(*),count(distinct person_id) from wei_lab_sander_IPF.case_group_for_non_IPF_control_semmeddb_weekly_ss_umls_code where before_after_dx = 'before'
# MAGIC union all
# MAGIC select 'non_IPF_control' as group,count(*),count(distinct person_id) from wei_lab_sander_IPF.non_IPF_control_group_semmeddb_weekly_ss_umls_code where before_after_dx = 'before'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Symptom Pair Cooccurrence before diagnosis

# COMMAND ----------

import pyspark.sql.functions as F

# Load weekly data

for group in group_list:
    for model in model_list:

        print(f"Processing group: {group}, Processing {model}...")

        if group == "case":
            df = spark.sql(
                f"""
                SELECT *
                FROM wei_lab_sander_IPF.{group}_group_for_non_IPF_control_{model}_weekly_ss_umls_code
                WHERE before_after_dx = 'before'
                """
            )
        else:
            df = spark.sql(
                f"""
                SELECT *
                FROM wei_lab_sander_IPF.non_IPF_control_group_{model}_weekly_ss_umls_code
                WHERE before_after_dx = 'before'
                """
            )

        # Step 1: Filter code = 1 (symptom is actually present)
        df_active = (
            df.filter(F.col("code") == 1)
              .select("person_id", "week_start", "umls_name")
              .distinct()
        )

        # Step 2: Generate symptom pairs co-occurring in the same week
        df_pairs = (
            df_active.alias("a")
            .join(
                df_active.alias("b"),
                on=["person_id", "week_start"]
            )
            .filter(F.col("a.umls_name") < F.col("b.umls_name"))
            .select(
                "person_id",
                "week_start",  # keeps the exact week of co-occurrence
                F.col("a.umls_name").alias("symptom_1"),
                F.col("b.umls_name").alias("symptom_2")
            )
        )

        # Step 3: Count co-occurrence frequency and number of patients
        df_pair_counts = (
            df_pairs
            .groupBy("symptom_1", "symptom_2")
            .agg(
                F.count("*").alias("count"),
                F.countDistinct("person_id").alias("distinct_person_count")
            )
        )

        # Step 4: Compute total occurrence count for symptom_1
        df_symptom_freq = (
            df_active
            .groupBy("umls_name")
            .agg(F.count("*").alias("symptom_1_total"))
        )

        # Step 5: Join and compute conditional co-occurrence rate
        df_pair_prob = (
            df_pair_counts
            .join(
                df_symptom_freq.withColumnRenamed("umls_name", "symptom_1"),
                on="symptom_1",
                how="left"
            )
            .withColumn(
                "cooccurrence_rate",
                F.col("count") / F.col("symptom_1_total")
            )
        )

        # Step 6: Save results
        if group == "case":
            df_pair_prob.write.mode("overwrite").saveAsTable(
                f"wei_lab_sander_IPF.{group}_for_non_IPF_control_{model}_symptom_pair_cooccurrence"
            )
        else:
            df_pair_prob.write.mode("overwrite").saveAsTable(
                f"wei_lab_sander_IPF.non_IPF_control_{model}_symptom_pair_cooccurrence"
            )

        print("Done!")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Multiple Sympton pairs before diagnosis

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_IPF.case_for_non_IPF_control_semmeddb_symptom_pair_cooccurrence order by distinct_person_count desc

# COMMAND ----------

# MAGIC %md
# MAGIC ### We select at least 10 pts have pairs to build their state info

# COMMAND ----------

from pyspark.sql import functions as F, Window

for group in group_list:
    for model in model_list:
        print(f"Processing group: {group}, Processing {model}...")

        # ----------------------------
        # 1. Load highly co-occurring symptom pairs
        # ----------------------------
        if group == "case":
            co_table = f"wei_lab_sander_IPF.{group}_for_non_IPF_control_{model}_symptom_pair_cooccurrence"
        else:
            co_table = f"wei_lab_sander_IPF.non_IPF_control_{model}_symptom_pair_cooccurrence"

        df_co = spark.table(co_table)

        df_top = (
            df_co
            .filter(F.col("distinct_person_count") >= 10)
            .select("symptom_1", "symptom_2")
        )

        top_pairs_list = [(r["symptom_1"], r["symptom_2"]) for r in df_top.collect()]

        # ----------------------------
        # 2. Load weekly data and add week_index
        # ----------------------------
        if group == "case":
            df_week = spark.sql(
                f"""
                SELECT *
                FROM wei_lab_sander_IPF.{group}_group_for_non_IPF_control_{model}_weekly_ss_umls_code
                WHERE before_after_dx = 'before'
                """
            )
        else:
            df_week = spark.sql(
                f"""
                SELECT *
                FROM wei_lab_sander_IPF.non_IPF_control_group_{model}_weekly_ss_umls_code
                WHERE before_after_dx = 'before'
                """
            )

        df_week_unique = df_week.select("person_id", "week_start").distinct()

        w_idx = Window.partitionBy("person_id").orderBy("week_start")
        df_week_indexed = (
            df_week_unique
            .withColumn("week_index", (F.row_number().over(w_idx) - 1).cast("int"))
        )

        df_week = df_week.join(
            df_week_indexed, on=["person_id", "week_start"], how="left"
        )

        # ----------------------------
        # 3. Build joint_state per pair, and add severity and week_start
        # ----------------------------
        all_results = []

        for s1, s2 in top_pairs_list:

            # Build joint state
            df_pair_week = (
                df_week
                .filter(F.col("umls_name").isin([s1, s2]))
                .groupBy("person_id", "week_index", "week_start")
                .pivot("umls_name", [s1, s2])
                .agg(F.max("code"))
                .fillna(0)
                .withColumn(
                    "joint_state",
                    F.concat_ws("_", F.col(s1), F.col(s2))
                )
            )

            # Next-week joint_state
            w_next = Window.partitionBy("person_id").orderBy("week_index")
            df_pair_week = df_pair_week.withColumn(
                "next_state", F.lead("joint_state").over(w_next)
            )
            df_pair_week = df_pair_week.filter(F.col("next_state").isNotNull())

            # Add current-week severity
            df_sev_cur = (
                df_week
                .filter(F.col("umls_name").isin([s1, s2]))
                .select("person_id", "week_index", "umls_name", "severity")
            )

            df_sev_cur_pivot = (
                df_sev_cur
                .groupBy("person_id", "week_index")
                .pivot("umls_name", [s1, s2])
                .agg(F.first("severity"))
                .withColumnRenamed(s1, "severity_1")
                .withColumnRenamed(s2, "severity_2")
            )

            df_pair_week = df_pair_week.join(
                df_sev_cur_pivot,
                on=["person_id", "week_index"],
                how="left"
            )

            # Other symptoms in the next week
            df_next_sym = (
                df_week
                .filter(
                    (F.col("code") == 1) &
                    (~F.col("umls_name").isin([s1, s2]))
                )
                .select("person_id", "week_index", "umls_name", "severity")
            )

            # Concatenate symptom name and severity
            df_next_sym = df_next_sym.withColumn(
                "symptom_with_severity",
                F.concat_ws(":", F.col("umls_name"), F.col("severity"))
            )

            # Collect other symptoms in the next week
            df_joined = (
                df_pair_week.alias("cur")
                .join(
                    df_next_sym.alias("nxt"),
                    (F.col("cur.person_id") == F.col("nxt.person_id")) &
                    (F.col("cur.week_index") + 1 == F.col("nxt.week_index")),
                    how="left"
                )
                .groupBy(
                    "cur.person_id",
                    "cur.week_index",
                    "cur.week_start",
                    "joint_state",
                    "next_state",
                    "severity_1",
                    "severity_2"
                )
                .agg(
                    F.collect_set("nxt.umls_name").alias("other_symptoms_in_next_week"),
                    F.collect_set("nxt.symptom_with_severity").alias("other_symptoms_with_severity")
                )
                .withColumn("symptom_pair", F.lit(f"{s1}|{s2}"))
            )

            all_results.append(df_joined)

        # ----------------------------
        # 4. Merge and save results
        # ----------------------------
        df_final = all_results[0]
        for df in all_results[1:]:
            df_final = df_final.unionByName(df)

        if group == "case":
            df_final.write.mode("overwrite").saveAsTable(
                f"wei_lab_sander_IPF.{group}_group_for_non_IPF_control_{model}_pfsa_pair_with_symptoms"
            )
        else:
            df_final.write.mode("overwrite").saveAsTable(
                f"wei_lab_sander_IPF.non_IPF_control_{model}_pfsa_pair_with_symptoms"
            )

        print("Done!")


# COMMAND ----------

# MAGIC %sql
# MAGIC select 'case' as group, count(*),count(distinct person_id) from wei_lab_sander_IPF.case_group_for_non_IPF_control_semmeddb_pfsa_pair_with_symptoms where joint_state = '1_1'
# MAGIC union all
# MAGIC select 'control' as group, count(*),count(distinct person_id) from wei_lab_sander_IPF.non_IPF_control_semmeddb_pfsa_pair_with_symptoms where joint_state = '1_1'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Important score - single sympton

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

# model_list = ["gemini", "claude", "ds", "ro", "o3"]

for model in model_list:
    print("Processing model:", model)

    # Load IPF and Control group data
    df_ipf = spark.sql(f"""
        SELECT person_id, umls_name
        FROM wei_lab_sander_IPF.case_group_for_non_IPF_control_{model}_weekly_ss_umls_code
        WHERE code = 1
          AND before_after_dx = 'before'
    """)

    df_ctrl = spark.sql(f"""
        SELECT person_id, umls_name
        FROM wei_lab_sander_IPF.non_IPF_control_group_{model}_weekly_ss_umls_code
        WHERE code = 1
          AND before_after_dx = 'before'
    """)

    # Count the number of patients with each symptom in the IPF group
    ipf_freq = (
        df_ipf
        .groupBy("umls_name")
        .agg(F.countDistinct("person_id").alias("freq_ipf"))
    )

    # Count the number of patients with each symptom in the Control group
    ctrl_freq = (
        df_ctrl
        .groupBy("umls_name")
        .agg(F.countDistinct("person_id").alias("freq_ctrl"))
    )

    # Merge and compute importance score
    df_score = (
        ipf_freq
        .join(ctrl_freq, on="umls_name", how="outer")
        .fillna(0)
        .withColumn(
            "importance_score",
            F.log2((F.col("freq_ipf") + 5) / (F.col("freq_ctrl") + 5))
        )
    )

    # Save results
    df_score.write.mode("overwrite").saveAsTable(
        f"wei_lab_sander_IPF.ss_case_non_IPF_control_{model}_importance_scores"
    )

    print("Done!")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_IPF.ss_case_non_IPF_control_allmodel_importance_scores order by importance_score asc

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final weight (importance score with frequency_within_patient)

# COMMAND ----------

#model_list = ["gemini", "claude", "ds","ro","o3"]
group_list = ["case","non_IPF_control"]

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Load weekly symptom table

for group in group_list:
    for model in model_list:

        print(f"Processing group: {group}, Processing model: {model}")
        print("Loading data...")

        if group == "case":
            df = spark.sql(
                f"""
                SELECT *
                FROM wei_lab_sander_IPF.{group}_group_for_non_IPF_control_{model}_weekly_ss_umls_code
                WHERE before_after_dx = 'before'
                """
            )
        else:
            df = spark.sql(
                f"""
                SELECT *
                FROM wei_lab_sander_IPF.non_IPF_control_group_{model}_weekly_ss_umls_code
                WHERE before_after_dx = 'before'
                """
            )

        # Load importance scores
        imp_df = spark.table(
            f"wei_lab_sander_IPF.ss_case_non_IPF_control_{model}_importance_scores"
        )  # columns: umls_name, importance_score

        # Only consider weeks where the symptom is actually present (code = 1)
        df_present = df.filter(F.col("code") == 1)

        # Compute frequency within each patient
        df_freq = (
            df_present
            .groupBy("person_id", "umls_name")
            .agg(F.countDistinct("week_start").alias("freq_within_patient"))
        )

        # Join importance_score
        df_score = df_freq.join(imp_df, on="umls_name", how="left")

        # Compute umls_weight = importance_score × log(1 + frequency_within_patient)
        df_score = df_score.withColumn(
            "umls_weight",
            F.col("importance_score") * F.log1p(F.col("freq_within_patient"))  # log(1 + x)
        )

        # Save results
        if group == "case":
            (
                df_score
                .select(
                    "person_id",
                    "umls_name",
                    "freq_within_patient",
                    "importance_score",
                    "umls_weight"
                )
                .write.mode("overwrite")
                .saveAsTable(
                    f"wei_lab_sander_IPF.{group}_group_for_non_IPF_control_{model}_patient_symptom_weight"
                )
            )
        else:
            (
                df_score
                .select(
                    "person_id",
                    "umls_name",
                    "freq_within_patient",
                    "importance_score",
                    "umls_weight"
                )
                .write.mode("overwrite")
                .saveAsTable(
                    f"wei_lab_sander_IPF.non_IPF_control_{model}_patient_symptom_weight"
                )
            )

        print("Done!")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_IPF.non_IPF_control_semmeddb_patient_symptom_weight

# COMMAND ----------

# MAGIC %md
# MAGIC ### Having final weight for pt's sign and symptom

# COMMAND ----------

from pyspark.sql.functions import split, col, sum as F_sum

for group in group_list:
    for model in model_list:
        print(f"Processing group: {group}, Processing model: {model}")
        print("Loading data...")

        # Split symptom_pair into s1 and s2
        if group == "case":
            pair_df = spark.table(
                f"wei_lab_sander_IPF.{group}_group_for_non_IPF_control_{model}_pfsa_pair_with_symptoms"
            )
        else:
            pair_df = spark.table(
                f"wei_lab_sander_IPF.non_IPF_control_{model}_pfsa_pair_with_symptoms"
            )

        pair_df_split = (
            pair_df
            .withColumn("s1", split("symptom_pair", "\\|")[0])
            .withColumn("s2", split("symptom_pair", "\\|")[1])
        )

        # Load patient-level final symptom weights
        # and rename columns separately for s1 and s2
        if group == "case":
            weight_df = spark.table(
                f"wei_lab_sander_IPF.{group}_group_for_non_IPF_control_{model}_patient_symptom_weight"
            )
        else:
            weight_df = spark.table(
                f"wei_lab_sander_IPF.non_IPF_control_{model}_patient_symptom_weight"
            )

        # Prepare s1 weight columns
        weight_s1 = weight_df.select(
            col("person_id"),
            col("umls_name").alias("s1"),
            col("umls_weight").alias("umls_weight_1")
        )

        # Prepare s2 weight columns
        weight_s2 = weight_df.select(
            col("person_id"),
            col("umls_name").alias("s2"),
            col("umls_weight").alias("umls_weight_2")
        )

        # Join s1 weights
        pair_with_w1 = pair_df_split.join(
            weight_s1, on=["person_id", "s1"], how="left"
        )

        # Join s2 weights
        pair_with_w2 = pair_with_w1.join(
            weight_s2, on=["person_id", "s2"], how="left"
        )

        # Extract all symptoms (s1 / s2) with their weights
        # and unify them into a single column
        symptom_1_df = (
            pair_with_w2
            .select("person_id", "s1", "umls_weight_1")
            .withColumnRenamed("s1", "umls")
            .withColumnRenamed("umls_weight_1", "weight")
        )

        symptom_2_df = (
            pair_with_w2
            .select("person_id", "s2", "umls_weight_2")
            .withColumnRenamed("s2", "umls")
            .withColumnRenamed("umls_weight_2", "weight")
        )

        # Merge and remove duplicates
        # (avoid double-counting the same symptom across multiple pairs)
        all_symptoms_df = (
            symptom_1_df
            .unionByName(symptom_2_df)
            .dropDuplicates(["person_id", "umls"])
        )

        # Compute total symptom weight per patient
        patient_symptom_weight_df = (
            all_symptoms_df
            .groupBy("person_id")
            .agg(F_sum("weight").alias("total_symptom_weight"))
        )

        # Save results
        if group == "case":
            patient_symptom_weight_df.write.mode("overwrite").saveAsTable(
                f"wei_lab_sander_IPF.{group}_group_for_non_IPF_control_{model}_pfsa_pair_pt_total_weight"
            )
        else:
            patient_symptom_weight_df.write.mode("overwrite").saveAsTable(
                f"wei_lab_sander_IPF.non_IPF_control_{model}_pfsa_pair_pt_total_weight"
            )

        print("Done!")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_IPF.non_IPF_control_allmodel_pfsa_pair_pt_total_weight order by total_symptom_weight desc