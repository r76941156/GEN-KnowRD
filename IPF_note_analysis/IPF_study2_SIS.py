# Databricks notebook source
# MAGIC %md
# MAGIC # PFSA-Based IPF Progression Modeling Pipeline
# MAGIC
# MAGIC This pipeline models disease progression in IPF using patient symptom trajectories derived from EHR data. It includes weekly symptom coding, co-occurrence analysis, pairwise transitions, and patient-specific weighting.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🟦 Step 0: Justification for Using Symptom Pairs
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
# MAGIC ## 🟦 Step 4: Global Symptom Importance Scoring or TF-IDF Weights
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
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Step 9: Logistic Regression for Performance Evaluation
# MAGIC
# MAGIC ---
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data preparation for signs and symptoms
# MAGIC
# MAGIC

# COMMAND ----------

model_list = ["allmodel","claude", "ds","ro","o3"]
group_list = ["case","control"]

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

for group in group_list:
    for model in model_list:
        print(f"Processing {group} group, Processing {model} model...")

        # 1–2. Read data (keep note_date and severity)
        df = spark.sql(f"""
            SELECT person_id, umls_name, note_date, severity
            FROM wei_lab_sander_ipf.{group}_group_{model}_sign_symptom_notes
        """)

        # 3. Add week_start + deduplicate based on priority
        df = df.withColumn("week_start", F.date_trunc("week", F.col("note_date")))

        # Define priority:
        # severity != 'unknown' → priority 0
        # severity == 'unknown' → priority 1
        # then order by note_date (newest first)
        priority_col = F.when(F.col("severity") != "unknown", 0).otherwise(1)

        w = (
            Window
            .partitionBy("person_id", "umls_name", "week_start")
            .orderBy(priority_col, F.col("note_date").desc())
        )

        df_dedup = (
            df.withColumn("rn", F.row_number().over(w))
              .filter(F.col("rn") == 1)   # keep only top-ranked record
              .drop("rn")                 # drop helper column
        )

        # 4. Continue weekly / flag / code logic using df_dedup
        df_weekly = df_dedup.select(
            "person_id", "umls_name", "week_start", "severity"
        )

        # ----------------------------
        # 5. Create presence flag (code)
        # ----------------------------
        # Flag indicating the UMLS appears in that week
        df_present = df_weekly.withColumn("has_umls", F.lit(1))

        # All combinations: all persons × all weeks × all symptoms
        all_weeks = df_weekly.select("person_id", "week_start").distinct()
        all_symptoms = df_weekly.select("person_id", "umls_name").distinct()
        all_combos = all_weeks.join(all_symptoms, on="person_id")

        # Left join observed records
        df_flags = (
            all_combos.join(
                df_present.select(
                    "person_id", "week_start", "umls_name",
                    "has_umls", "severity"
                ),
                on=["person_id", "week_start", "umls_name"],
                how="left"
            )
            .fillna({"has_umls": 0, "severity": "unknown"})
        )

        # Add flag: whether ANY symptom appears in that week
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

        # Final code definition:
        # 1 = this UMLS appears
        # 2 = other symptoms appear, but not this one
        # 0 = no symptoms at all
        df_final = df_flags.withColumn(
            "code",
            F.when(F.col("has_umls") == 1, 1)
             .when(F.col("has_any") == 1, 2)
             .otherwise(0)
        )

        # ----------------------------
        # 0. Read IPF diagnosis date
        # ----------------------------
        diagnosis_df = spark.sql(f"""
            SELECT
                {group}_id AS person_id,
                {group}_dx_date AS dx_date
            FROM wei_lab_sander.ipf_case_control_unique_pairs_171
        """)

        # ----------------------------
        # 6. Add pre-/post-diagnosis label and save
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
                "person_id", "umls_name", "week_start",
                "code", "severity", "before_after_dx"
            )
        )

        output_df.write.mode("overwrite").saveAsTable(
            f"wei_lab_sander_IPF.{group}_group_{model}_weekly_ss_umls_code"
        )

        print("Done!")


# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select 'case' as group,count(*),count(distinct person_id) from wei_lab_sander_IPF.case_group_semmeddb_weekly_ss_umls_code where before_after_dx = 'before'
# MAGIC union all
# MAGIC select 'control' as group,count(*),count(distinct person_id) from wei_lab_sander_IPF.control_group_semmeddb_weekly_ss_umls_code where before_after_dx = 'before'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Symptom Pair Cooccurrence before diagnosis

# COMMAND ----------

import pyspark.sql.functions as F

# Read weekly data
for group in group_list:
    for model in model_list:
        print(f"Processing {group} group, model: {model}...")

        df = spark.sql(
            f"""
            SELECT *
            FROM wei_lab_sander_IPF.{group}_group_{model}_weekly_ss_umls_code
            WHERE before_after_dx = 'before'
            """
        )

        # Step 1: Filter code = 1 (symptom truly appears)
        df_active = (
            df.filter(F.col("code") == 1)
              .select("person_id", "week_start", "umls_name")
              .distinct()
        )

        # Step 2: Generate symptom co-occurrence pairs within the same person-week
        df_pairs = (
            df_active.alias("a")
            .join(
                df_active.alias("b"),
                on=["person_id", "week_start"]
            )
            .filter(F.col("a.umls_name") < F.col("b.umls_name"))
            .select(
                "person_id",
                "week_start",  # keep the exact co-occurrence week
                F.col("a.umls_name").alias("symptom_1"),
                F.col("b.umls_name").alias("symptom_2")
            )
        )

        # Step 3: Count co-occurrence frequency and number of distinct patients
        df_pair_counts = (
            df_pairs.groupBy("symptom_1", "symptom_2")
            .agg(
                F.count("*").alias("count"),
                F.countDistinct("person_id").alias("distinct_person_count")
            )
        )

        # Step 4: Count total occurrences of symptom_1
        df_symptom_freq = (
            df_active.groupBy("umls_name")
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
        df_pair_prob.write.mode("overwrite").saveAsTable(
            f"wei_lab_sander_IPF.{group}_{model}_symptom_pair_cooccurrence"
        )

        print("Done!")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Multiple Sympton pairs before diagnosis

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_IPF.case_semmeddb_symptom_pair_cooccurrence order by distinct_person_count desc

# COMMAND ----------

# MAGIC %md
# MAGIC ### We select at least 10 pts have pairs to build their state info

# COMMAND ----------

from pyspark.sql import functions as F, Window

# ----------------------------
# 1. Load highly associated symptom co-occurrence pairs
# ----------------------------
for group in group_list:
    for model in model_list:
        print(f"Processing {group} group, model: {model}...")

        co_table = f"wei_lab_sander_IPF.{group}_{model}_symptom_pair_cooccurrence"
        df_co = spark.table(co_table)

        df_top = (
            df_co.filter(F.col("distinct_person_count") >= 10)
                 .select("symptom_1", "symptom_2")
        )

        top_pairs_list = [(r["symptom_1"], r["symptom_2"]) for r in df_top.collect()]

        # ----------------------------
        # 2. Load weekly data and add week_index
        # ----------------------------
        df_week = spark.sql(f"""
            SELECT *
            FROM wei_lab_sander_IPF.{group}_group_{model}_weekly_ss_umls_code
            WHERE before_after_dx = 'before'
        """)

        df_week_unique = df_week.select("person_id", "week_start").distinct()

        w_idx = Window.partitionBy("person_id").orderBy("week_start")
        df_week_indexed = (
            df_week_unique
            .withColumn("week_index", (F.row_number().over(w_idx) - 1).cast("int"))
        )

        df_week = df_week.join(
            df_week_indexed,
            on=["person_id", "week_start"],
            how="left"
        )

        # ----------------------------
        # 3. Build joint_state for each symptom pair,
        #    including severity and week_start
        # ----------------------------
        all_results = []

        for s1, s2 in top_pairs_list:

            # Build basic joint state
            df_pair_week = (
                df_week.filter(F.col("umls_name").isin([s1, s2]))
                       .groupBy("person_id", "week_index", "week_start")
                       .pivot("umls_name", [s1, s2])
                       .agg(F.max("code"))
                       .fillna(0)
                       .withColumn(
                           "joint_state",
                           F.concat_ws("_", F.col(s1), F.col(s2))
                       )
            )

            # Next-week joint state
            w_next = Window.partitionBy("person_id").orderBy("week_index")
            df_pair_week = df_pair_week.withColumn(
                "next_state",
                F.lead("joint_state").over(w_next)
            )
            df_pair_week = df_pair_week.filter(F.col("next_state").isNotNull())

            # Add current-week severity
            df_sev_cur = (
                df_week.filter(F.col("umls_name").isin([s1, s2]))
                       .select("person_id", "week_index", "umls_name", "severity")
            )

            df_sev_cur_pivot = (
                df_sev_cur.groupBy("person_id", "week_index")
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
                df_week.filter(
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
        # 4. Merge and save
        # ----------------------------
        df_final = all_results[0]
        for df in all_results[1:]:
            df_final = df_final.unionByName(df)

        df_final.write.mode("overwrite").saveAsTable(
            f"wei_lab_sander_IPF.{group}_{model}_pfsa_pair_with_symptoms"
        )

        print("Done!")


# COMMAND ----------

# MAGIC %sql
# MAGIC select 'case' as group, count(*),count(distinct person_id) from wei_lab_sander_IPF.case_semmeddb_pfsa_pair_with_symptoms where joint_state = '1_1'
# MAGIC union all
# MAGIC select 'control' as group, count(*),count(distinct person_id) from wei_lab_sander_IPF.control_semmeddb_pfsa_pair_with_symptoms where joint_state = '1_1'

# COMMAND ----------

# MAGIC %md
# MAGIC ### Important score - single sympton

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

model_list = ["claude", "ds", "ro", "o3"]

for model in model_list:
    print("Processing model:", model)

    # Load IPF and Control group data
    df_ipf = spark.sql(f"""
        SELECT person_id, umls_name
        FROM wei_lab_sander_IPF.case_group_{model}_weekly_ss_umls_code
        WHERE code = 1 AND before_after_dx = 'before'
    """)

    df_ctrl = spark.sql(f"""
        SELECT person_id, umls_name
        FROM wei_lab_sander_IPF.control_group_{model}_weekly_ss_umls_code
        WHERE code = 1 AND before_after_dx = 'before'
    """)

    # Count number of patients with each symptom in the IPF group
    ipf_freq = (
        df_ipf.groupBy("umls_name")
              .agg(F.countDistinct("person_id").alias("freq_ipf"))
    )

    # Count number of patients with each symptom in the Control group
    ctrl_freq = (
        df_ctrl.groupBy("umls_name")
               .agg(F.countDistinct("person_id").alias("freq_ctrl"))
    )

    # Merge and compute importance score
    df_score = (
        ipf_freq.join(ctrl_freq, on="umls_name", how="outer")
                .fillna(0)
                .withColumn(
                    "importance_score",
                    F.log2((F.col("freq_ipf") + 5) / (F.col("freq_ctrl") + 5))
                )
    )

    # Save results
    df_score.write.mode("overwrite").saveAsTable(
        f"wei_lab_sander_IPF.ss_{model}_importance_scores"
    )

    print("Done!")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_IPF.ss_semmeddb_importance_scores order by importance_score desc

# COMMAND ----------

# MAGIC %md
# MAGIC ### Final weight (importance score with frequency_within_patient)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Load weekly symptom table
for group in group_list:
    for model in model_list:
        print(f"Processing group: {group}, processing model: {model}")
        print("Loading data...")

        df = spark.sql(f"""
            SELECT *
            FROM wei_lab_sander_IPF.{group}_group_{model}_weekly_ss_umls_code
            WHERE before_after_dx = 'before'
        """)

        # Load importance scores
        imp_df = spark.table(
            f"wei_lab_sander_IPF.ss_{model}_importance_scores"
        )  # columns: umls_name, importance_score

        # Only consider weeks where the symptom actually appears (code = 1)
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
            F.col("importance_score") * F.log1p(F.col("freq_within_patient"))
        )

        # Save results
        df_score.select(
            "person_id",
            "umls_name",
            "freq_within_patient",
            "importance_score",
            "umls_weight"
        ).write.mode("overwrite").saveAsTable(
            f"wei_lab_sander_IPF.{group}_{model}_patient_symptom_weight"
        )

        print("Done!")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_IPF.case_semmeddb_patient_symptom_weight

# COMMAND ----------

# MAGIC %md
# MAGIC ### Having final weight for pt's sign and symptom

# COMMAND ----------

from pyspark.sql.functions import split, col, sum as F_sum

for group in group_list:
    for model in model_list:

        print(f"Processing group: {group}, Processing model: {model}")
        print("Loading data...")

        # Split symptom_pair → s1, s2
        pair_df = spark.table(
            f"wei_lab_sander_IPF.{group}_{model}_pfsa_pair_with_symptoms"
        )

        pair_df_split = (
            pair_df
            .withColumn("s1", split("symptom_pair", "\\|")[0])
            .withColumn("s2", split("symptom_pair", "\\|")[1])
        )

        # Load patient-level final weights, and rename columns separately for s1 / s2
        weight_df = spark.table(
            f"wei_lab_sander_IPF.{group}_{model}_patient_symptom_weight"
        )

        # Prepare s1 columns
        weight_s1 = weight_df.select(
            col("person_id"),
            col("umls_name").alias("s1"),
            col("umls_weight").alias("umls_weight_1")
        )

        # Prepare s2 columns
        weight_s2 = weight_df.select(
            col("person_id"),
            col("umls_name").alias("s2"),
            col("umls_weight").alias("umls_weight_2")
        )

        # Join s1 weight
        pair_with_w1 = pair_df_split.join(
            weight_s1,
            on=["person_id", "s1"],
            how="left"
        )

        # Join s2 weight
        pair_with_w2 = pair_with_w1.join(
            weight_s2,
            on=["person_id", "s2"],
            how="left"
        )

        # 6️⃣ Extract all symptoms (s1 / s2) with their weights → unify into a single column
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

        # 7️⃣ Merge and deduplicate
        # (avoid double-counting the same symptom across multiple pairs)
        all_symptoms_df = (
            symptom_1_df
            .unionByName(symptom_2_df)
            .dropDuplicates(["person_id", "umls"])
        )

        # 8️⃣ Compute total symptom weight per patient
        patient_symptom_weight_df = (
            all_symptoms_df
            .groupBy("person_id")
            .agg(F_sum("weight").alias("total_symptom_weight"))
        )

        # Save results
        patient_symptom_weight_df.write.mode("overwrite").saveAsTable(
            f"wei_lab_sander_IPF.{group}_{model}_pfsa_pair_pt_total_weight"
        )

        print("Done!")


# COMMAND ----------

# MAGIC %sql
# MAGIC select * from wei_lab_sander_IPF.case_semmeddb_pfsa_pair_pt_total_weight order by total_symptom_weight desc

# COMMAND ----------

# MAGIC %md
# MAGIC ### AUC/ROC curve
# MAGIC - Feature set for AUC/ROC Curve

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

for group in group_list:
    for model in model_list:

        print(f"Processing group: {group}, Processing model: {model}")
        print("Loading data...")

        # Load original symptom pair data
        # (only pre-diagnosis data and joint_state = '1_1')
        pair_df = spark.table(
            f"wei_lab_sander_IPF.{group}_{model}_pfsa_pair_with_symptoms"
        )

        # Filter pre-diagnosis records with joint_state = '1_1'
        filtered = pair_df.filter(F.col("joint_state") == "1_1")

        # Count occurrences per person per symptom pair
        # (number of distinct weeks)
        pair_freq = (
            filtered
            .groupBy("person_id", "symptom_pair")
            .agg(
                F.countDistinct("week_start").alias("pair_week_distinct")
            )
        )

        # Count occurrences where severity is "worsen"
        worsen_count = (
            filtered
            .filter(
                (F.col("severity_1") == "worsen") |
                (F.col("severity_2") == "worsen")
            )
            .groupBy("person_id", "symptom_pair")
            .agg(
                F.count("week_start").alias("worsen_count")
            )
        )

        output = (
            pair_freq
            .join(
                worsen_count,
                on=["person_id", "symptom_pair"],
                how="left"
            )
            .fillna(0, subset=["worsen_count"])
        )

        # Save results
        output.write.mode("overwrite").saveAsTable(
            f"wei_lab_sander_IPF.{group}_{model}_pfsa_pair_with_major_features"
        )

        print("Done!")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Bootstrap version

# COMMAND ----------

# =====================================================
# Imports
# =====================================================
from pyspark.sql import functions as F
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)

import matplotlib.pyplot as plt

# =====================================================
# Label mapping
# =====================================================
label_map = {
    "allmodel": "Ensemble",
    "gemini": "Gemini-2.5 Pro",
    "claude": "Claude-Sonnet-4",
    "ds": "DeepSeek R1",
    "ro": "Rarediseases.org",
    "o3": "OpenAI o3"
}

# =====================================================
# Model list
# =====================================================
models = ["allmodel", "gemini", "claude", "ds", "ro", "o3"]
versions = [""]

FEATURE_COLS = [
    "normalized_weight",
    "pair_week_distinct",
    "worsen_count"
]

# =====================================================
# Stratified OOB Bootstrap Function
# =====================================================
def stratified_oob_bootstrap(
    pdf,
    n_bootstrap=100,
    random_state=42,
    min_oob_per_class=20
):
    rng = np.random.default_rng(random_state)

    pdf_case = pdf[pdf["label"] == 1].reset_index(drop=True)
    pdf_ctrl = pdf[pdf["label"] == 0].reset_index(drop=True)

    auc_list, cutoff_list = [], []
    precision_list, recall_list, f1_list = [], [], []

    for _ in range(n_bootstrap):

        # ---- Stratified bootstrap sampling ----
        idx_case = rng.integers(0, len(pdf_case), len(pdf_case))
        idx_ctrl = rng.integers(0, len(pdf_ctrl), len(pdf_ctrl))

        train_df = pd.concat(
            [pdf_case.iloc[idx_case], pdf_ctrl.iloc[idx_ctrl]],
            axis=0
        )

        oob_case = pdf_case.drop(idx_case, errors="ignore")
        oob_ctrl = pdf_ctrl.drop(idx_ctrl, errors="ignore")

        if len(oob_case) < min_oob_per_class or len(oob_ctrl) < min_oob_per_class:
            continue

        oob_df = pd.concat([oob_case, oob_ctrl], axis=0)

        # ---- Train LR on bootstrap sample ----
        X_train = train_df[FEATURE_COLS].values
        y_train = train_df["label"].values

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)

        # ---- Determine cutoff on training only ----
        y_train_prob = clf.predict_proba(X_train)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_train, y_train_prob)
        cutoff = thresholds[np.argmax(tpr - fpr)]

        # ---- Evaluate on OOB ----
        X_oob = oob_df[FEATURE_COLS].values
        y_oob = oob_df["label"].values
        y_oob_prob = clf.predict_proba(X_oob)[:, 1]

        auc_list.append(roc_auc_score(y_oob, y_oob_prob))
        cutoff_list.append(cutoff)

        y_oob_pred = (y_oob_prob >= cutoff).astype(int)
        precision_list.append(precision_score(y_oob, y_oob_pred))
        recall_list.append(recall_score(y_oob, y_oob_pred))
        f1_list.append(f1_score(y_oob, y_oob_pred))

    return {
        "AUC": np.array(auc_list),
        "Cutoff": np.array(cutoff_list),
        "Precision": np.array(precision_list),
        "Recall": np.array(recall_list),
        "F1": np.array(f1_list)
    }

# =====================================================
# Containers
# =====================================================
roc_data = []
bootstrap_summary = {}
conf_matrices = {}

# =====================================================
# Main Loop
# =====================================================
for model in models:
    for version in versions:

        print(f"\n==============================")
        print(f"Running model: {label_map[model]}")
        print(f"==============================")

        # -----------------------------
        # CASE
        # -----------------------------
        df_case_pair = (
            spark.table(f"wei_lab_sander_IPF.case_{model}_pfsa_pair_with_major_features")
            .groupBy("person_id")
            .agg(
                F.sum("pair_week_distinct").alias("pair_week_distinct"),
                F.sum("worsen_count").alias("worsen_count")
            )
        )

        df_case_symptom = spark.table(
            f"wei_lab_sander_IPF.case_{model}_pfsa_pair_pt_total_weight{version}"
        )

        df_case = (
            df_case_pair.join(df_case_symptom, on="person_id", how="left")
            .withColumn(
                "normalized_weight",
                F.col("total_symptom_weight") / (F.col("pair_week_distinct") + 1)
            )
            .withColumn("label", F.lit(1))
        )

        # -----------------------------
        # CONTROL
        # -----------------------------
        df_ctrl_pair = (
            spark.table(f"wei_lab_sander_IPF.control_{model}_pfsa_pair_with_major_features")
            .groupBy("person_id")
            .agg(
                F.sum("pair_week_distinct").alias("pair_week_distinct"),
                F.sum("worsen_count").alias("worsen_count")
            )
        )

        df_ctrl_symptom = spark.table(
            f"wei_lab_sander_IPF.control_{model}_pfsa_pair_pt_total_weight{version}"
        )

        df_ctrl = (
            df_ctrl_pair.join(df_ctrl_symptom, on="person_id", how="left")
            .withColumn(
                "normalized_weight",
                F.col("total_symptom_weight") / (F.col("pair_week_distinct") + 1)
            )
            .withColumn("label", F.lit(0))
        )

        # -----------------------------
        # Merge + Pandas
        # -----------------------------
        pdf = (
            df_case.unionByName(df_ctrl)
            .toPandas()
            .dropna()
            .reset_index(drop=True)
        )

        # -----------------------------
        # Stratified OOB Bootstrap
        # -----------------------------
        boot = stratified_oob_bootstrap(pdf, n_bootstrap=100)
        bootstrap_summary[model] = boot

        auc_ci = np.percentile(boot["AUC"], [2.5, 97.5])
        f1_ci  = np.percentile(boot["F1"],  [2.5, 97.5])

        print(
            f"AUC = {boot['AUC'].mean():.3f} "
            f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] | "
            f"F1 = {boot['F1'].mean():.3f} "
            f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]"
        )

        # -----------------------------
        # ROC (fit once on full data, visualization only)
        # -----------------------------
        X_full = pdf[FEATURE_COLS].values
        y_full = pdf["label"].values

        clf_full = LogisticRegression(max_iter=1000)
        clf_full.fit(X_full, y_full)
        y_prob_full = clf_full.predict_proba(X_full)[:, 1]

        fpr, tpr, _ = roc_curve(y_full, y_prob_full)
        roc_data.append((model, fpr, tpr))

# =====================================================
# Plot ROC Curves (visual only)
# =====================================================
plt.figure(figsize=(10, 6))
colors = ["tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]

for i, (model, fpr, tpr) in enumerate(roc_data):
    plt.plot(
        fpr, tpr,
        color=colors[i % len(colors)],
        linewidth=2,
        label=label_map.get(model, model)
    )

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison by LLM Source (SIS, Stratified OOB Bootstrap)")
plt.legend(loc="upper left", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()


# COMMAND ----------

rows = []

for model, boot in bootstrap_summary.items():

    def mean_ci(arr):
        return arr.mean(), np.percentile(arr, [2.5, 97.5])

    auc_m, auc_ci = mean_ci(boot["AUC"])
    cut_m, cut_ci = mean_ci(boot["Cutoff"])
    p_m, p_ci = mean_ci(boot["Precision"])
    r_m, r_ci = mean_ci(boot["Recall"])
    f1_m, f1_ci = mean_ci(boot["F1"])

    rows.append({
        "Model": label_map.get(model, model),

        "AUC": f"{auc_m:.3f} [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]",
        "Cutoff": f"{cut_m:.3f} [{cut_ci[0]:.3f}, {cut_ci[1]:.3f}]",

        "Precision": f"{p_m:.3f} [{p_ci[0]:.3f}, {p_ci[1]:.3f}]",
        "Recall": f"{r_m:.3f} [{r_ci[0]:.3f}, {r_ci[1]:.3f}]",
        "F1": f"{f1_m:.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]",
    })

bootstrap_table = pd.DataFrame(rows)
bootstrap_table



# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# Label mapping (case-insensitive)
# =====================================================
label_map = {
    "ro": "Rarediseases.org",
    "ds": "DeepSeek-R1",
    "o3": "OpenAI o3",
    "claude": "Claude-Sonnet-4",
    "gemini": "Gemini-2.5 Pro",
    "allmodel": "Ensemble"
}

# =====================================================
# Use bootstrap_summary ONLY (SIS scheme)
# =====================================================
models = list(bootstrap_summary.keys())

# Bootstrap point estimates (means)
auc_scores = {m: bootstrap_summary[m]["AUC"].mean() for m in models}
f1_scores  = {m: bootstrap_summary[m]["F1"].mean()  for m in models}

# 95% bootstrap confidence intervals
auc_ci = {
    m: np.percentile(bootstrap_summary[m]["AUC"], [2.5, 97.5])
    for m in models
}
f1_ci = {
    m: np.percentile(bootstrap_summary[m]["F1"], [2.5, 97.5])
    for m in models
}

# =====================================================
# Plot
# =====================================================
plt.figure(figsize=(8, 6))

markers = ["o", "s", "^", "D", "X", "P"]
colors  = ["red", "blue", "green", "orange", "purple", "brown"]

for i, model in enumerate(models):
    model_key = model.lower()
    display_label = label_map.get(model_key, model)

    # --- Bootstrap CI (draw first) ---
    plt.errorbar(
        auc_scores[model],
        f1_scores[model],
        xerr=[
            [auc_scores[model] - auc_ci[model][0]],
            [auc_ci[model][1] - auc_scores[model]]
        ],
        yerr=[
            [f1_scores[model] - f1_ci[model][0]],
            [f1_ci[model][1] - f1_scores[model]]
        ],
        fmt="none",
        ecolor=colors[i % len(colors)],
        elinewidth=2.2,
        capsize=6,
        alpha=0.85,
        zorder=3
    )

    # --- Point estimate ---
    plt.scatter(
        auc_scores[model],
        f1_scores[model],
        s=160,
        marker=markers[i % len(markers)],
        color=colors[i % len(colors)],
        edgecolors="black",
        linewidths=1.0,
        zorder=4,
        label=display_label
    )

# =====================================================
# Axes, grid, title (axes start at 0)
# =====================================================
plt.xlabel("AUROC")
plt.ylabel("F1 Score")
plt.title("Overall Performance: AUROC vs F1 Across Models (SIS Scheme, Bootstrap)")

plt.grid(True, linestyle="--", alpha=0.35, zorder=1)

# ⬇️ Start both axes from zero
plt.xlim(0.5, max(auc_scores.values()) + 0.03)
plt.ylim(0.3, max(f1_scores.values()) + 0.05)

# =====================================================
# Legend
# =====================================================
plt.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.30),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.show()
