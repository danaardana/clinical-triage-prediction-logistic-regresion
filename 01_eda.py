# 01_eda.py - Exploratory Data Analysis dataset triase Hong et al.
# PySpark digunakan untuk membaca dan meringkas dataset besar.
# Pandas/Seaborn hanya digunakan untuk membuat grafik dari hasil ringkasan atau sampel.

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType

# ======================== PENGATURAN ========================
INPUT_PATH = "data/dataset_original.parquet"
OUTPUT_DIR = Path("results/eda")
TARGET = "esi"
ESI_CLASSES = [1, 2, 3, 4, 5]
SAMPLE_SIZE = 30000

AGE_COL = "age"
GENDER_COL = "gender"
VITAL_COLS = {
    "HR": "triage_vital_hr",
    "RR": "triage_vital_rr",
    "Temperature": "triage_vital_temp",
}


def col(name):
    """Mengakses nama kolom secara utuh, termasuk jika ada titik atau koma."""
    return F.col("`" + name.replace("`", "``") + "`")


def save_plot(filename):
    """Merapikan dan menyimpan grafik."""
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "charts" / filename, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "charts").mkdir(parents=True, exist_ok=True)

    spark = (SparkSession.builder
             .appName("Hong-ESI-EDA")
             .config("spark.sql.shuffle.partitions", "4")
             .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    try:
        # 1. Membaca dataset dan melihat struktur dasarnya.
        df = spark.read.parquet(INPUT_PATH)
        columns = df.columns
        row_count = df.count()

        print("\n=== DATASET ===")
        print(f"Jumlah baris : {row_count:,}")
        print(f"Jumlah kolom : {len(columns):,}")
        df.printSchema()
        df.select(*columns[:12]).show(5, truncate=True)

        # Nama kolom ditetapkan langsung sesuai schema dataset Parquet.
        required_cols = [TARGET, AGE_COL, GENDER_COL, *VITAL_COLS.values()]
        missing_required = [name for name in required_cols if name not in columns]
        if missing_required:
            raise ValueError(f"Kolom wajib tidak ditemukan dalam dataset: {missing_required}")

        age_col = AGE_COL
        gender_col = GENDER_COL
        vital_cols = VITAL_COLS.copy()

        # Kenali kolom numerik dari schema agar pemeriksaan NaN hanya diterapkan
        # pada kolom yang memang bertipe angka.
        numeric_cols = {
            field.name for field in df.schema.fields
            if isinstance(field.dataType, NumericType)
        }

        # 2. Menghitung missing value per 40 kolom agar agregasi tidak terlalu besar.
        missing = []
        for start in range(0, len(columns), 40):
            group = columns[start:start + 40]
            expressions = []
            for i, name in enumerate(group):
                value = col(name).isNull()
                if name in numeric_cols:
                    value = value | F.isnan(col(name).cast("double"))
                expressions.append(F.sum(F.when(value, 1).otherwise(0)).alias(f"m{i}"))
            values = df.agg(*expressions).first().asDict()
            missing.extend({"feature": name, "count": int(values.get(f"m{i}") or 0)}
                           for i, name in enumerate(group))
        missing = [x for x in missing if x["count"] > 0]

        # 3. Distribusi tingkat ESI.
        esi_rows = []
        invalid_esi = None
        if TARGET in columns:
            esi_value = col(TARGET).cast("double")
            valid_esi = esi_value.isin([float(x) for x in ESI_CLASSES])
            invalid_esi = df.filter(~F.coalesce(valid_esi, F.lit(False))).count()
            esi_rows = [r.asDict() for r in (
                df.withColumn("_esi", esi_value)
                  .filter(F.col("_esi").isin([float(x) for x in ESI_CLASSES]))
                  .groupBy("_esi").count().orderBy("_esi").collect())]

            esi_pd = pd.DataFrame([{"ESI": int(r["_esi"]), "Jumlah": r["count"]}
                                   for r in esi_rows])
            if not esi_pd.empty:
                plt.figure(figsize=(8, 5))
                sns.barplot(data=esi_pd, x="ESI", y="Jumlah", order=ESI_CLASSES)
                plt.title("Distribusi Pasien berdasarkan ESI")
                plt.xlabel("Emergency Severity Index")
                plt.ylabel("Jumlah Pasien")
                save_plot("01_distribusi_esi.png")

        # 4. Distribusi gender keseluruhan dalam pie chart persentase.
        gender_rows = []
        if gender_col:
            gender_rows = [r.asDict() for r in (
                df.groupBy(F.when(col(gender_col).isNull(), "Missing")
                           .otherwise(col(gender_col).cast("string")).alias("gender"))
                  .count().orderBy(F.desc("count")).collect())]
            gender_pd = pd.DataFrame(gender_rows)
            if not gender_pd.empty:
                plt.figure(figsize=(7, 7))
                plt.pie(gender_pd["count"], labels=gender_pd["gender"],
                        autopct="%1.1f%%", startangle=90)
                plt.title("Persentase Gender Pasien")
                save_plot("02_persentase_gender.png")

        # 5. Mengambil sampel kecil untuk boxplot umur dan tanda vital.
        # Pengambilan sampel mencegah seluruh ratusan ribu baris masuk ke Pandas.
        sample_cols = [TARGET]
        if age_col:
            sample_cols.append(age_col)
        if gender_col:
            sample_cols.append(gender_col)
        sample_cols += list(vital_cols.values())
        sample_cols = list(dict.fromkeys(sample_cols))

        sample_pd = pd.DataFrame()
        if TARGET in columns and sample_cols:
            sample_pd = (
                df.select(*[col(x) for x in sample_cols])
                  .withColumn(TARGET, col(TARGET).cast("double"))
                  .filter(col(TARGET).isin([float(x) for x in ESI_CLASSES]))
                  .sample(False, min(1.0, SAMPLE_SIZE / max(row_count, 1)), 42)
                  .limit(SAMPLE_SIZE)
                  .toPandas()
            )
            sample_pd[TARGET] = sample_pd[TARGET].astype(int)

        # 6. Boxplot umur berdasarkan tingkat ESI.
        age_summary = []
        if age_col and not sample_pd.empty:
            sample_pd[age_col] = pd.to_numeric(sample_pd[age_col], errors="coerce")
            age_data = sample_pd[[TARGET, age_col]].dropna()
            age_data = age_data[age_data[age_col] >= 0]
            if not age_data.empty:
                plt.figure(figsize=(9, 5))
                sns.boxplot(data=age_data, x=TARGET, y=age_col, order=ESI_CLASSES)
                plt.title("Distribusi Umur pada Setiap Tingkat ESI")
                plt.xlabel("Tingkat ESI")
                plt.ylabel("Umur (tahun)")
                save_plot("03_boxplot_umur_per_esi.png")

        if age_col:
            age_summary = [r.asDict() for r in (
                df.select(col(TARGET).cast("double").alias("esi"),
                          col(age_col).cast("double").alias("age"))
                  .filter(F.col("esi").isin([float(x) for x in ESI_CLASSES]) &
                          F.col("age").isNotNull() & ~F.isnan(F.col("age")) &
                          (F.col("age") >= 0))
                  .groupBy("esi")
                  .agg(F.count("*").alias("jumlah"),
                       F.round(F.avg("age"), 2).alias("rata_rata_umur"),
                       F.expr("percentile_approx(age, 0.5)").alias("median_umur"))
                  .orderBy("esi").collect())]

        # 7. Komposisi gender pada setiap tingkat ESI (persentase 100%).
        gender_by_esi = []
        if gender_col:
            gender_counts = (
                df.select(col(TARGET).cast("double").alias("esi"),
                          F.coalesce(col(gender_col).cast("string"), F.lit("Missing")).alias("gender"))
                  .filter(F.col("esi").isin([float(x) for x in ESI_CLASSES]))
                  .groupBy("esi", "gender").count()
                  .orderBy("esi", "gender")
            )
            gender_by_esi = [r.asDict() for r in gender_counts.collect()]
            gender_table_pd = pd.DataFrame(gender_by_esi)
            if not gender_table_pd.empty:
                gender_table = pd.crosstab(
                    gender_table_pd["esi"], gender_table_pd["gender"],
                    values=gender_table_pd["count"], aggfunc="sum", normalize="index"
                ) * 100
                gender_table = gender_table.reindex(ESI_CLASSES)
                ax = gender_table.plot(kind="bar", stacked=True, figsize=(9, 5))
                ax.set_title("Persentase Gender pada Setiap Tingkat ESI")
                ax.set_xlabel("Tingkat ESI")
                ax.set_ylabel("Persentase (%)")
                ax.legend(title="Gender", bbox_to_anchor=(1.02, 1), loc="upper left")
                save_plot("04_gender_per_esi.png")

        # 8. Boxplot setiap tanda vital berdasarkan ESI.
        # Setiap tanda vital dibuat terpisah karena satuannya berbeda.
        vital_summary = {}
        for label, vital in vital_cols.items():
            # Statistik dihitung dari seluruh dataset, bukan dari sampel grafik.
            vital_summary[label] = (
                df.select(col(TARGET).cast("double").alias("esi"),
                          col(vital).cast("double").alias("value"))
                  .filter(F.col("esi").isin([float(x) for x in ESI_CLASSES]) &
                          F.col("value").isNotNull() & ~F.isnan(F.col("value")))
                  .groupBy("esi")
                  .agg(F.count("*").alias("jumlah"),
                       F.round(F.avg("value"), 2).alias("mean"),
                       F.min("value").alias("min"),
                       F.max("value").alias("max"))
                  .orderBy("esi").toPandas().to_dict("records")
            )

        # 9. Menyimpan ringkasan hasil EDA.
        result = {
            "dataset": {"input": INPUT_PATH, "rows": row_count, "columns": len(columns)},
            "target_esi": {"invalid_or_missing": invalid_esi, "distribution": esi_rows},
            "missing_values": {"total_missing_cells": sum(x["count"] for x in missing),
                               "by_column": missing},
            "age_by_esi": age_summary,
            "gender_distribution": gender_rows,
            "gender_by_esi": gender_by_esi,
            "vital_by_esi": vital_summary,
            "duplicate_rows": None
        }
        with open(OUTPUT_DIR / "eda_summary.json", "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False, default=str)

        print("\nEDA selesai.")
        print(f"Ringkasan JSON: {(OUTPUT_DIR / 'eda_summary.json').resolve()}")
        print(f"Folder grafik : {(OUTPUT_DIR / 'charts').resolve()}")
        print(f"Kolom vital yang digunakan: {vital_cols}")
        for label, stats in vital_summary.items():
            print(f"\n{label} ({VITAL_COLS[label]}):")
            print(pd.DataFrame(stats).to_string(index=False))

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
