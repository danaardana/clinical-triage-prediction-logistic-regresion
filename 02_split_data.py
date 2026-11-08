"""Split dataset ESI menjadi train/test secara stratified.
Script ini hanya memvalidasi target dan membagi data.
Tidak melakukan imputasi, encoding, scaling, seleksi fitur, atau penghapusan duplikat.
"""

from pathlib import Path
import csv
import json
import os

# Pastikan Hadoop home menunjuk ke folder induk.
os.environ["HADOOP_HOME"] = r"C:\hadoop"

from pyspark.sql import SparkSession, functions as F


# ======================== PENGATURAN ========================
INPUT_PATH = "data/dataset_original.parquet"
OUTPUT_DIR = Path("data")
TARGET = "esi"
ESI_CLASSES = [1, 2, 3, 4, 5]
TRAIN_RATIO = 0.80
SEED = 42

# Membagi output menjadi lebih banyak partisi agar satu partisi
# yang dikirim ke Python tidak terlalu besar.
OUTPUT_PARTITIONS = 32


def write_csv_streaming(frame, filename):
    """Tulis CSV baris demi baris, satu partisi Spark pada satu waktu."""
    output_path = OUTPUT_DIR / filename
    columns = frame.columns

    # Repartition mengecilkan ukuran rata-rata setiap partisi.
    # Ini memerlukan shuffle, tetapi menghindari satu partisi besar
    # ditampung sekaligus oleh toLocalIterator.
    output_frame = frame.repartition(OUTPUT_PARTITIONS)

    written = 0
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(columns)

        for row in output_frame.toLocalIterator(prefetchPartitions=False):
            writer.writerow(row)
            written += 1

    print(f"CSV tersimpan: {output_path} ({written:,} baris)")
    return written


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("ESI_Stratified_Data_Split_Windows")
        .master("local[2]")
        .config("spark.driver.memory", "5g")
        .config("spark.sql.shuffle.partitions", str(OUTPUT_PARTITIONS))
        .config("spark.default.parallelism", str(OUTPUT_PARTITIONS))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("Membaca dataset...")
        df = spark.read.parquet(INPUT_PATH)

        if TARGET not in df.columns:
            raise ValueError(f"Kolom target '{TARGET}' tidak ditemukan.")

        input_count = df.count()
        print(f"Total baris input: {input_count:,}")

        # Validasi target: hanya ESI 1-5 yang masuk proses split.
        esi_numeric = F.col(TARGET).cast("double")
        valid_target = F.coalesce(
            esi_numeric.isin([float(x) for x in ESI_CLASSES]),
            F.lit(False)
        )

        discarded = (
            df.filter(~valid_target)
            .withColumn("_discard_reason", F.lit("invalid_or_missing_target"))
        )

        valid = df.filter(valid_target).withColumn(
            TARGET, esi_numeric.cast("int")
        )

        # Stratified split dilakukan terpisah untuk setiap kelas.
        train_parts = []
        test_parts = []

        for esi_value in ESI_CLASSES:
            print(f"Menyiapkan split ESI {esi_value}...")
            class_df = valid.filter(F.col(TARGET) == esi_value)

            train_class, test_class = class_df.randomSplit(
                [TRAIN_RATIO, 1.0 - TRAIN_RATIO],
                seed=SEED + esi_value
            )

            train_parts.append(train_class)
            test_parts.append(test_class)

        train = train_parts[0]
        test = test_parts[0]

        for part in train_parts[1:]:
            train = train.unionByName(part)

        for part in test_parts[1:]:
            test = test.unionByName(part)

        # Tulis satu CSV per dataset. Tidak menggunakan Spark CSV writer,
        # sehingga tidak melewati Hadoop FileOutputCommitter Windows.
        print("\nMenulis train.csv...")
        train_written = write_csv_streaming(train, "train.csv")

        print("\nMenulis test.csv...")
        test_written = write_csv_streaming(test, "test.csv")

        print("\nMenulis discarded.csv...")
        discarded_written = write_csv_streaming(discarded, "discarded.csv")

        # Ringkasan distribusi kelas; hanya mengumpulkan agregat kecil.
        def distribution(frame):
            rows = (
                frame.groupBy(TARGET)
                .count()
                .orderBy(TARGET)
                .collect()
            )
            return [
                {"esi": int(row[TARGET]), "count": int(row["count"])}
                for row in rows
            ]

        summary = {
            "input": INPUT_PATH,
            "split": "stratified approximate 80:20 using randomSplit per class",
            "seed": SEED,
            "target": TARGET,
            "output_partitions": OUTPUT_PARTITIONS,
            "preprocessing_performed": False,
            "duplicate_removal_performed": False,
            "feature_missing_values_removed": False,
            "rows": {
                "input": input_count,
                "train": train_written,
                "test": test_written,
                "discarded_invalid_or_missing_target": discarded_written,
            },
            "train_distribution": distribution(train),
            "test_distribution": distribution(test),
        }

        summary_path = OUTPUT_DIR / "split_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print("\n=== DATA SPLIT SELESAI ===")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("\nOutput:")
        print(f"- {OUTPUT_DIR / 'train.csv'}")
        print(f"- {OUTPUT_DIR / 'test.csv'}")
        print(f"- {OUTPUT_DIR / 'discarded.csv'}")
        print(f"- {summary_path}")

    finally:
        try:
            spark.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
