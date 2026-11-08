"""Evaluasi final model ESI pada test set tanpa training ulang.

Membaca data/test.csv, memuat PipelineModel hasil script 03, lalu menyimpan:
- results/evaluation/evaluation_metrics.json
- results/evaluation/predictions.csv

Prediksi ditulis dengan Python CSV writer agar tidak memakai Hadoop
FileOutputCommitter pada Windows.
"""

import csv
import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)
from pyspark.ml.functions import vector_to_array


# Konfigurasi default (tanpa argparse)
TEST_PATH = Path("data/test.csv")
DEFAULT_MODEL_DIR = Path("models")
METADATA_PATH = Path("results/model/model_metadata.json")
OUTPUT_DIR = Path("results/evaluation")
ESI_CLASSES = [1, 2, 3, 4, 5]


def write_predictions_csv(frame, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["actual_esi", "predicted_esi", "probability"])

        selected = frame.select("esi", "predicted_esi", "probability_array")
        for row in selected.toLocalIterator(prefetchPartitions=False):
            probs = json.dumps([float(x) for x in row["probability_array"]])
            writer.writerow([int(row["esi"]), int(row["predicted_esi"]), probs])
            count += 1

    return count


def main():
    for path, label in [
        (TEST_PATH, "Test CSV"),
        (METADATA_PATH, "Metadata model"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} tidak ditemukan: {path.resolve()}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    model_path = Path(metadata.get("model_dir", str(DEFAULT_MODEL_DIR)))

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path.resolve()}. "
            "Pastikan script 03 berhasil menyimpan model terlebih dahulu."
        )

    spark = (
        SparkSession.builder
        .appName("Hong-ESI-Evaluation")
        .master("local[2]")
        .config("spark.driver.memory", "5g")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )
    predictions = None

    try:
        spark.sparkContext.setLogLevel("WARN")
        print(f"Membaca test set: {TEST_PATH}")

        raw_test = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .option("mode", "PERMISSIVE")
            .option("nullValue", "")
            .csv(str(TEST_PATH))
        )

        if "esi" not in raw_test.columns:
            raise ValueError("Kolom target 'esi' tidak ditemukan pada test.csv.")

        # Jika model 03 memakai nama kolom internal, gunakan mapping dari metadata.
        # Jika tidak, gunakan nama kolom asli seperti pada versi 03_model.py awal.
        mapping = metadata.get("column_mapping")
        if isinstance(mapping, dict) and mapping:
            expected_cols = list(mapping.values())

            # Jangan gunakan select(*expected_cols) untuk nama kolom mentah:
            # Spark menafsirkan titik (.) sebagai akses nested field.
            # CSV train dan test berasal dari schema yang sama, sehingga validasi
            # urutan header lalu toDF() aman untuk mengganti nama berdasarkan posisi.
            if raw_test.columns != expected_cols:
                missing = [c for c in expected_cols if c not in raw_test.columns]
                extra = [c for c in raw_test.columns if c not in expected_cols]
                raise ValueError(
                    "Urutan/nama kolom test.csv berbeda dari data training. "
                    f"Kolom hilang: {missing[:10]}; kolom tambahan: {extra[:10]}. "
                    "Pastikan train.csv dan test.csv berasal dari split dataset "
                    "yang sama dan memiliki urutan header identik."
                )

            test = raw_test.toDF(*list(mapping.keys()))
        else:
            test = raw_test

        test = (
            test.withColumn("esi", F.col("esi").cast("double"))
            .filter(F.col("esi").isin(1.0, 2.0, 3.0, 4.0, 5.0))
            .withColumn("label", (F.col("esi") - F.lit(1.0)).cast("double"))
            .withColumn("esi", F.col("esi").cast("int"))
        )

        test_rows = test.count()
        if test_rows == 0:
            raise ValueError("Tidak ada baris test dengan target ESI valid 1–5.")

        print(f"Jumlah baris test valid: {test_rows:,}")
        print(f"Memuat PipelineModel: {model_path}")

        # Script 03 menyimpan PipelineModel, bukan LogisticRegressionModel saja.
        model = PipelineModel.load(str(model_path))
        predictions = (
            model.transform(test)
            .withColumn(
                "predicted_esi",
                (F.col("prediction") + F.lit(1)).cast("int")
            )
            .withColumn(
                "probability_array",
                vector_to_array(F.col("probability"))
            )
            .cache()
        )

        # Evaluator menggunakan label internal 0–4.
        metrics = {}
        for metric_name in [
            "accuracy", "f1", "weightedPrecision", "weightedRecall"
        ]:
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction",
                metricName=metric_name,
            )
            metrics[metric_name] = float(evaluator.evaluate(predictions))

        # Confusion matrix dihitung dengan label ESI asli 1–5.
        grouped = (
            predictions.groupBy("esi", "predicted_esi")
            .count()
            .collect()
        )
        counts = {
            (int(r["esi"]), int(r["predicted_esi"])): int(r["count"])
            for r in grouped
        }

        per_class = []
        for cls in ESI_CLASSES:
            tp = counts.get((cls, cls), 0)
            fp = sum(v for (a, p), v in counts.items() if p == cls and a != cls)
            fn = sum(v for (a, p), v in counts.items() if a == cls and p != cls)
            support = tp + fn
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / support if support else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            per_class.append({
                "esi": cls,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

        metrics["macroPrecision"] = sum(x["precision"] for x in per_class) / 5
        metrics["macroRecall"] = sum(x["recall"] for x in per_class) / 5
        metrics["macroF1"] = sum(x["f1"] for x in per_class) / 5

        # AUROC macro One-vs-Rest berdasarkan probabilitas masing-masing kelas.
        per_class_auc = []
        for cls in ESI_CLASSES:
            binary = (
                predictions
                .withColumn(
                    "binary_label",
                    F.when(F.col("esi") == cls, 1.0).otherwise(0.0)
                )
                .withColumn("class_score", F.col("probability_array")[cls - 1])
            )
            has_positive = binary.filter(F.col("binary_label") == 1.0).limit(1).count()
            has_negative = binary.filter(F.col("binary_label") == 0.0).limit(1).count()

            if has_positive and has_negative:
                auc_eval = BinaryClassificationEvaluator(
                    labelCol="binary_label",
                    rawPredictionCol="class_score",
                    metricName="areaUnderROC",
                )
                auc = float(auc_eval.evaluate(binary))
            else:
                auc = None
            per_class_auc.append({"esi": cls, "auc_roc": auc})

        auc_valid = [x["auc_roc"] for x in per_class_auc if x["auc_roc"] is not None]
        metrics["macroAUROC_ovr"] = (
            sum(auc_valid) / len(auc_valid) if auc_valid else None
        )

        confusion = [
            {"actual": a, "predicted": p, "count": n}
            for (a, p), n in sorted(counts.items())
        ]

        result = {
            "test_rows": test_rows,
            "metrics": metrics,
            "per_class_metrics": per_class,
            "per_class_auroc_ovr": per_class_auc,
            "confusion_matrix": confusion,
            "label_mapping": {"0": 1, "1": 2, "2": 3, "3": 4, "4": 5},
            "model_type": "Spark PipelineModel",
            "test_file": str(TEST_PATH),
            "model_dir": str(model_path),
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metrics_file = OUTPUT_DIR / "evaluation_metrics.json"
        metrics_file.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        predictions_file = OUTPUT_DIR / "predictions.csv"
        written = write_predictions_csv(predictions, predictions_file)

        print("\n=== EVALUASI SELESAI ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"Metrics tersimpan: {metrics_file.resolve()}")
        print(f"Prediksi tersimpan: {predictions_file.resolve()}")
        print(f"Jumlah prediksi ditulis: {written:,}")

    finally:
        if predictions is not None:
            predictions.unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
