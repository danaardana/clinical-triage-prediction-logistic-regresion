"""Train Multinomial Logistic Regression untuk klasifikasi ESI dengan fitur pre-triage.

Fitur yang digunakan:
- Demografis: age dan gender (dengan alternatif nama in_years/sex).
- Chief complaint: seluruh kolom berawalan cc_, kecuali cc_other.
- Tanda vital saat triage: HR, SBP, DBP, RR, SpO2, perangkat O2, dan temperatur.

Preprocessing dilakukan hanya pada fitur terpilih:
- imputasi median untuk fitur numerik;
- StringIndexer + OneHotEncoder untuk fitur kategorikal;
- VectorAssembler;
- label ESI 1-5 dipetakan menjadi label internal 0-4.

Nama kolom internal dibuat aman untuk Spark SQL. Metadata menyimpan mapping
seluruh kolom agar kompatibel dengan script evaluasi yang membaca CSV lengkap.
"""

import json
from pathlib import Path

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import NumericType
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)


# Konfigurasi
TRAIN_PATH = Path("data/train.csv")
MODEL_DIR = Path("models/logistic_regression")
OUTPUT_DIR = Path("results/model")
REG_PARAM = 0.0
MAX_ITER = 100

# Kolom demografis yang diprioritaskan; alternatif digunakan jika nama utama tidak ada.
DEMOGRAPHIC_CANDIDATES = {
    "age": ["age", "in_years"],
    "gender": ["gender", "sex"],
}

# Nama kolom tanda vital pada dataset Hong et al.
VITAL_FEATURES = [
    "triage_vital_hr",
    "triage_vital_sbp",
    "triage_vital_dbp",
    "triage_vital_rr",
    "triage_vital_o2",
    "triage_vital_o2_device",
    "triage_vital_temp",
]


def select_pretriage_features(original_columns):
    """Pilih fitur berdasarkan nama kolom asli, sebelum normalisasi nama Spark."""
    columns_set = set(original_columns)

    demographic_cols = []
    missing_demographics = []
    for group, candidates in DEMOGRAPHIC_CANDIDATES.items():
        selected = next((name for name in candidates if name in columns_set), None)
        if selected is None:
            missing_demographics.append(f"{group} ({' atau '.join(candidates)})")
        else:
            demographic_cols.append(selected)

    missing_vitals = [name for name in VITAL_FEATURES if name not in columns_set]
    if missing_demographics or missing_vitals:
        details = []
        if missing_demographics:
            details.append("Demografis tidak ditemukan: " + ", ".join(missing_demographics))
        if missing_vitals:
            details.append("Tanda vital tidak ditemukan: " + ", ".join(missing_vitals))
        raise ValueError(
            "Struktur kolom dataset tidak sesuai asumsi seleksi fitur. "
            + " | ".join(details)
        )

    # CC dipilih berdasarkan prefix; cc_other dikecualikan secara eksplisit.
    cc_cols = sorted(
        [
            name for name in original_columns
            if name.lower().startswith("cc_") and name.lower() != "cc_other"
        ],
        key=str.lower,
    )
    if not cc_cols:
        raise ValueError(
            "Tidak ditemukan kolom chief complaint berawalan 'cc_' "
            "(selain cc_other). Periksa header data/train.csv."
        )

    selected_original = demographic_cols + cc_cols + VITAL_FEATURES
    # Hilangkan duplikasi sambil mempertahankan urutan.
    selected_original = list(dict.fromkeys(selected_original))
    groups = {
        "demographic": demographic_cols,
        "chief_complaint": cc_cols,
        "vital_signs": VITAL_FEATURES,
    }
    return selected_original, groups


def main():
    input_path = TRAIN_PATH
    if not input_path.exists():
        raise FileNotFoundError(
            f"File training tidak ditemukan: {input_path.resolve()}\n"
            "Jalankan 02_split_data.py terlebih dahulu."
        )

    spark = (
        SparkSession.builder
        .appName("Hong-ESI-Multinomial-Logistic-Regression")
        .master("local[2]")
        .config("spark.driver.memory", "5g")
        .config("spark.sql.shuffle.partitions", "32")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        print(f"Membaca data training: {input_path}")
        raw = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .option("mode", "PERMISSIVE")
            .option("nullValue", "")
            .csv(str(input_path))
        )

        original_columns = raw.columns
        if "esi" not in original_columns:
            raise ValueError("Kolom target 'esi' tidak ditemukan dalam CSV.")
        if len(original_columns) != len(set(original_columns)):
            raise ValueError(
                "CSV memiliki nama kolom duplikat. Periksa header data/train.csv."
            )

        selected_original, feature_groups = select_pretriage_features(original_columns)

        # Normalisasi seluruh nama kolom. Ini juga mencegah titik/koma pada
        # nama kolom asli ditafsirkan sebagai ekspresi nested-field oleh Spark.
        safe_columns = [
            "esi" if col == "esi" else f"feature_{i:04d}"
            for i, col in enumerate(original_columns)
        ]
        column_mapping = {
            safe: original
            for original, safe in zip(original_columns, safe_columns)
        }
        original_to_safe = {
            original: safe
            for original, safe in zip(original_columns, safe_columns)
        }

        train = raw.toDF(*safe_columns)

        # Hanya kolom fitur terpilih yang masuk ke pipeline.
        selected_internal = [original_to_safe[name] for name in selected_original]
        train = train.select("esi", *selected_internal)

        train = (
            train.withColumn("esi", F.col("esi").cast("double"))
            .filter(F.col("esi").isin(1.0, 2.0, 3.0, 4.0, 5.0))
            .withColumn("label", (F.col("esi") - F.lit(1.0)).cast("double"))
        )

        training_rows = train.count()
        if training_rows == 0:
            raise ValueError("Tidak ada baris training dengan ESI valid (1-5).")

        feature_cols = selected_internal
        numeric_cols = [
            field.name
            for field in train.schema.fields
            if field.name in feature_cols and isinstance(field.dataType, NumericType)
        ]
        categorical_cols = [c for c in feature_cols if c not in numeric_cols]

        if not numeric_cols and not categorical_cols:
            raise ValueError("Tidak ditemukan fitur numerik maupun kategorikal.")

        print(f"Jumlah baris training : {training_rows:,}")
        print(f"Fitur demografis      : {len(feature_groups['demographic'])}")
        print(f"Fitur chief complaint: {len(feature_groups['chief_complaint'])}")
        print(f"Fitur tanda vital     : {len(feature_groups['vital_signs'])}")
        print(f"Total fitur terpilih  : {len(selected_internal)}")
        print(f"Fitur numerik         : {len(numeric_cols)}")
        print(f"Fitur kategorikal     : {len(categorical_cols)}")
        print("cc_other dan kolom di luar kelompok pre-triage tidak digunakan.")

        stages = []
        assembled_inputs = []

        if numeric_cols:
            imputed_cols = [f"{c}__imputed" for c in numeric_cols]
            stages.append(
                Imputer(
                    inputCols=numeric_cols,
                    outputCols=imputed_cols,
                    strategy="median",
                )
            )
            assembled_inputs.extend(imputed_cols)

        if categorical_cols:
            indexed_cols = [f"{c}__indexed" for c in categorical_cols]
            encoded_cols = [f"{c}__ohe" for c in categorical_cols]

            for input_col, output_col in zip(categorical_cols, indexed_cols):
                stages.append(
                    StringIndexer(
                        inputCol=input_col,
                        outputCol=output_col,
                        handleInvalid="keep",
                    )
                )

            stages.append(
                OneHotEncoder(
                    inputCols=indexed_cols,
                    outputCols=encoded_cols,
                    handleInvalid="keep",
                )
            )
            assembled_inputs.extend(encoded_cols)

        stages.append(
            VectorAssembler(
                inputCols=assembled_inputs,
                outputCol="features",
                handleInvalid="error",
            )
        )

        stages.append(
            LogisticRegression(
                featuresCol="features",
                labelCol="label",
                predictionCol="prediction",
                probabilityCol="probability",
                family="multinomial",
                maxIter=MAX_ITER,
                regParam=REG_PARAM,
                standardization=True,
            )
        )

        pipeline = Pipeline(stages=stages)
        print("Memulai preprocessing dan training Logistic Regression...")
        pipeline_model = pipeline.fit(train)

        MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
        pipeline_model.write().overwrite().save(str(MODEL_DIR))

        lr_model = pipeline_model.stages[-1]
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        metadata = {
            "algorithm": "Multinomial Logistic Regression",
            "family": "multinomial",
            "input_file": str(input_path),
            "model_dir": str(MODEL_DIR),
            "target": "esi",
            "original_classes": [1, 2, 3, 4, 5],
            "internal_label_mapping": {
                "0": 1, "1": 2, "2": 3, "3": 4, "4": 5
            },
            "max_iter": MAX_ITER,
            "reg_param": REG_PARAM,
            "standardization": True,
            "imputation": "median for selected numeric features",
            "categorical_encoding": "StringIndexer + OneHotEncoder",
            "feature_selection_method": "rule-based pre-triage feature groups",
            "feature_groups": feature_groups,
            "selected_original_features": selected_original,
            "selected_internal_features": selected_internal,
            "excluded_feature": ["cc_other"],
            "training_rows": training_rows,
            "numeric_feature_count": len(numeric_cols),
            "categorical_feature_count": len(categorical_cols),
            "selected_feature_count": len(selected_internal),
            "num_classes": int(lr_model.numClasses),
            "num_features": int(lr_model.numFeatures),
            "column_mapping": column_mapping,
        }

        metadata_path = OUTPUT_DIR / "model_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n=== TRAINING SELESAI ===")
        print(json.dumps(
            {k: v for k, v in metadata.items() if k != "column_mapping"},
            indent=2,
            ensure_ascii=False,
        ))
        print(f"Model tersimpan di: {MODEL_DIR.resolve()}")
        print(f"Metadata tersimpan di: {metadata_path.resolve()}")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
