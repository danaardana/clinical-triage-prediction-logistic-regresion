"""Konversi RData ke Parquet. Pandas hanya digunakan pada tahap konversi."""
import argparse
from pathlib import Path
import pyreadr

p = argparse.ArgumentParser()
p.add_argument("--input", default="5v_cleandf.rdata")
p.add_argument("--output", default="data/dataset_original.parquet")
a = p.parse_args()
if not Path(a.input).exists():
    raise FileNotFoundError(a.input)

# pyreadr membaca objek RData; ambil objek DataFrame pertama.
objects = pyreadr.read_r(a.input)
df = next((x for x in objects.values() if hasattr(x, "columns")), None)
if df is None:
    raise ValueError("Tidak ditemukan DataFrame dalam RData.")
Path(a.output).parent.mkdir(parents=True, exist_ok=True)

# Parquet adalah format kolumnar yang cocok untuk dibaca Spark.
df.to_parquet(a.output, engine="pyarrow", index=False)
print(f"Berhasil: {len(df):,} baris x {len(df.columns):,} kolom -> {a.output}")
