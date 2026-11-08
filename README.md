# ESI Pre-Triage Classification

Proyek ini mengembangkan dan mengevaluasi model *machine learning* untuk mengklasifikasikan tingkat kegawatan pasien berdasarkan **Emergency Severity Index (ESI) 1–5**. Fokus proyek adalah penggunaan informasi yang tersedia pada tahap awal *pre-triage*, sebelum pemeriksaan lanjutan, tindakan, atau hasil laboratorium tersedia.

## Tujuan

1. Mengolah dataset triase ke format yang dapat diproses dengan Apache Spark.
2. Melakukan *Exploratory Data Analysis* (EDA) untuk memahami struktur, kualitas, dan distribusi data.
3. Membagi data menjadi data latih dan data uji dengan mempertahankan proporsi kelas ESI.
4. Melatih model klasifikasi menggunakan fitur yang tersedia pada tahap pre-triage.
5. Mengevaluasi performa model menggunakan metrik keseluruhan dan metrik per kelas.

> **Catatan ruang lingkup:** Skrip training yang didokumentasikan di sini menjalankan **Multinomial Logistic Regression**

## Dataset

Dataset utama yang digunakan adalah dataset triase dari Hong et al. (2018), dengan file sumber yang diharapkan bernama `5v_cleandf.rdata`.

Ringkasan dataset yang tercatat pada proses EDA:

| Informasi | Nilai |
|---|---:|
| Jumlah baris awal | 560.486 |
| Jumlah kolom | 972 |
| Target klasifikasi | `esi` |
| Kelas target | ESI 1, 2, 3, 4, 5 |
| Baris dengan target tidak valid/kosong | 2.457 |
| Baris dengan target valid | 558.029 |

Distribusi target pada dataset valid:

| Kelas ESI | Jumlah |
|---|---:|
| ESI 1 | 5.271 |
| ESI 2 | 163.534 |
| ESI 3 | 236.229 |
| ESI 4 | 125.003 |
| ESI 5 | 27.992 |

Distribusi tersebut menunjukkan bahwa kelas target tidak seimbang. Karena itu, evaluasi sebaiknya tidak hanya mengandalkan accuracy; metrik macro dan hasil per kelas juga perlu diperhatikan.

**Akses dan lisensi data:** file dataset sumber tidak disertakan dalam repository ini. Pengguna harus memperoleh dataset dari sumber/publikasi yang berwenang dan memastikan ketentuan penggunaan serta sitasinya dipenuhi.

## Fitur pre-triage

Skrip `03_model_pretriage.py` memilih fitur berdasarkan kelompok berikut:

| Kelompok | Fitur |
|---|---|
| Demografis | `age` (atau `in_years`) dan `gender` (atau `sex`) |
| Keluhan utama | Kolom yang diawali `cc_`, kecuali `cc_other` |
| Tanda vital | `triage_vital_hr`, `triage_vital_sbp`, `triage_vital_dbp`, `triage_vital_rr`, `triage_vital_o2`, `triage_vital_o2_device`, `triage_vital_temp` |

Kolom di luar kelompok tersebut tidak dimasukkan ke pipeline model. Pembatasan ini dimaksudkan agar prediksi menggunakan informasi yang relevan dengan tahap awal triase dan mengurangi risiko *data leakage* dari informasi yang baru tersedia setelah proses triase.

Pemilihan fitur berdasarkan nama kolom perlu diverifikasi terhadap dokumentasi dataset. Khususnya, pastikan kolom `cc_` memang merepresentasikan informasi yang sudah tersedia pada saat pre-triage.

## Alur pemrosesan

```text
5v_cleandf.rdata
       |
       v
00_convert.py
       |
       v
data/dataset_original.parquet
       |
       +----------------------+
       |                      |
       v                      v
01_eda.py               02_split_data.py
       |                      |
       v                      v
results/eda/        data/train_csv/ dan data/test_csv/
                              |
                              v
                       03_model_pretriage.py
                              |
                              v
                    models/logistic_regression/
                    results/model/model_metadata.json
                              |
                              v
                       04_evaluation_fixed_v2.py
                              |
                              v
                    results/evaluation/
```

## Struktur repository

Struktur berikut merupakan struktur direktori yang disarankan. Beberapa folder akan dibuat otomatis saat skrip dijalankan.

```text
.
├── 5v_cleandf.rdata
├── 00_convert.py
├── 01_eda.py
├── 02_split_data.py
├── 03_model_pretriage.py
├── 04_evaluation_fixed_v2.py
├── data/
│   ├── dataset_original.parquet
│   ├── train_csv/
│   ├── test_csv/
│   └── discarded_csv/
├── models/
│   └── logistic_regression/
└── results/
    ├── eda/
    ├── model/
    │   └── model_metadata.json
    └── evaluation/
        ├── evaluation_metrics.json
        └── predictions.csv
```

File hasil dan dataset berukuran besar sebaiknya tidak dimasukkan ke Git. Atur `.gitignore` sesuai kebijakan penyimpanan data proyek.

## Persyaratan lingkungan

Lingkungan pengembangan yang pernah digunakan dalam proyek:

- Windows
- Python 3.14
- Java Temurin OpenJDK 17
- PySpark 4.3.0
- Hadoop native binaries untuk Windows (`winutils.exe` dan `hadoop.dll`)

Dependensi Python yang digunakan oleh skrip:

```bash
pip install pyspark pyreadr pandas pyarrow matplotlib seaborn
```

Versi paket dan kompatibilitas Python–PySpark–Java perlu disesuaikan dengan lingkungan yang benar-benar digunakan. Versi di atas adalah catatan lingkungan proyek, bukan jaminan bahwa semua kombinasi paket sudah kompatibel pada setiap komputer.

### Konfigurasi Hadoop pada Windows

Pada lingkungan Windows, Spark dapat membutuhkan Hadoop native binaries. Jika binary tersedia di `C:\hadoop\bin`, konfigurasi sementara pada PowerShell:

```powershell
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.7.7-hotspot"
$env:HADOOP_HOME = "C:\hadoop"
$env:PATH = "C:\hadoop\bin;$env:PATH"
```

Periksa apakah file dapat ditemukan:

```powershell
where.exe winutils.exe
where.exe hadoop.dll
java -version
```

Sesuaikan lokasi `JAVA_HOME` dan `HADOOP_HOME` dengan instalasi komputer masing-masing. Jika masalah native Hadoop tetap terjadi, menjalankan proyek melalui WSL2/Linux merupakan alternatif yang dapat dipertimbangkan.

## Cara menjalankan

Jalankan perintah dari direktori utama repository.

### 1. Konversi dataset

Pastikan `5v_cleandf.rdata` tersedia di direktori proyek, lalu jalankan:

```bash
python 00_convert.py
```

Secara default, skrip menghasilkan:

```text
data/dataset_original.parquet
```

Lokasi input dan output dapat diubah melalui argumen:

```bash
python 00_convert.py --input "path/dataset.rdata" --output "data/dataset_original.parquet"
```

### 2. Exploratory Data Analysis

Skrip EDA yang tersedia menggunakan konfigurasi input:

```python
INPUT_PATH = "data/processed/hong_triage.parquet"
```

Sementara skrip konversi di atas menghasilkan `data/dataset_original.parquet`. **Samakan `INPUT_PATH` pada skrip EDA dengan lokasi Parquet yang benar sebelum menjalankannya.** Jika menggunakan hasil konversi default:

```python
INPUT_PATH = "data/dataset_original.parquet"
```

Kemudian jalankan:

```bash
python 01_eda.py
```

Hasil EDA disimpan di `results/eda/` sesuai konfigurasi skrip.

### 3. Membagi data train dan test

Jalankan:

```bash
python 02_split_data.py
```

Skrip ini memvalidasi target `esi`, membuang baris dengan target kosong/tidak valid dari proses split, kemudian melakukan pembagian stratified 80:20 dengan seed 42. Skrip split tidak melakukan imputasi, encoding, scaling, seleksi fitur, atau penghapusan duplikat.

**Penting — samakan format path antar-skrip:** versi `02_split_data.py` yang menggunakan `write_csv()` menghasilkan direktori CSV Spark:

```text
data/train_csv/
data/test_csv/
data/discarded_csv/
```

Direktori tersebut berisi satu atau lebih file `part-*.csv`, bukan file tunggal `train.csv` dan `test.csv`. Agar sesuai dengan output tersebut, ubah konfigurasi pada skrip training dan evaluasi:

Pada `03_model_pretriage.py`:

```python
TRAIN_PATH = Path("data/train_csv")
```

Pada `04_evaluation_fixed_v2.py`:

```python
TEST_PATH = Path("data/test_csv")
```

Spark dapat membaca direktori CSV yang berisi file part. Pastikan train dan test berasal dari satu proses split yang sama.

### 4. Training model

Setelah path training disesuaikan dan data split tersedia:

```bash
python 03_model_pretriage.py
```

Model yang dilatih adalah Multinomial Logistic Regression. Pipeline melakukan:

1. Pemilihan fitur pre-triage.
2. Imputasi median pada fitur numerik.
3. `StringIndexer` dan `OneHotEncoder` pada fitur kategorikal.
4. Penggabungan fitur menggunakan `VectorAssembler`.
5. Training Multinomial Logistic Regression.

Label target asli ESI 1–5 dipetakan menjadi label internal 0–4 untuk kebutuhan Spark ML. Mapping tersebut disimpan dalam metadata.

Output utama:

```text
models/logistic_regression/
results/model/model_metadata.json
```

Metadata mencatat parameter model, kelompok dan daftar fitur terpilih, jumlah data training, serta mapping nama kolom.

### 5. Evaluasi model

Setelah model berhasil disimpan, pastikan `TEST_PATH` pada skrip evaluasi sudah menunjuk ke direktori test yang benar. Jalankan:

```bash
python 04_evaluation_fixed_v2.py
```

Evaluasi tidak melakukan training ulang. Skrip memuat `PipelineModel` yang dihasilkan pada tahap training dan mengukur performanya pada test set.

Output:

```text
results/evaluation/evaluation_metrics.json
results/evaluation/predictions.csv
```

File prediksi berisi `actual_esi`, `predicted_esi`, dan probabilitas prediksi untuk setiap kelas.

## Metrik evaluasi

Skrip evaluasi menghitung:

- **Accuracy**: proporsi prediksi yang benar.
- **Weighted Precision, Recall, dan F1-score**: metrik yang dibobot berdasarkan jumlah sampel tiap kelas.
- **Macro Precision, Recall, dan F1-score**: rata-rata metrik antar-kelas dengan bobot sama untuk setiap kelas.
- **AUROC One-vs-Rest per kelas** dan rata-rata macro AUROC.
- **Confusion matrix**: jumlah prediksi untuk setiap kombinasi kelas aktual dan kelas prediksi.

Untuk dataset yang tidak seimbang, macro F1, macro recall, dan metrik per kelas penting untuk dilaporkan bersama accuracy dan weighted F1. Kinerja pada ESI 1 dan ESI 5 perlu diperiksa secara khusus karena jumlah sampelnya lebih sedikit dibanding kelas tengah.

## Hasil evaluasi yang tercatat

Salah satu berkas `evaluation_metrics.json` yang tersedia mencatat hasil berikut:

| Metrik | Nilai |
|---|---:|
| Test rows | 111.626 |
| Accuracy | 0,6485 |
| Weighted Precision | 0,6504 |
| Weighted Recall | 0,6485 |
| Weighted F1 | 0,6371 |
| Macro Precision | 0,6348 |
| Macro Recall | 0,4657 |
| Macro F1 | 0,4916 |
| Macro AUROC (OvR) | 0,8716 |

Metrik per kelas pada berkas tersebut:

| Kelas | Precision | Recall | F1-score | AUROC |
|---|---:|---:|---:|---:|
| ESI 1 | 0,6431 | 0,1893 | 0,2925 | 0,9273 |
| ESI 2 | 0,7202 | 0,6162 | 0,6642 | 0,8657 |
| ESI 3 | 0,6355 | 0,7210 | 0,6755 | 0,7928 |
| ESI 4 | 0,6057 | 0,6967 | 0,6480 | 0,8883 |
| ESI 5 | 0,5697 | 0,1054 | 0,1779 | 0,8838 |

**Interpretasi awal:** model memiliki accuracy sekitar 64,85%, tetapi macro recall sekitar 46,57%. Recall ESI 1 dan ESI 5 relatif rendah, sehingga model sering tidak berhasil mengenali dua kelas tersebut. Untuk konteks triase, hal ini merupakan keterbatasan penting dan perlu dibahas; accuracy saja tidak cukup untuk menyimpulkan model sudah memadai.

> **Catatan reproduksibilitas:** tabel di atas menyalin angka dari berkas evaluasi yang tersedia. Sebelum mengutipnya sebagai hasil final skripsi, pastikan file evaluasi tersebut dihasilkan dari model, fitur, dan split yang sama dengan versi skrip yang sedang digunakan. Periksa juga timestamp/riwayat run dan metadata model.

## Catatan metodologis dan keterbatasan

- Pembagian data yang tercatat adalah 80:20 train:test. Jika rancangan metodologi penelitian mensyaratkan 60:20:20 untuk train:validation:test, implementasi split perlu diubah dan eksperimen diulang. Saat ini skrip yang didokumentasikan belum menyediakan validation set terpisah.
- Stratifikasi menjaga proporsi kelas pada pembagian data, tetapi tidak mengatasi ketidakseimbangan kelas saat training.
- Fitur dipilih berdasarkan nama kolom dan asumsi ketersediaan pada tahap pre-triage. Validasi makna dan waktu ketersediaan setiap fitur tetap diperlukan.
- Belum ada klaim bahwa model telah divalidasi secara eksternal pada rumah sakit atau populasi lain.
- Performa klasifikasi tidak dengan sendirinya membuktikan keamanan penggunaan klinis. Model ini merupakan eksperimen penelitian dan bukan pengganti keputusan tenaga medis.
- Pastikan tidak ada pasien/kunjungan yang sama tersebar pada train dan test apabila dataset memiliki beberapa kunjungan per pasien. Skrip split yang didokumentasikan melakukan stratifikasi berdasarkan baris dan target, bukan pemisahan berdasarkan identitas pasien.

## Pemecahan masalah

| Kendala | Penyebab/indikasi | Tindakan yang dapat dilakukan |
|---|---|---|
| `PATH_NOT_FOUND` saat membaca data | Path input tidak sama dengan lokasi output tahap sebelumnya | Samakan konstanta path pada skrip dan struktur folder aktual |
| `FIELD_NOT_FOUND` pada nama kolom yang mengandung titik/koma | Spark dapat menafsirkan karakter tertentu sebagai ekspresi field | Gunakan nama kolom internal yang aman dan mapping metadata |
| `NativeIO$Windows.access0` / `UnsatisfiedLinkError` | Masalah akses native Hadoop pada Windows | Periksa `HADOOP_HOME`, `PATH`, `winutils.exe`, dan `hadoop.dll`; pertimbangkan WSL2/Linux |
| `Java heap space` | Penggunaan memori Spark terlalu besar untuk kapasitas yang tersedia | Kurangi partisi/kompleksitas operasi, hindari operasi window yang mahal bila tidak diperlukan, atau gunakan lingkungan dengan memori lebih besar |
| Model tidak ditemukan saat evaluasi | Training belum berhasil menyimpan model atau path berbeda | Pastikan `03_model_pretriage.py` selesai dan `model_dir` pada metadata benar |
| Header train dan test berbeda | File berasal dari split/dataset yang berbeda atau urutan kolom berubah | Buat ulang split dari satu dataset yang sama dan jangan mengubah header secara manual |

## Referensi

Hong et al. (2018), dataset triase yang menjadi sumber data utama penelitian. Lengkapi entri bibliografi dengan judul publikasi, nama jurnal/prosiding, volume, halaman, DOI, serta informasi lisensi sesuai sumber dataset asli sebelum repository atau laporan dipublikasikan.

## Status proyek

README ini mendokumentasikan alur dan konfigurasi skrip yang tersedia. Sebelum eksperimen final, lakukan pemeriksaan konsistensi path, jalankan seluruh pipeline dari awal, simpan versi skrip dan metadata yang digunakan, serta pastikan hasil evaluasi berasal dari run yang sama.
