# Laporan Proyek Machine Learning - Azza WAfiqurrohmah

# Calories Burnt Prediction

## Domain Proyek

Di era modern saat ini, meningkatnya gaya hidup sedentari dan pola makan tidak sehat telah menyebabkan lonjakan signifikan kasus obesitas dan penyakit metabolik seperti diabetes dan penyakit jantung. Menurut World Health Organization (WHO), obesitas telah menjadi salah satu tantangan kesehatan global terbesar, yang berdampak pada kualitas hidup dan beban ekonomi dunia.

Salah satu cara efektif dalam mencegah dan mengendalikan masalah tersebut adalah melalui pengelolaan energi, yakni menyeimbangkan kalori yang masuk dan kalori yang terbakar oleh tubuh. Namun, memantau jumlah kalori yang terbakar selama aktivitas fisik secara akurat menjadi tantangan besar bagi banyak orang, terutama karena keterbatasan akses ke alat pengukur kalori yang canggih dan mahal.

Dengan solusi prediksi kalori terbakar yang berbasis data dan machine learning, diharapkan dapat membuka akses bagi siapa saja untuk memantau aktivitas fisik mereka secara real-time dengan biaya terjangkau, sekaligus mendorong perubahan gaya hidup lebih sehat yang berdampak jangka panjang.

### Referensi
- Harvard Health Publishing. (2021). Calories burned in 30 minutes for people of three different weights. Harvard Medical School
- Mayo Clinic Staff. (2022). Exercise for weight loss: Calories burned in 1 hour. Mayo Clinic
- Mathur, P., & Aggarwal, P. (2018). Obesity and lifestyle management. Journal of Clinical & Diagnostic Research, 12(3). DOI: 10.7860/JCDR/2018/31723.11327

## Business Understanding

### Problem Statements
 - Bagaimana memprediksi kalori terbakar secara akurat menggunakan data sederhana yang mudah diperoleh?
 - Algoritma regresi mana yang lebih efektif dalam memodelkan hubungan antara fitur-fitur fisik dan kalori terbakar?
 - Bagaimana membandingkan performa dua algoritma, yaitu K-Nearest Neighbors (KNN) dan Random Forest, untuk memilih model terbaik?

### Goals
 - Mengembangkan model prediksi kalori terbakar dengan akurasi tinggi.
 - Membandingkan performa KNN dan Random Forest berdasarkan metrik evaluasi seperti Mean Squared Error (MSE).
 - Memilih algoritma terbaik yang dapat diterapkan secara praktis pada aplikasi kebugaran.

### Solution statements
 - Melakukan pengujian dua algoritma regresi, KNN dan Random Forest, untuk membangun model prediksi kalori terbakar.
 - Melakukan tuning hyperparameter secara manual pada kedua model untuk mengoptimalkan performa berdasarkan evaluasi MSE.
 - Mengevaluasi dan membandingkan hasil prediksi kedua model menggunakan data testing.
 - Memberikan rekomendasi model terbaik berdasarkan hasil evaluasi yang diperoleh.
 

## Data Understanding
Pada proyek ini, dataset yang digunakan adalah dataset Calories Burnt Prediction yang tersedia di Kaggle melalui tautan berikut:
https://www.kaggle.com/datasets/ruchikakumbhar/calories-burnt-prediction.

Dataset ini berisi data yang merekam berbagai atribut fisik dan aktivitas seseorang yang dapat digunakan untuk memprediksi jumlah kalori yang terbakar. Dataset mencakup sejumlah fitur yang berperan sebagai variabel input dan satu variabel target, yaitu kalori terbakar.

### Variabel-variabel pada dataset ini adalah sebagai berikut:
| Variabel       | Tipe Data   | Deskripsi                                             |
|----------------|-------------|-------------------------------------------------------|
| User_ID        | Kategori    | ID unik dari individu yang melakukan aktivitas        |
| Gender         | Kategori    | Jenis kelamin (Male/Female)                           |
| Age            | Numerik     | Umur individu dalam tahun                             |
| Height         | Numerik     | Tinggi badan dalam centimeter                         |
| Weight         | Numerik     | Berat badan dalam kilogram                            |
| Duration       | Numerik     | Durasi aktivitas dalam detik                          |
| Heart_Rate     | Numerik     | Detak jantung rata-rata selama aktivitas (bpm)        |
| Body_Temp      | Numerik     | Suhu tubuh selama aktivitas (derajat Celsius)         |
| Calories_Burnt | Numerik     | Jumlah kalori yang terbakar selama aktivitas (target) |

## Exploratory Data Analysis (EDA)

### 1. Statistik Deskriptif

Berikut adalah ringkasan statistik dasar dari variabel numerik pada dataset Calories Burnt Prediction:

| Variabel         | Mean   | Median | Std Dev | Min  | Max  |
|------------------|--------|--------|---------|------|------|
| Age              | 42.79  | 39     | 16.98   | 20   | 79   |
| Height (cm)      | 174.47 | 175    | 14.26   | 123  | 222  |
| Weight (kg)      | 74.97  | 74     | 15.04   | 36   | 132  |
| Duration (s)     | 15.53  | 16     | 8.32    | 1    | 30   |
| Heart Rate (bpm) | 95.52  | 96     | 9.58    | 67   | 128  |
| Body Temp (°C)   | 40.03  | 40.20  | 0.78    | 37.1 | 41.5 |
| Calories Burnt   | 89.54  | 79     | 62.46   | 1    | 314  |

- Rata-rata umur responden adalah sekitar 43 tahun dengan rentang antara 20 hingga 79 tahun.  
- Tinggi badan rata-rata adalah 174.5 cm dengan variasi dari 123 cm hingga 222 cm.  
- Durasi aktivitas rata-rata sekitar 15.5 detik, dengan durasi terpendek 1 detik dan terlama 30 detik.  
- Kalori terbakar bervariasi cukup besar, dengan rata-rata sekitar 89.5 kalori dan maksimum mencapai 314 kalori.

### 2. Distribusi fitur numerik
![image](https://github.com/user-attachments/assets/2ccdd57f-86b4-478e-bd19-1914ecbc9347)

### 3. Analisis Variabel Kategori
![image](https://github.com/user-attachments/assets/c4b37e93-4d9a-4af1-acfd-95ef701228d8)

### 4. Multivariate Analisys
#### 1. Categorical Features
![image](https://github.com/user-attachments/assets/428bef62-eb26-4d3c-9eff-6cdcdac14c14)

#### 2. Numerical Features
![image](https://github.com/user-attachments/assets/93a409fc-ab9a-46ac-bd10-b164b86cc622)


## Data Preparation
Sebelum melakukan proses pemodelan, beberapa tahapan data preparation dilakukan untuk memastikan data bersih dan siap diproses oleh algoritma regresi. Berikut adalah langkah-langkah yang dilakukan:

### 1. Pemeriksaan Missing Values
Dataset dicek untuk mengetahui apakah terdapat nilai yang hilang (missing values). Hasil pemeriksaan menunjukkan bahwa dataset ini tidak memiliki missing values sehingga tidak diperlukan proses imputasi.

Alasan: Missing values dapat memengaruhi hasil pelatihan model. Jika ditemukan, perlu diatasi agar model tidak belajar dari data yang tidak lengkap.

### 2. Pemeriksaan Duplikasi
Dilakukan pengecekan terhadap duplikat baris pada dataset. Hasilnya, tidak ditemukan baris duplikat sehingga tidak ada data yang dihapus.

Alasan: Duplikasi dapat menyebabkan bias pada model karena informasi yang sama dihitung lebih dari sekali. Pemeriksaan ini penting untuk memastikan integritas data.

### 3. Encoding Variabel Kategorikal
Variabel Gender merupakan variabel kategorikal dengan dua nilai: Male dan Female. Untuk memproses data ini ke dalam model regresi, dilakukan encoding menggunakan pendekatan label encoding:
- Male → 1
- Female → 0

Alasan: Algoritma regresi memerlukan input numerik. Encoding konversi kategori ke angka diperlukan agar model bisa memproses fitur ini.

### 4. Feature Selection
Seleksi fitur dilakukan dengan menganalisis hubungan (korelasi) antara fitur numerik dan target variabel (`Calories_Burnt`). Korelasi dihitung menggunakan metode Pearson.

Dari hasil analisis, ditemukan bahwa dua fitur memiliki hubungan yang sangat lemah terhadap target, sehingga fitur-fitur tersebut **tidak disertakan dalam pemodelan** untuk menghindari noise atau informasi tidak relevan.

**Fitur yang digunakan dalam model:**
- Gender (setelah encoding)
- Age
- Duration
- Heart Rate
- Body Temperature

**Target:**
- Calories Burnt

**Alasan**:  
Karna melakukan feature selection dengan menganalisis korelasi, maka feature `Height` dan `Weight` yang memiliki korelasi sebesar 0,02 dan 0,04 tidak digunakan dalam melatih model.

### 5. Split Data (Training dan Testing)
Data dibagi menjadi dua bagian: training (80%) dan testing (20%) menggunakan fungsi train_test_split() dari scikit-learn.

Alasan: Pembagian ini bertujuan untuk melatih model pada sebagian data dan menguji performanya pada data yang belum pernah dilihat, sehingga dapat mengukur kemampuan generalisasi model.

### 6. Feature Scaling (Normalisasi)

Semua fitur numerik dinormalisasi ke dalam rentang 0–1 menggunakan `MinMaxScaler` dari scikit-learn.
Meskipun algoritma **Random Forest** secara teori tidak memerlukan scaling karena berbasis pohon keputusan, proses normalisasi tetap diterapkan **demi menjaga konsistensi preprocessing antar model**. Ini juga memastikan bahwa algoritma **K-Nearest Neighbors (KNN)** — yang sensitif terhadap skala fitur — dapat bekerja secara optimal.
Dengan demikian, kedua model mendapatkan input yang telah dinormalisasi secara seragam.


## Modeling

Pada proyek ini, digunakan dua algoritma regresi untuk memprediksi jumlah kalori yang terbakar (`Calories_Burnt`), yaitu **K-Nearest Neighbors (KNN)** dan **Random Forest (RF)**. Tujuan utama proyek ini adalah membandingkan performa kedua algoritma dalam memprediksi kalori terbakar secara akurat.

### Tahapan dan Parameter Modeling

- **K-Nearest Neighbors (KNN)**  
  Model KNN menggunakan parameter `n_neighbors=5`. Karena KNN bergantung pada jarak antar data, dilakukan normalisasi fitur numerik menggunakan `MinMaxScaler` agar skala fitur seragam dan prediksi optimal.

- **Random Forest Regressor (RF)**  
  Model RF menggunakan parameter `n_estimators=50` (jumlah pohon keputusan) dan `max_depth=16` (kedalaman maksimal pohon). Meskipun RF tidak terlalu sensitif pada skala fitur, normalisasi tetap dilakukan untuk konsistensi preprocessing.

### Kelebihan dan Kekurangan

| Algoritma        | Kelebihan                                               | Kekurangan                                               |
|------------------|---------------------------------------------------------|----------------------------------------------------------|
| **KNN**          | - Mudah diimplementasikan                                | - Sensitif terhadap skala fitur dan outlier              |
|                  | - Tidak perlu asumsi distribusi data                     | - Performa menurun jika data sangat besar                 |
| **Random Forest** | - Mampu menangani data non-linear dan kompleks          | - Model kompleks dan kurang mudah diinterpretasi         |
|                  | - Robust terhadap outlier dan overfitting               | - Proses training lebih lambat dibanding model sederhana |
|                  | - Memberikan estimasi pentingnya fitur                   |                                                          |

### Pemilihan Model Terbaik

Berdasarkan evaluasi, Random Forest menunjukkan nilai Mean Squared Error (MSE) yang lebih rendah pada data training dan testing dibanding KNN. Hal ini menunjukkan bahwa Random Forest memiliki kemampuan prediksi yang lebih baik dan generalisasi yang lebih kuat, sehingga dipilih sebagai model terbaik.

---

## Evaluation

### Metrik Evaluasi

Metrik yang digunakan adalah **Mean Squared Error (MSE)**, yang mengukur rata-rata kuadrat dari selisih antara nilai aktual dan prediksi.
MSE yang lebih kecil berarti model memiliki akurasi prediksi yang lebih baik.

### Hasil Evaluasi

| Metrik    | KNN       | Random Forest |
|-----------|-----------|--------------|
| Train MSE | 0.000148  | 0.000023     |
| Test MSE  | 0.000196  | 0.000163     |

Kedua model menghasilkan nilai MSE yang rendah, menunjukkan performa prediksi kalori terbakar yang baik. Namun, Random Forest lebih unggul dengan nilai MSE lebih kecil di data testing, menandakan kemampuan generalisasi yang lebih baik.
