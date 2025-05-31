# Laporan Proyek Machine Learning - Azza Wafiqurrohmah

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

### Jumlah Baris dan Kolom :
Dataset ini terdiri dari 15.000 baris dan 9 kolom. Setiap baris merepresentasikan satu observasi aktivitas fisik dari seorang individu.

### Struktur Variabel
Berikut adalah deskripsi masing-masing variabel dalam dataset:

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

### Kondisi Data
- Missing Values: Dari hasil pemeriksaan awal, tidak ditemukan missing values pada dataset ini. Semua entri lengkap dan siap digunakan untuk proses modeling.
- Data Duplikat: Tidak Ditemukan data duplikat di dataset ini.
- Outliers : Berdasarkan proses analisis dengan menggunakan boxplot, ditemukan beberepa outlier yg tersebar di beberapa kolom, yaitu kolom Height, Weight, Body Temperature, dan kolom Calories.

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

### 1. Pemeriksaan dan Penanganan Duplikasi
Dataset diperiksa terlebih dahulu untuk mengetahui apakah terdapat baris yang duplikat. Hasilnya menunjukkan bahwa tidak terdapat data duplikat, sehingga tidak dilakukan penghapusan pada tahap ini.

Alasan: Duplikasi dapat menyebabkan bias dalam pelatihan model, karena informasi yang sama dihitung lebih dari sekali.

### 2. Pemeriksaan Missing Values
Dilakukan pengecekan terhadap nilai kosong (missing values) pada setiap kolom. Hasil pemeriksaan menunjukkan bahwa seluruh entri terisi lengkap, sehingga tidak diperlukan teknik imputasi.

Alasan: Kehadiran nilai kosong dapat mengganggu proses pelatihan model dan menurunkan akurasi prediksi jika tidak ditangani.

### 3. Penghapusan Kolom yang Tidak Relevan
Kolom User_ID dihapus karena hanya berfungsi sebagai identifier unik dan tidak memiliki nilai prediktif terhadap target (Calories_Burnt).

Alasan: Menyertakan kolom yang tidak relevan berpotensi menambah noise dalam data dan menurunkan performa model.

### 4. Normalisasi Data (Scaling)
Fitur-fitur numerik dinormalisasi menggunakan teknik Min-Max Scaling agar nilai setiap fitur berada dalam rentang [0, 1].

Alasan: Scaling sangat penting terutama untuk algoritma seperti K-Nearest Neighbors (KNN) yang sensitif terhadap perbedaan skala antar fitur.

### 5. Deteksi dan Penanganan Outlier
Outlier dideteksi melalui visualisasi boxplot untuk masing-masing fitur numerik. Nilai-nilai yang berada di luar batas interkuartil (menggunakan metode IQR) kemudian dihapus dari dataset.

Alasan: Outlier dapat mengganggu distribusi data dan menyebabkan model belajar dari informasi yang tidak representatif.

### 6. Feature Selection
Seleksi fitur dilakukan dengan menganalisis hubungan (korelasi) antara fitur numerik dan target variabel (`Calories_Burnt`). Korelasi dihitung menggunakan metode Pearson.

Dari hasil analisis, ditemukan bahwa dua fitur memiliki hubungan yang sangat lemah terhadap target, sehingga fitur-fitur tersebut **tidak disertakan dalam pemodelan** untuk menghindari noise atau informasi tidak relevan.

**Fitur yang digunakan dalam model:**
- Gender
- Age
- Duration
- Heart Rate
- Body Temperature

**Target:**
- Calories Burnt

### 7. Encoding Variabel Kategorikal
Setelah tahap EDA selesai, dilakukan encoding terhadap variabel kategorikal Gender menggunakan metode One-Hot Encoding. Karena hanya terdapat dua kategori, hanya satu dummy variable (Gender_Male) yang disimpan dengan mengaktifkan drop_first=True untuk menghindari multikolinearitas.

Alasan: Algoritma regresi memerlukan data dalam format numerik. One-Hot Encoding efektif untuk menangani fitur kategorikal non-ordinal seperti gender.

### 8. Split Data (Training dan Testing)
Data dibagi menjadi dua bagian: training (80%) dan testing (20%) menggunakan fungsi train_test_split() dari scikit-learn.

Alasan: Pembagian ini bertujuan untuk melatih model pada sebagian data dan menguji performanya pada data yang belum pernah dilihat, sehingga dapat mengukur kemampuan generalisasi model.


## Modeling
Pada proyek ini, digunakan dua algoritma regresi untuk memprediksi jumlah kalori yang terbakar (Calories_Burnt), yaitu K-Nearest Neighbors (KNN) dan Random Forest Regressor (RF). Tujuan utama proyek ini adalah membandingkan performa kedua algoritma dalam melakukan prediksi secara akurat dan andal.

### 1. K-Nearest Neighbors (KNN)
#### Cara Kerja :
KNN adalah algoritma non-parametrik yang memprediksi nilai target berdasarkan rata-rata nilai dari k tetangga terdekat di ruang fitur. Jarak antar data biasanya dihitung menggunakan Euclidean Distance. Dalam konteks regresi, KNN akan mencari k data poin yang paling dekat dengan data baru, lalu menghitung rata-rata dari nilai target (Calories_Burnt) milik tetangga-tetangga tersebut sebagai hasil prediksi.

#### Parameter : 
Model KNN diinisialisasi dengan parameter berikut:
- n_neighbors=10 → Jumlah tetangga terdekat yang digunakan untuk prediksi

Catatan: Sebelum diterapkan ke model KNN, semua fitur numerik dinormalisasi menggunakan MinMaxScaler agar skala antar fitur seragam, mengingat KNN sangat sensitif terhadap skala data.

### 2. Random Forest Regressor (RF)
#### Cara Kerja :
Random Forest adalah algoritma ensemble learning berbasis pohon keputusan. Ia membangun banyak decision tree secara acak pada subset data (dengan teknik bootstrap) dan menggabungkan prediksi semua pohon (dalam regresi: rata-rata prediksi) untuk meningkatkan akurasi dan mengurangi risiko overfitting. Random Forest juga melakukan pemilihan fitur secara acak pada setiap split, yang membuatnya sangat kuat dalam menangani data kompleks dan non-linear.

#### Parameter :
Model Random Forest diinisialisasi dengan parameter sebagai berikut:
- n_estimators=50 → Jumlah pohon yang dibangun
- max_depth=16 → Kedalaman maksimum setiap pohon
- random_state=55 → Seed acak untuk memastikan hasil konsisten
- n_jobs=-1 → Menggunakan seluruh core CPU untuk mempercepat proses training

Catatan: Meskipun Random Forest tidak sensitif terhadap skala fitur seperti KNN, normalisasi tetap diterapkan demi konsistensi preprocessing.

## Kelebihan dan Kekurangan

| Algoritma                  | Kelebihan                                                                                     | Kekurangan                                                                |
|----------------------------|-----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| **K-Nearest Neighbors (KNN)** | - Mudah diimplementasikan<br>- Tidak memerlukan proses training<br>- Cocok untuk dataset kecil | - Sensitif terhadap skala fitur dan outlier<br>- Kurang efisien untuk dataset besar |
| **Random Forest Regressor (RF)** | - Mampu menangani data non-linear dan kompleks<br>- Robust terhadap overfitting dan outlier<br>- Menyediakan estimasi pentingnya fitur | - Model lebih kompleks dan sulit diinterpretasi<br>- Proses training relatif lambat |


4. Pemilihan Model Terbaik
Berdasarkan evaluasi menggunakan metrik Mean Squared Error (MSE), Random Forest menunjukkan performa yang lebih baik pada data training dan testing dibandingkan KNN. Hal ini menunjukkan bahwa Random Forest mampu menangkap pola dalam data secara lebih efektif serta memiliki kemampuan generalisasi yang lebih tinggi. Oleh karena itu, model Random Forest dipilih sebagai model terbaik untuk prediksi jumlah kalori terbakar.
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
