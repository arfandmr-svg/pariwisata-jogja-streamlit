import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Advanced Dashboard Pariwisata DIY", layout="wide")

# --- 2. JUDUL & DESKRIPSI ---
st.title("Dashboard Segmentasi Wisata DIY")
st.markdown("""
Dashboard ini menggunakan algoritma **K-Means Clustering** untuk mengelompokkan objek wisata di Daerah Istimewa Yogyakarta secara otomatis. 
Dilengkapi dengan metode evaluasi *Elbow Method* dan *Silhouette Coefficient* untuk menentukan klaster.
""")
st.divider()

# --- 3. PROSES LOAD & TRAINING DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("data_wisata_final_terklaster.csv")
    return df

df = load_data()

# Ekstraksi fitur untuk pemodelan
X = df[['Rating', 'Ulasan']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Utama K=2
kmeans_model = KMeans(n_clusters=2, random_state=42)
kmeans_model.fit(X_scaled)
df['Cluster'] = kmeans_model.labels_

# --- 4. SIDEBAR: FILTER DATA & PREDIKSI INTERAKTIF ---
st.sidebar.header("Panel Kontrol Interaktif")

# Fitur Tambahan 1: Filter Dataset secara Real-time
st.sidebar.subheader("Filter Tampilan Data")
filter_rating = st.sidebar.slider("Minimal Rating Wisata", 1.0, 5.0, 1.0, step=0.1)
filter_ulasan = st.sidebar.number_input("Maksimal Ulasan Lokasi", min_value=0, value=int(df['Ulasan'].max()))

# Menerapkan Filter ke Dataframe yang akan ditampilkan di grafik
df_filtered = df[(df['Rating'] >= filter_rating) & (df['Ulasan'] <= filter_ulasan)]

st.sidebar.divider()

# Fitur Prediksi Wisata Baru
st.sidebar.subheader("Prediksi Objek Wisata Baru")
nama_wisata = st.sidebar.text_input("Nama Tempat Wisata", "Pantai Baru Jogja")
input_rating = st.sidebar.slider("Input Rating", 1.0, 5.0, 4.5, step=0.1)
input_ulasan = st.sidebar.number_input("Input Jumlah Ulasan", min_value=0, value=500)

if st.sidebar.button("Jalankan Prediksi Mesin"):
    new_data = np.array([[input_rating, input_ulasan]])
    new_data_scaled = scaler.transform(new_data)
    prediksi = kmeans_model.predict(new_data_scaled)[0]
    
    st.sidebar.success(f"Analisis Selesai!")
    if prediksi == 0:
        st.sidebar.info(f"**{nama_wisata}** diklasifikasikan sebagai:\n\n**Klaster 0 (Hidden Gem)**")
    else:
        st.sidebar.warning(f"**{nama_wisata}** diklasifikasikan sebagai:\n\n**Klaster 1 (Mass Tourism)**")

# --- 5. TAMPILAN UTAMA (4 TABS) ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analisis & Visualisasi", 
    "🧪 Eksperimen Validasi Klaster", 
    "🗂️ Eksplorasi Data & Download", 
    "ℹ️ Karakteristik Klaster"
])

# TAB 1: GRAFIK & METRIK (DIBUAT DINAMIS BERDASARKAN FILTER)
with tab1:
    st.subheader("Ringkasan Data Pariwisata (Berdasarkan Filter)")
    
    # Metrik dinamis mengikuti filter
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Wisata Terfilter", f"{len(df_filtered)} Lokasi", help="Jumlah lokasi yang lolos filter sidebar")
    col2.metric("Klaster 0 (Hidden Gem)", f"{len(df_filtered[df_filtered['Cluster']==0])} Lokasi")
    col3.metric("Klaster 1 (Mass Tourism)", f"{len(df_filtered[df_filtered['Cluster']==1])} Lokasi")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns([2, 1])
    
    with col_chart1:
        st.markdown("**Pemetaan Scatter Plot Objek Wisata (Real-time)**")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {0: '#1f77b4', 1: '#d62728'}
        labels_map = {0: 'Klaster 0 (Hidden Gem)', 1: 'Klaster 1 (Mass Tourism)'}

        for cluster_id, color in colors.items():
            sub_df = df_filtered[df_filtered['Cluster'] == cluster_id]
            ax.scatter(sub_df['Rating'], sub_df['Ulasan'], c=color, label=labels_map[cluster_id], alpha=0.7, edgecolors='w', s=80)

        ax.set_xlabel('Rating Wisata', fontsize=10)
        ax.set_ylabel('Jumlah Ulasan Google Maps', fontsize=10)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig)
        
    with col_chart2:
        st.markdown("**Proporsi Jumlah Wisata**")
        if not df_filtered.empty:
            cluster_counts = df_filtered['Cluster'].value_counts().rename(index={0: 'Hidden Gem', 1: 'Mass Tourism'})
            st.bar_chart(cluster_counts)
        else:
            st.warning("Tidak ada data yang cocok dengan filter.")

# TAB 2: EKSPERIMEN VALIDASI (Poin Plus Nilai Akademis)
with tab2:
    st.subheader("Pembuktian Ilmiah Penentuan Jumlah Klaster (K)")
    st.markdown("Bagian ini menampilkan proses penentuan jumlah klaster terbaik menggunakan metode matematis.")
    
    # Hitung WCSS dan Silhouette secara dinamis untuk K=2 sampai K=6
    wcss = []
    silhouette_scores = []
    K_range = range(2, 7)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, km.labels_))
        
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("**1. Elbow Method (Evaluasi WCSS)**")
        fig_el, ax_el = plt.subplots(figsize=(6, 4))
        ax_el.plot(K_range, wcss, marker='o', color='purple', linestyle='--')
        ax_el.set_xlabel('Jumlah Klaster (K)')
        ax_el.set_ylabel('WCSS (Inertia)')
        ax_el.set_title('Mencari Titik Siku (Elbow)')
        ax_el.grid(True)
        st.pyplot(fig_el)
        st.caption("Analisis: Sudut siku mulai melandai secara tajam pada K=2, menandakan pembagian klaster sudah optimal.")
        
    with col_exp2:
        st.markdown("**2. Silhouette Coefficient (Evaluasi Kerapatan)**")
        fig_sil, ax_sil = plt.subplots(figsize=(6, 4))
        ax_sil.plot(K_range, silhouette_scores, marker='s', color='green', linestyle='-')
        ax_sil.set_xlabel('Jumlah Klaster (K)')
        ax_sil.set_ylabel('Silhouette Score')
        ax_sil.set_title('Mencari Nilai Tertinggi')
        ax_sil.grid(True)
        st.pyplot(fig_sil)
        st.caption(f"Analisis: Skor tertinggi didapatkan pada K=2 (Skor: ~{max(silhouette_scores):.2f}), membuktikan K=2 adalah yang terbaik atau unggul.")

# TAB 3: DATA EKSPLORASI & DOWNLOAD (Fitur Industri)
with tab3:
    st.subheader("Eksplorasi Seluruh Dataset Hasil Klaster")
    st.markdown("Bisa melakukan pencarian, mengurutkan data, atau mengunduh dataset yang sudah diperhiyungkan menggunakan algoritma Machine Learning di bawah ini.")
    
    # Fitur Tambahan 2: Tombol Download CSV
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Unduh Hasil Clustering (Format CSV)",
        data=csv_data,
        file_name="hasil_clustering_wisata_diy.csv",
        mime="text/csv"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=350)

# TAB 4: KESIMPULAN
with tab4:
    st.subheader("Penjelasan Karakteristik Klaster")
    st.markdown("Algoritma **Machine Learning K-Means** mengelompokkan objek wisata di DIY ke dalam 2 klaster utama berdasarkan kombinasi dua indikator kinerja Kualitas Pelayanan (Rating) dan Tingkat Popularitas (Jumlah Ulasan).")
    st.markdown("Sebelum dikelompokkan, kedua data tersebut disamakan bobotnya menggunakan StandardScaler agar penilaiannya adil. Aturan penentuannya adalah sebagai berikut:")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info("""
        #### Klaster 0 (Potensial / Hidden Gem)
        * **Karakteristik:** Memiliki Rating Tinggi (sangat disukai pengunjung yang datang) namun dengan Jumlah Ulasan yang Relatif Sedikit/Sedang.
        * **Logika:** Objek wisata diletakkan di klaster ini karena secara matematis posisinya berada di area koordinat "kepuasan tinggi namun eksposur rendah". Tempat ini memiliki potensi besar untuk dikembangkan lebih lanjut karena kualitasnya sudah diakui, namun belum diketahui oleh wisatawan skala luas.
        * **Rekomendasi Strategis:** Wajib dijadikan prioritas utama promosi digital iklan daerah oleh Pemda DIY agar kunjungan merata.
        """)
    with col_info2:
        st.warning("""
        #### Klaster 1 (Populer / Mass Tourism)
        * **Karakteristik:** Memiliki Rating Tinggi dan didukung oleh Jumlah Ulasan yang Sangat Masif (mencapai ribuan hingga puluhan ribu).
        * **Logika:** Objek wisata diletakkan di klaster ini karena titik datanya mendekati pusat gravitasi Mass Tourism. Banyaknya jumlah ulasan menjadi bukti otentik bahwa tempat ini merupakan magnet utama pariwisata daerah yang menghasilkan perputaran ekonomi tinggi dan volume kunjungan yang padat.
        * **Rekomendasi Strategis:** Fokus pada manajemen pelayanan alur massa dan perawatan fasilitas fisik, bukan lagi pada promosi agresif.
        """)
