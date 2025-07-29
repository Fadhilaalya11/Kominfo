import streamlit as st
import pandas as pd

# Judul
st.title("Kategori Emisi Karbon Berdasarkan Karakteristik Wilayah")

# Deskripsi
st.markdown("""
Informasi ini digunakan untuk mengklasifikasikan wilayah berdasarkan tingkat emisi karbon berdasarkan **persentase area hijau**, **kepadatan jalan**, dan **kepadatan bangunan**. Kategori ini membantu memahami sejauh mana sebuah wilayah berkontribusi terhadap emisi karbon dan bagaimana kondisi lingkungannya.
""")

# Data Tabel
data = {
    "Kategori Emisi Karbon": ["Normal (Rendah)", "Sedang", "Parah (Tinggi)"],
    "Persentase Area Hijau": [">= 30%", "15% – 30%", "< 15%"],
    "Kepadatan Jalan (km/km²)": ["≤ 5 km/km²", "5 – 10 km/km²", "> 10 km/km²"],
    "Kepadatan Bangunan (bangunan/km²)": ["≤ 1000", "1000 – 3000", "> 3000"],
    "Keterangan": [
        "Kota berkelanjutan, banyak RTH, lalu lintas rendah",
        "Kota berkembang, RTH mulai berkurang, lalu lintas padat",
        "Urban sprawl, minim RTH, lalu lintas dan kepadatan bangunan sangat tinggi"
    ]
}

# Buat DataFrame
df_kategori = pd.DataFrame(data)

# Tampilkan tabel
st.table(df_kategori)

# Penutup
st.info("RTH = Ruang Terbuka Hijau. Kategori ini dapat digunakan sebagai acuan awal dalam memprediksi atau mengevaluasi emisi karbon pada level kecamatan atau wilayah tertentu.")