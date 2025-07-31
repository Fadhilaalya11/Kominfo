import streamlit as st
import pandas as pd

st.header("🔎 Latar Belakang & Kondisi Emisi Terkini")

# Poin 1
st.markdown("""
### 1. Tahun 2023, emisi karbon dunia mencapai hampir 37 miliar ton dan terus meningkat setiap tahun
""")

# Poin 2 dengan grafik
st.markdown("""
### 2. Di Indonesia, emisi karbon meningkat drastis dalam 10 tahun terakhir, terutama sejak 2012.
""")

# Gambar grafik (ganti path jika perlu)
st.image("Grafik_Emisi.png", caption="Grafik Kenaikan Emisi Karbon di Indonesia", use_column_width=True)

# Poin 3
st.markdown("""
### 3. Kabupaten Banyumas juga menghadapi tantangan besar, dimana belum ada sistem yang bisa memantau kadar emisi karbon.
""")

st.markdown("---")
st.header("🎯 Tujuan Pembuatan Peta Emisi Karbon")

st.markdown("""
- Memetakan potensi emisi berdasarkan densitas infrastruktur dan tutupan lahan per grid 1 km².
- Menyediakan alat bantu visual untuk perencanaan penggunaan lahan dan mitigasi emisi di level **kecamatan**.
- Menyajikan hasil secara **kuantitatif** (skor/ranking grid) dan **spasial** (peta interaktif/visualisasi).
""")

st.markdown("---")
st.header("📂 Data & Metodologi")

st.markdown("""
| Jenis Data              | Deskripsi                            |
|-------------------------|--------------------------------------|
| Batas Administratif     | SHP GADM Kabupaten Banyumas (EPSG:32749) |
| Jaringan Jalan          | Layer `JALAN_LN_25K` (RBI geodatabase) |
| Bangunan                | Layer `PERUMAHAN_AR_25K`, `PERMUKIMAN_AR_25K` |
| Area Hijau              | Layer `HUTANLAHANTINGGI_AR_25K`, `HUTANLAHANRENDAH_AR_25K`,`HERBADANRUMPUT_AR_25K`,`HUTANTANAMAN_AR_25K`|
| Grid Spasial            | Grid 1 km × 1 km yang dibentuk dari bounding box Kabupaten |
| Fitur/Parameter         | Kepadatan bangunan (m²/km²), kepadatan jalan (km/km²), persentase hijau (%) |
| Klasifikasi             | ` Emisi Skor = (α × Kepadatan Jalan) + (β × Kepadatan Bangunan) − (γ × Persentase Area Hijau)`|
""")

st.markdown("---")
st.header("📌 Catatan Penting")

st.info("""
- Emisi yang dipetakan bersifat **estimasi spasial**, bukan pengukuran langsung CO₂ lapangan.
- Hasil ini **hanya mencerminkan potensi relatif emisi** berdasarkan pengaruh karakteristik wilayah.
- Gunakan data ini sebagai alat pendukung dalam kebijakan tata ruang dan mitigasi karbon lokal.
""")
