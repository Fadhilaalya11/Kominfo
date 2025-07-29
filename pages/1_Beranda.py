import streamlit as st
import pandas as pd

st.header("🔎 Latar Belakang & Kondisi Emisi Terkini")

st.markdown("""
Kabupaten Banyumas, bagian dari **Provinsi Jawa Tengah**, berada di tengah dinamika pembangunan kota dan lingkungan hijau.
Menurut data inventarisasi nasional GRK sektor energi by Kementerian LHK:
- Provinsi **Jawa Tengah menghasilkan sekitar 183,76 gigagram (Gg CO₂e)** pada tahun 2022 — termasuk dari energi, transportasi, dan industri :contentReference[oaicite:4]{index=4}.
- Secara nasional, emisi GRK Indonesia meningkat dari **864,85 Mt CO₂e tahun 2013** menjadi **~1.200 Mt CO₂e pada tahun 2023** :contentReference[oaicite:5]{index=5}.

📌 Potensi emisi di Banyumas terkait intensifikasi infrastruktur dan penggunaan lahan menjadi penting untuk dipetakan dan dimitigasi.
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
