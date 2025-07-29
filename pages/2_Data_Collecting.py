import streamlit as st

# Judul utama dan deskripsi
st.title("🌍 Data Collecting")
st.markdown("""
Data yang ditampilkan mencakup:

- 🗺️ Batas Administratif
- 🚧 Jaringan Jalan
- 🏠 Distribusi Bangunan
- 🌿 Area Hijau (Ruang Terbuka Hijau)

---
""")

# === Peta Spasial ===
with st.expander("🖼️ Peta Spasial Kabupaten Banyumas", expanded=False):
    with st.columns(1)[0]:
        st.image("peta_banyumas.png", caption="Peta Visualisasi Spasial Kabupaten Banyumas", use_column_width=True)

# === Grid Banyumas ===
with st.expander("🖼️ Grid Kabupaten Banyumas", expanded=False):
    with st.columns(1)[0]:
        st.image("grid_banyumas.png", caption="Peta Grid Kabupaten Banyumas", use_column_width=True)

