import streamlit as st

# Judul dan deskripsi
st.title("📊 Data Understanding")

st.markdown("""
Data Understanding mencakup pemrosesan dan analisis distribusi data berdasarkan kategori emisi karbon.  
Beberapa hal yang ditampilkan:

- Distribusi kelas sebelum dan sesudah oversampling
- Distribusi data training vs testing

---
""")

# === Grafik Distribusi Kategori Emisi ===
with st.expander("🖼️ Grafik Distribusi Kategori", expanded=False):
    with st.columns(1)[0]:
        st.image("distribusi_kategori.png", caption="Grafik Distribusi Kategori Emisi", use_column_width=True)

# === Grafik Distribusi Train/Test Split ===
with st.expander("🖼️ Grafik Distribusi Data Train/Test", expanded=False):
    with st.columns(1)[0]:
        st.image("distribusi data (training 783, testing 213).png", caption="Distribusi Data Training vs Testing", use_column_width=True)


