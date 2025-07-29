import streamlit as st

st.set_page_config(page_title="Peta Emisi Karbon", layout="centered")

st.title("🌍 Visualisasi Emisi Karbon 💨")

st.markdown("""
Berikut adalah peta interaktif hasil perhitungan emisi karbon 🔥 berdasarkan wilayah yang telah dianalisis.  
Gunakan peta di bawah ini untuk mengeksplorasi sebaran emisi dan dampaknya terhadap lingkungan sekitar ♻️.
""")

# Link embed dari Google My Maps (gunakan mode "embed", bukan "edit")
map_url = "https://www.google.com/maps/d/u/0/view?mid=13E0bOPs1jObnzDEzjsztCjJF70igCkc&usp=sharing"

# Tampilkan iframe peta
st.components.v1.iframe(src=map_url, width=800, height=600)
