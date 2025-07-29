import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Menu", layout="wide")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="📁 Menu",  # judul menu sidebar
        options=[
            "Beranda",
            "Data Collecting",
            "Data Labeling",
            "Data Understanding",
            "Model Comparison",
            "Result Visualization",
            "Code"
        ],
        icons=[
            "bar-chart", "database", "pencil-square",
            "graph-up", "layers", "globe", "code-slash"
        ],
        menu_icon="cast",  # ikon utama sidebar
        default_index=0,
        orientation="vertical"
    )

# Navigasi ke file di dalam folder `modules/`
if selected == "Beranda":
    st.switch_page("pages/1_Beranda.py")
elif selected == "Data Collecting":
    st.switch_page("pages/2_Data_Collecting.py")
elif selected == "Data Labeling":
    st.switch_page("pages/3_Data_Labeling.py")
elif selected == "Data Understanding":
    st.switch_page("pages/4_Data_Understanding.py")
elif selected == "Model Comparison":
    st.switch_page("pages/5_Model_Comparison.py")
elif selected == "Result Visualization":
    st.switch_page("pages/6_Result_Visualization.py")
elif selected == "Code":
    st.switch_page("pages/Code.py")
