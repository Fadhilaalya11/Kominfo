import streamlit as st

# Title & deskripsi
st.set_page_config(layout="wide")
st.title("🌍 Model Comparison")
st.markdown("""
Model Comparison merupakan perbandingan model yang kami lakukan untuk mencari perhitungan dengan akurasi yang paling baik.
Model yang digunakan:
1. Random Forest
2. Logistic Regression
3. MLP
4. LightGBM
5. XGBoost
6. SVM
---
""")

# === Random Forest ===
with st.expander("🖼️ Random Forest", expanded=False):
    st.markdown("""
**Berikut parameter yang digunakan:**

✅ Akurasi Model: 0.9765 (97.65%)
""")
    
    st.code("""
    Parameter terbaik: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}
    """, language="python")

    col1, col2 = st.columns(2)
    with col1:
        st.image("ROC_RandomForest.png", caption="AUC & ROC Random Forest", use_column_width=True)
    with col2:
        st.image("CM_RandomForest.png", caption="Confusion Matrix Random Forest", use_column_width=True)
        
# === Logistic Regression ===
with st.expander("🖼️ Logistic Regression", expanded=False):
    st.markdown("""
**Berikut parameter yang digunakan:**

✅ Akurasi Prediksi Keseluruhan: 0.9671 (96.71%)
""")
    
    st.code("""
    Best Parameters: {'C': 0.01, 'l1_ratio': 0.0, 'max_iter': 500, 'penalty': 'none', 'solver': 'saga'}
    """, language="python")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image("ROC_LogisticRegression.png", caption="AUC & ROC Logistic Regression", use_column_width=True)
    with col2:
        st.image("CM_LogisticRegression.png", caption="Confusion Matrix Logistic Regression", use_column_width=True)

# === MLP ===
with st.expander("🖼️ MLP", expanded=False):
    st.markdown("""
**Berikut parameter yang digunakan:**

✅ Akurasi Model MLP: 0.9577 (95.77%)
""")
    
    st.code("""
    Best Parameters: {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (128, 64), 'learning_rate': 'constant', 'max_iter': 300, 'solver': 'adam'}
    """, language="python")

    col1, col2 = st.columns(2)
    with col1:
        st.image("ROC_MLP.png", caption="AUC & ROC MLP", use_column_width=True)
    with col2:
        st.image("CM_MLP.png", caption="Confusion Matrix MLP", use_column_width=True)

# === LightGBM ===
with st.expander("🖼️ Light GBM", expanded=False):
    st.markdown("""
**Berikut parameter yang digunakan:**

✅ Akurasi Prediksi Keseluruhan: 0.9671 (96.71%)
""")
    
    st.code("""
    Best Parameters: {'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 100, 'subsample': 0.8}
    """, language="python")

    col1, col2 = st.columns(2)
    with col1:
        st.image("ROC_LightGBM.png", caption="AUC & ROC LightGBM", use_column_width=True)
    with col2:
        st.image("CM_LightGBM.png", caption="Confusion Matrix LightGBM", use_column_width=True)

# === XGBoost ===
with st.expander("🖼️ XGBoost", expanded=False):
    st.markdown("""
**Berikut parameter yang digunakan:**

✅ Akurasi Model: 0.9765 (97.65%)
""")
    
    st.code("""
    Parameter terbaik: {'n_estimators: 150', 'max_depth: 6', 'learning_rate: 0.1', 'subsample: 0.8', 'colsample_bytree: 0.8'}
    """, language="python")

    col1, col2 = st.columns(2)
    with col1:
        st.image("ROC_XGBoost.png", caption="AUC & ROC XGBoost", use_column_width=True)
    with col2:
        st.image("CM_XGBoost.png", caption="Confusion Matrix XGBoost", use_column_width=True)

# === SVM ===
with st.expander("🖼️ SVM", expanded=False):
    st.markdown("""
**Berikut parameter yang digunakan:**

✅ Akurasi Prediksi Keseluruhan: 0.9859 (98.59%)
""")
    
    st.code("""
    Best Params: {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}
    """, language="python")
    col1, col2 = st.columns(2)
    with col1:
        st.image("ROC_SVM.png", caption="AUC & ROC SVM", use_column_width=True)
    with col2:
        st.image("CM_SVM.png", caption="Confusion Matrix SVM", use_column_width=True)


