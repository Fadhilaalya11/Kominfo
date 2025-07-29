import streamlit as st 

st.title("Code Best Model SVM carbon stock calculation")

# === import ===
with st.expander("🖼️ Package", expanded=False): 
    st.code("""
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    from shapely.geometry import box
    import fiona
    import os
    import sklearn
    import folium
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
    from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, ConfusionMatrixDisplay
    from imblearn.over_sampling import RandomOverSampler
    from folium.features import GeoJsonTooltip
    from sklearn.preprocessing import label_binarize, LabelEncoder
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    """, language="python")


