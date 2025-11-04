import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # For sample data; replace with your real data

st.title("Chromatogram Grapher & Labeler")

# Upload file or load data
uploaded_file = st.file_uploader("Upload your chromatogram data (CSV)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview:", df.head())

    # Sample plot (adapt to your columns, e.g., time vs intensity)
    fig, ax = plt.subplots()
    ax.plot(df['time'], df['intensity'])  # Replace with your column names
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Intensity")
    # Labeling peaks (example: assume peaks in a 'peaks' column)
    if 'peaks' in df.columns:
        for peak in df['peaks']:
            ax.annotate('Peak', xy=(peak, df.loc[df['time']==peak, 'intensity']), xytext=(5, 5))
    st.pyplot(fig)

# Sidebar for options
st.sidebar.header("Labeling Options")
auto_label = st.sidebar.checkbox("Auto-label peaks")
if auto_label:
    st.info("Peaks labeled automatically!")
