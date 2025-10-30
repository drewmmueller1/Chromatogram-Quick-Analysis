import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import io

# Page config
st.set_page_config(page_title="GC Data Processor", layout="wide")

st.title("GC Chromatogram Processor")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.success("Data loaded successfully!")
    
    # Display basic info
    st.subheader("Data Overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())
    
    # Scaling options
    st.subheader("Scaling Options")
    scale_method = st.radio(
        "Select scaling method for chromatograms:",
        options=["None", "Min/Max", "Sum", "Square Root Sum of Squares"]
    )
    
    # Apply scaling
    scaled_df = df.copy()
    if scale_method != "None":
        rt_col = df.iloc[:, 0]  # Retention time
        chrom_data = df.iloc[:, 1:].values  # Chromatogram columns
        
        if scale_method == "Min/Max":
            # Normalize each column to [0,1]
            chrom_min = np.min(chrom_data, axis=0)
            chrom_max = np.max(chrom_data, axis=0)
            chrom_scaled = (chrom_data - chrom_min) / (chrom_max - chrom_min + 1e-8)
        
        elif scale_method == "Sum":
            # Normalize by sum (area)
            sums = np.sum(chrom_data, axis=0)
            chrom_scaled = chrom_data / (sums + 1e-8)
        
        elif scale_method == "Square Root Sum of Squares":
            # RMS normalization
            sq_sums = np.sqrt(np.sum(chrom_data**2, axis=0))
            chrom_scaled = chrom_data / (sq_sums + 1e-8)
        
        # Reconstruct scaled df
        scaled_df = pd.DataFrame(chrom_scaled.T, columns=df.columns[1:])
        scaled_df.insert(0, df.columns[0], rt_col)
    
    st.success(f"Data scaled using '{scale_method}' method.")
    
    # Chromatogram selection
    st.subheader("Plot Individual Chromatogram")
    chrom_cols = scaled_df.columns[1:].tolist()
    selected_chrom = st.selectbox("Select chromatogram:", options=chrom_cols)
    
    if selected_chrom:
        # Axis bounds
        col1, col2 = st.columns(2)
        with col1:
            x_min = st.number_input("X-axis min (RT):", value=float(scaled_df.iloc[0, 0]), key="x_min")
            x_max = st.number_input("X-axis max (RT):", value=float(scaled_df.iloc[-1, 0]), key="x_max")
        with col2:
            y_min = st.number_input("Y-axis min:", value=0.0, key="y_min")
            y_max = st.number_input("Y-axis max:", value=float(scaled_df[selected_chrom].max()) * 1.1, key="y_max")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        rt = scaled_df.iloc[:, 0]
        y_data = scaled_df[selected_chrom]
        
        # Filter data for bounds if needed
        mask = (rt >= x_min) & (rt <= x_max)
        ax.plot(rt[mask], y_data[mask], linewidth=1)
        
        # Set limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # No grid
        ax.grid(False)
        
        # Ticks: major and minor
        ax.xaxis.set_major_locator(MultipleLocator(1))  # Adjust base on data
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.01))
        
        ax.set_xlabel("Retention Time")
        ax.set_ylabel("Intensity")
        ax.set_title(f"Chromatogram: {selected_chrom}")
        
        st.pyplot(fig)
    
    # Peak labels
    st.subheader("Add Peak Labels")
    label_input = st.text_area(
        "Enter peaks (format: Compound,RetentionTime per line, e.g., Benzene,5.2\nToluene,7.1):",
        placeholder="Compound1,RT1\nCompound2,RT2",
        height=150
    )
    
    if label_input:
        peaks = []
        for line in label_input.strip().split("\n"):
            if line.strip():
                parts = line.split(",")
                if len(parts) == 2:
                    compound = parts[0].strip()
                    try:
                        rt_peak = float(parts[1].strip())
                        peaks.append({"compound": compound, "rt": rt_peak})
                    except ValueError:
                        st.error(f"Invalid RT in line: {line}")
        
        if peaks:
            # Plot with labels
            fig, ax = plt.subplots(figsize=(10, 6))
            rt = scaled_df.iloc[:, 0]
            y_data = scaled_df[selected_chrom]
            
            # Filter data
            mask = (rt >= x_min) & (rt <= x_max)
            ax.plot(rt[mask], y_data[mask], linewidth=1)
            
            # Add labels
            label_numbers = list(range(1, len(peaks) + 1))
            for i, (peak, num) in enumerate(zip(peaks, label_numbers)):
                rt_target = peak["rt"]
                # Find closest index
                idx = np.argmin(np.abs(rt - rt_target))
                y_val = y_data.iloc[idx]
                offset = (y_max - y_min) * 0.02  # Small offset above peak
                ax.annotate(str(num), (rt.iloc[idx], y_val + offset), ha='center', fontsize=10, fontweight='bold')
            
            # Set limits and ticks as before
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(False)
            ax.xaxis.set_major_locator(MultipleLocator(1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.01))
            ax.set_xlabel("Retention Time")
            ax.set_ylabel("Intensity")
            ax.set_title(f"Chromatogram: {selected_chrom} with Peak Labels")
            
            st.pyplot(fig)
            
            # Table
            st.subheader("Peak Table")
            table_data = [{"Peak #": i+1, "Compound": p["compound"], "Retention Time": p["rt"]} for i, p in enumerate(peaks)]
            st.table(table_data)
else:
    st.info("Please upload a CSV file to get started.")