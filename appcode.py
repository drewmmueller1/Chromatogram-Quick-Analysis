import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Page config
st.set_page_config(page_title="GC Data Processor", layout="wide")

st.title("GC Chromatogram Processor")

# Upload mode selection
upload_mode = st.radio("Upload Mode:", ["Single combined file", "Multiple single chromatogram files"])

df = None
if upload_mode == "Single combined file":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Combined file loaded successfully!")
elif upload_mode == "Multiple single chromatogram files":
    uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)
    if uploaded_files:
        # Read first file for retention time
        first_file = uploaded_files[0]
        df_first = pd.read_csv(first_file, skiprows=3, header=None, names=['Retention_Time', 'Intensity'])
        retention_time = df_first['Retention_Time']
        
        # Initialize combined DataFrame
        combined_df = pd.DataFrame({'Retention_Time': retention_time})
        
        # Process each file
        for file in uploaded_files:
            df_temp = pd.read_csv(file, skiprows=3, header=None, names=['Retention_Time', 'Intensity'])
            intensity = df_temp['Intensity']
            
            # Column name from filename
            filename = file.name.replace('.csv', '')
            combined_df[filename] = intensity
        
        df = combined_df
        st.success(f"Combined {len(uploaded_files)} files into one DataFrame.")

if df is not None:
    st.success("Data loaded successfully!")
    
    # Display basic info
    st.subheader("Data Overview")
    st.write(f"Shape: {df.shape}")
    st.dataframe(df.head())
    
    # Initialize peaks if not present
    if 'peaks' not in st.session_state:
        st.session_state.peaks = []
    
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
        
        # Reconstruct scaled df correctly
        scaled_df.iloc[:, 1:] = chrom_scaled
    
    st.success(f"Data scaled using '{scale_method}' method.")
    
    # Chromatogram selection
    st.subheader("Plot Individual Chromatogram")
    chrom_cols = scaled_df.columns[1:].tolist()
    selected_chrom = st.selectbox("Select chromatogram:", options=chrom_cols)
    
    if selected_chrom:
        rt = scaled_df.iloc[:, 0]
        y_data = scaled_df[selected_chrom]
        
        # Sidebar for customization
        st.sidebar.header("Graph Customization")
        
        custom_title = st.sidebar.text_input("Graph Title:", value=f"Chromatogram: {selected_chrom}")
        custom_xlabel = st.sidebar.text_input("X-axis Label:", value="Retention Time")
        custom_ylabel = st.sidebar.text_input("Y-axis Label:", value="Intensity")
        
        # Tick intervals
        major_x_step = st.sidebar.number_input("Major X Tick Step:", value=1.0, key="major_x")
        minor_x_step = st.sidebar.number_input("Minor X Tick Step:", value=0.1, key="minor_x")
        major_y_step = st.sidebar.number_input("Major Y Tick Step:", value=0.1, key="major_y")
        minor_y_step = st.sidebar.number_input("Minor Y Tick Step:", value=0.01, key="minor_y")
        
        # Axis bounds in sidebar
        x_min = st.sidebar.number_input("X-axis min (RT):", value=float(scaled_df.iloc[0, 0]), key="x_min")
        x_max = st.sidebar.number_input("X-axis max (RT):", value=float(scaled_df.iloc[-1, 0]), key="x_max")
        y_min = st.sidebar.number_input("Y-axis min:", value=0.0, key="y_min")
        y_max = st.sidebar.number_input("Y-axis max:", value=float(scaled_df[selected_chrom].max()) * 1.1, key="y_max")
        
        # Peak labels section
        st.subheader("Add Peak Labels")
        col1, col2 = st.columns(2)
        with col1:
            compound = st.text_input("Compound Name:")
        with col2:
            rt_peak = st.number_input("Retention Time:", value=0.0, key="rt_peak_input")
        
        if st.button("Add Peak"):
            if compound.strip():
                st.session_state.peaks.append({
                    "compound": compound.strip(), 
                    "rt": rt_peak, 
                    "label_x": rt_peak, 
                    "label_y": np.nan
                })
                st.rerun()
        
        # Option to clear peaks
        if st.button("Clear All Peaks"):
            st.session_state.peaks = []
            st.rerun()
        
        # Display and edit peaks table if any
        if st.session_state.peaks:
            df_peaks = pd.DataFrame(st.session_state.peaks)
            
            if st.button("Auto-calculate label positions"):
                for idx in df_peaks.index:
                    rt_target = df_peaks.loc[idx, 'rt']
                    # Find closest index
                    closest_idx = np.argmin(np.abs(rt - rt_target))
                    y_val = y_data.iloc[closest_idx]
                    offset = (y_max - y_min) * 0.05  # Larger offset for better spacing
                    df_peaks.loc[idx, 'label_x'] = rt_target
                    df_peaks.loc[idx, 'label_y'] = y_val + offset
                st.session_state.peaks = df_peaks.to_dict('records')
                st.rerun()
            
            # Data editor for editing positions (note: avoid editing compound and rt)
            st.info("Edit label positions below. Do not change compound or RT columns.")
            edited_df = st.data_editor(
                df_peaks,
                column_config={
                    "compound": st.column_config.TextColumn("Compound"),
                    "rt": st.column_config.NumberColumn("Retention Time", format="%.2f"),
                    "label_x": st.column_config.NumberColumn("Label X Position"),
                    "label_y": st.column_config.NumberColumn("Label Y Position"),
                },
                num_rows="fixed",
                use_container_width=True
            )
            
            # Update session_state if edited
            if not edited_df.equals(df_peaks):
                # Preserve original compound and rt if somehow changed, but update positions
                for i, row in edited_df.iterrows():
                    st.session_state.peaks[i]['label_x'] = row['label_x']
                    st.session_state.peaks[i]['label_y'] = row['label_y']
                st.rerun()
            
            # Remove single peak
            st.subheader("Remove Peak")
            sorted_peaks = sorted(st.session_state.peaks, key=lambda x: x['rt'])
            if sorted_peaks:
                peak_options = [f"Peak #{i+1}: {p['compound']} ({p['rt']:.2f})" 
                                for i, p in enumerate(sorted_peaks)]
                peak_to_remove = st.selectbox("Select peak to remove:", options=peak_options)
                if st.button("Remove Selected Peak") and peak_to_remove:
                    remove_str = peak_to_remove
                    # Find matching peak by compound and rt
                    for idx, peak in enumerate(st.session_state.peaks):
                        if f"{peak['compound']} ({peak['rt']:.2f})" in remove_str:
                            del st.session_state.peaks[idx]
                            break
                    st.rerun()
        
        # Main plot area
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter data for bounds
        mask = (rt >= x_min) & (rt <= x_max)
        ax.plot(rt[mask], y_data[mask], linewidth=1)
        
        # Set limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Custom labels and title
        ax.set_xlabel(custom_xlabel)
        ax.set_ylabel(custom_ylabel)
        ax.set_title(custom_title)
        
        # No grid
        ax.grid(False)
        
        # Custom ticks
        ax.xaxis.set_major_locator(MultipleLocator(major_x_step))
        ax.xaxis.set_minor_locator(MultipleLocator(minor_x_step))
        ax.yaxis.set_major_locator(MultipleLocator(major_y_step))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_y_step))
        
        # Add peak labels if any
        if st.session_state.peaks:
            sorted_peaks = sorted(st.session_state.peaks, key=lambda x: x['rt'])
            for i, peak in enumerate(sorted_peaks):
                rt_target = peak["rt"]
                lx = peak.get('label_x', rt_target)
                ly = peak.get('label_y')
                if np.isnan(ly):
                    # Calculate if not set
                    mask_peak = (rt >= x_min) & (rt <= x_max)
                    rt_in_bounds = rt[mask_peak]
                    if len(rt_in_bounds) > 0:
                        idx = np.argmin(np.abs(rt_in_bounds - rt_target))
                        actual_rt = rt_in_bounds.iloc[idx]
                        y_val = y_data[mask_peak].iloc[idx]
                        ly = y_val + (y_max - y_min) * 0.02
                    else:
                        continue
                if x_min <= lx <= x_max and y_min <= ly <= y_max:
                    ax.annotate(str(i+1), (lx, ly), ha='center', fontsize=10, fontweight='bold')
        
        st.pyplot(fig)
else:
    st.info("Please upload files to get started.")
