import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import io
import os

# Page config
st.set_page_config(page_title="GC Data Processor", layout="wide")

st.title("GC Chromatogram Processor")

# File uploader for multiple files
uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)

if uploaded_files is not None and len(uploaded_files) > 0:
    # Load and combine data
    combined_df = None
    file_names = []
    
    for uploaded_file in uploaded_files:
        # Extract filename for column naming
        file_name = uploaded_file.name.replace('.csv', '').replace(' ', '_')
        file_names.append(file_name)
        
        # Load data starting from row 4 (skip first 3 rows)
        df_temp = pd.read_csv(uploaded_file, skiprows=3)
        
        # Assume first column is RT, second is data
        if len(df_temp.columns) < 2:
            st.error(f"File {file_name} does not have at least 2 columns.")
            continue
        
        # Rename columns for clarity
        df_temp.columns = ['RT', file_name] if len(df_temp.columns) == 2 else [df_temp.columns[0], file_name] + list(df_temp.columns[2:])
        
        # Keep only RT and the data column (assuming second column is the main data)
        df_temp = df_temp[['RT', file_name]]
        
        if combined_df is None:
            # First file: set as base
            combined_df = df_temp
        else:
            # Append data column to combined
            combined_df[file_name] = df_temp[file_name]
    
    if combined_df is None:
        st.error("No valid data loaded.")
    else:
        # Rename RT column to 'Retention Time' or keep as is; current code assumes first col is RT
        if combined_df.columns[0] != 'Retention Time':
            combined_df = combined_df.rename(columns={combined_df.columns[0]: 'Retention Time'})
        
        df = combined_df
        st.success(f"Data from {len(uploaded_files)} files combined successfully!")
        
        # Display basic info
        st.subheader("Data Overview")
        st.write(f"Shape: {df.shape}")
        st.write(f"Files combined: {', '.join(file_names)}")
        st.dataframe(df.head())
        
        # Initialize peaks if not present
        if 'peaks' not in st.session_state:
            st.session_state.peaks = []
        
        # Sidebar for options
        with st.sidebar:
            st.header("Options")
            
            # Scaling options in expander
            with st.expander("Scaling Options", expanded=False):
                scale_method = st.radio(
                    "Select scaling method for chromatograms:",
                    options=["None", "Min/Max", "Sum", "Square Root Sum of Squares"]
                )
            
            # Peak management in expander
            with st.expander("Peak Management", expanded=False):
                if st.button("Clear All Peaks"):
                    st.session_state.peaks = []
                    st.rerun()
                
                if st.session_state.peaks:
                    sorted_peaks = sorted(st.session_state.peaks, key=lambda x: x['rt'])
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
            
            # Graph customization in sidebar
            with st.sidebar:
                with st.expander("Graph Customization", expanded=False):
                    custom_title = st.text_input("Graph Title:", value=f"Chromatogram: {selected_chrom}")
                    custom_xlabel = st.text_input("X-axis Label:", value="Retention Time")
                    custom_ylabel = st.text_input("Y-axis Label:", value="Intensity")
                    
                    # Tick intervals
                    major_x_step = st.number_input("Major X Tick Step:", value=1.0, key="major_x")
                    minor_x_step = st.number_input("Minor X Tick Step:", value=0.1, key="minor_x")
                    major_y_step = st.number_input("Major Y Tick Step:", value=0.1, key="major_y")
                    minor_y_step = st.number_input("Minor Y Tick Step:", value=0.01, key="minor_y")
                    
                    # Axis bounds
                    x_min = st.number_input("X-axis min (RT):", value=float(rt.min()), key="x_min")
                    x_max = st.number_input("X-axis max (RT):", value=float(rt.max()), key="x_max")
                    y_min = st.number_input("Y-axis min:", value=0.0, key="y_min")
                    y_max = st.number_input("Y-axis max:", value=float(y_data.max()) * 1.1, key="y_max")
            
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
            
            # Display and edit peaks table if any
            if st.session_state.peaks:
                df_peaks = pd.DataFrame(st.session_state.peaks)
                
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
            
            # Display peaks table under graph if peaks exist
            if st.session_state.peaks:
                sorted_peaks = sorted(st.session_state.peaks, key=lambda x: x['rt'])
                peaks_display_df = pd.DataFrame({
                    "Peak #": range(1, len(sorted_peaks) + 1),
                    "Compound": [p["compound"] for p in sorted_peaks],
                    "Retention Time": [p["rt"] for p in sorted_peaks]
                })
                st.subheader("Peaks Table")
                st.dataframe(peaks_display_df, hide_index=True)
            
            # Export section after table
            st.subheader("Export Data")
            
            # Export normalized data
            csv_buffer = io.StringIO()
            scaled_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            st.download_button(
                label="Download Normalized Data as CSV",
                data=csv_data,
                file_name=f"combined_normalized_data_{scale_method.replace('/', '_')}.csv",
                mime="text/csv"
            )
            
            # Export peaks table if peaks exist
            if st.session_state.peaks:
                sorted_peaks = sorted(st.session_state.peaks, key=lambda x: x['rt'])
                peaks_export_df = pd.DataFrame({
                    "Peak #": range(1, len(sorted_peaks) + 1),
                    "Compound": [p["compound"] for p in sorted_peaks],
                    "Retention Time": [p["rt"] for p in sorted_peaks]
                })
                
                csv_buffer_peaks = io.StringIO()
                peaks_export_df.to_csv(csv_buffer_peaks, index=False)
                csv_data_peaks = csv_buffer_peaks.getvalue()
                st.download_button(
                    label="Download Peaks Table as CSV",
                    data=csv_data_peaks,
                    file_name="peaks_table.csv",
                    mime="text/csv"
                )
else:
    st.info("Please upload one or more CSV files to get started.")
