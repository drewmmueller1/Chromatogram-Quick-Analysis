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

@st.cache_data
def load_csv(uploaded_file):
    # Load with float32 to reduce memory usage
    return pd.read_csv(uploaded_file, dtype=np.float32)

if uploaded_file is not None:
    # Load data
    df = load_csv(uploaded_file)
    st.success("Data loaded successfully!")
    
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

    @st.cache_data
    def apply_scaling(df, scale_method):
        if scale_method == "None":
            return df
        rt_col = df.iloc[:, 0].values
        chrom_data = df.iloc[:, 1:].values  # Already float32
        
        if scale_method == "Min/Max":
            chrom_min = np.min(chrom_data, axis=0)
            chrom_max = np.max(chrom_data, axis=0)
            chrom_scaled = (chrom_data - chrom_min) / (chrom_max - chrom_min + 1e-8)
        
        elif scale_method == "Sum":
            sums = np.sum(chrom_data, axis=0)
            chrom_scaled = chrom_data / (sums + 1e-8)
        
        elif scale_method == "Square Root Sum of Squares":
            sq_sums = np.sqrt(np.sum(chrom_data**2, axis=0))
            chrom_scaled = chrom_data / (sq_sums + 1e-8)
        
        # Reconstruct without full copy: stack RT and scaled data
        scaled_data = np.column_stack([rt_col[:, np.newaxis], chrom_scaled])
        columns = df.columns
        return pd.DataFrame(scaled_data, columns=columns)
    
    # Apply scaling
    scaled_df = apply_scaling(df, scale_method)
    del df  # Free original dataframe memory
    
    if scale_method != "None":
        st.success(f"Data scaled using '{scale_method}' method.")
    
    # Chromatogram selection
    st.subheader("Plot Individual Chromatogram")
    chrom_cols = scaled_df.columns[1:].tolist()
    selected_chrom = st.selectbox("Select chromatogram:", options=chrom_cols)
    
    if selected_chrom:
        rt = scaled_df.iloc[:, 0].values  # Convert to numpy early
        y_data = scaled_df[selected_chrom].values  # Convert to numpy early
        
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
        
        mask = (rt >= x_min) & (rt <= x_max)
        rt_masked = rt[mask]
        y_masked = y_data[mask]
        
        # No downsampling: use full masked data
        rt_plot = rt_masked
        y_plot = y_masked
        
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
                    "compound": st.column_config.TextColumn("Compound", disabled=True),
                    "rt": st.column_config.NumberColumn("Retention Time", format="%.2f", disabled=True),
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
        
        # Sort peaks once for plotting and table
        sorted_peaks = sorted(st.session_state.peaks, key=lambda x: x['rt']) if st.session_state.peaks else []
        
        # Cached plot creation
        @st.cache_data
        def create_plot(rt_plot, y_plot, x_min, x_max, y_min, y_max, custom_title, custom_xlabel, custom_ylabel,
                        major_x_step, minor_x_step, major_y_step, minor_y_step, sorted_peaks, rt_masked, y_masked):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(rt_plot, y_plot, linewidth=1)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(custom_xlabel)
            ax.set_ylabel(custom_ylabel)
            ax.set_title(custom_title)
            ax.grid(False)
            ax.xaxis.set_major_locator(MultipleLocator(major_x_step))
            ax.xaxis.set_minor_locator(MultipleLocator(minor_x_step))
            ax.yaxis.set_major_locator(MultipleLocator(major_y_step))
            ax.yaxis.set_minor_locator(MultipleLocator(minor_y_step))
            
            # Peak annotations
            if sorted_peaks and len(rt_masked) > 0:
                for i, peak in enumerate(sorted_peaks):
                    rt_target = peak["rt"]
                    lx = peak.get('label_x', rt_target)
                    ly = peak.get('label_y', np.nan)
                    if np.isnan(ly):
                        # Calculate using full masked data
                        diffs = np.abs(rt_masked - rt_target)
                        idx = np.argmin(diffs)
                        y_val = y_masked[idx]
                        ly = y_val + (y_max - y_min) * 0.02
                    if x_min <= lx <= x_max and y_min <= ly <= y_max:
                        ax.annotate(str(i+1), (lx, ly), ha='center', fontsize=10, fontweight='bold')
            
            return fig
        
        # Main plot area
        fig = create_plot(rt_plot, y_plot, x_min, x_max, y_min, y_max, custom_title, custom_xlabel, custom_ylabel,
                          major_x_step, minor_x_step, major_y_step, minor_y_step, sorted_peaks, rt_masked, y_masked)
        st.pyplot(fig)
        
        # Display peaks table under graph if peaks exist
        if sorted_peaks:
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
        @st.cache_data
        def get_csv_data(scaled_df, scale_method):
            csv_buffer = io.StringIO()
            scaled_df.to_csv(csv_buffer, index=False)
            return csv_buffer.getvalue()
        
        csv_data = get_csv_data(scaled_df, scale_method)
        st.download_button(
            label="Download Normalized Data as CSV",
            data=csv_data,
            file_name=f"normalized_data_{scale_method.replace('/', '_').replace(' ', '_')}.csv",
            mime="text/csv"
        )
        
        # Export peaks table if peaks exist
        if st.session_state.peaks:
            @st.cache_data
            def get_peaks_csv(peaks):
                sorted_peaks_local = sorted(peaks, key=lambda x: x['rt'])
                peaks_export_df = pd.DataFrame({
                    "Peak #": range(1, len(sorted_peaks_local) + 1),
                    "Compound": [p["compound"] for p in sorted_peaks_local],
                    "Retention Time": [p["rt"] for p in sorted_peaks_local]
                })
                csv_buffer = io.StringIO()
                peaks_export_df.to_csv(csv_buffer, index=False)
                return csv_buffer.getvalue()
            
            csv_data_peaks = get_peaks_csv(st.session_state.peaks)
            st.download_button(
                label="Download Peaks Table as CSV",
                data=csv_data_peaks,
                file_name="peaks_table.csv",
                mime="text/csv"
            )
else:
    st.info("Please upload a CSV file to get started.")
