import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# Page config
st.set_page_config(page_title="Chromatogram Grapher & Labeler", layout="wide")

st.title("🚀 Chromatogram Grapher & Labeler")
st.markdown("Upload your chromatogram data (CSV format with 'time' and 'intensity' columns) to visualize and manually label peaks.")

# Sidebar for options
st.sidebar.header("📊 Plot Options")
show_labels = st.sidebar.checkbox("Show peak labels", value=True)
normalize = st.sidebar.checkbox("Normalize intensity", value=False)

st.sidebar.header("🏷️ Manual Peak Labeling")
manual_peaks_input = st.sidebar.text_area(
    "Enter peak times (comma-separated, e.g., 2.5, 5.0, 7.2):",
    placeholder="2.5,5.0,7.2",
    height=100
)

# Parse manual peaks
manual_peaks = []
if manual_peaks_input.strip():
    try:
        manual_peaks = [float(t.strip()) for t in manual_peaks_input.split(',')]
    except ValueError:
        st.sidebar.error("Invalid input: Please enter numbers separated by commas.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read the data
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded data: {len(df)} rows")
        st.write("Data preview:")
        st.dataframe(df.head())
        
        # Assume columns: if not present, use first two
        time_col = 'time' if 'time' in df.columns else df.columns[0]
        intensity_col = 'intensity' if 'intensity' in df.columns else df.columns[1]
        
        # Check if columns exist
        if time_col not in df.columns or intensity_col not in df.columns:
            st.error(f"Required columns '{time_col}' and '{intensity_col}' not found. Please ensure your CSV has 'time' and 'intensity' columns.")
            st.stop()
        
        # Normalize if selected
        intensity = df[intensity_col].values
        if normalize:
            intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
        
        time = df[time_col].values
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time, intensity, 'b-', linewidth=1, label='Chromatogram')
        
        if len(manual_peaks) > 0:
            # Filter peaks within time range
            valid_peaks = [p for p in manual_peaks if np.min(time) <= p <= np.max(time)]
            if valid_peaks:
                peak_indices = [np.argmin(np.abs(time - p)) for p in valid_peaks]
                peak_times = time[peak_indices]
                peak_intensities = intensity[peak_indices]
                ax.plot(peak_times, peak_intensities, "ro", label=f'Manual Peaks ({len(valid_peaks)})')
                
                if show_labels:
                    for i, (pt, pi) in enumerate(zip(peak_times, peak_intensities)):
                        ax.annotate(f'Peak {i+1}\n({pt:.2f}, {pi:.2f})', 
                                   xy=(pt, pi), xytext=(5, 5), 
                                   textcoords='offset points', 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel(f"{time_col} (min)")
        ax.set_ylabel(f"{intensity_col} (normalized)" if normalize else intensity_col)
        ax.set_title("Chromatogram with Manual Peak Labels")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Display plot
        st.pyplot(fig)
        
        # Show peak data
        if len(valid_peaks) > 0:
            peak_df = pd.DataFrame({
                'Peak #': range(1, len(valid_peaks) + 1),
                'Time (min)': valid_peaks,
                'Intensity': [intensity[np.argmin(np.abs(time - p))] for p in valid_peaks]
            })
            st.subheader("Manual Peaks")
            st.dataframe(peak_df)
            
            # Download peaks as CSV
            csv_buffer = io.StringIO()
            peak_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Peaks CSV",
                data=csv_buffer.getvalue(),
                file_name="manual_peaks.csv",
                mime="text/csv"
            )
        else:
            st.info("Add peak times in the sidebar to label them.")
            
    except Exception as e:
        st.error(f"Error loading file: {str(e)}. Please check your CSV format.")

else:
    # Sample data if no file uploaded
    st.info("👆 Upload a file to get started. Or try the sample below.")
    
    if st.button("Generate Sample Data"):
        # Sample chromatogram data
        t = np.linspace(0, 10, 1000)
        signal = np.exp(-((t - 2)**2)/0.1) + 0.5 * np.exp(-((t - 5)**2)/0.2) + np.random.normal(0, 0.05, len(t))
        sample_df = pd.DataFrame({'time': t, 'intensity': signal})
        
        csv_buffer = sample_df.to_csv(index=False).encode()
        st.download_button(
            label="📥 Download Sample CSV",
            data=csv_buffer,
            file_name="sample_chromatogram.csv",
            mime="text/csv"
        )
        
        # Plot sample
        fig_sample, ax_sample = plt.subplots()
        ax_sample.plot(t, signal)
        ax_sample.set_title("Sample Chromatogram")
        st.pyplot(fig_sample)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Pandas, Matplotlib & NumPy")
