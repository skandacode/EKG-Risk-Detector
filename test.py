import numpy as np
import matplotlib.pyplot as plt
import wfdb
import neurokit2 as nk
import os
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

# Configuration
record_name = 'slp02a'
database = 'slpdb'
local_data_dir = 'data'  # Local directory to store downloaded records

# Create local data directory if it doesn't exist
os.makedirs(local_data_dir, exist_ok=True)

# Check if record exists locally
local_record_path = os.path.join(local_data_dir, record_name)
if os.path.exists(f"{local_record_path}.dat") and os.path.exists(f"{local_record_path}.hea"):
    print(f"Loading {record_name} from local storage...")
    record = wfdb.rdrecord(local_record_path)
else:
    print(f"Downloading {record_name} from PhysioNet...")
    # Download the record
    record = wfdb.rdrecord(record_name, pn_dir=database)
    # Save locally
    wfdb.wrsamp(record_name, fs=record.fs, units=record.units, sig_name=record.sig_name, 
                p_signal=record.p_signal, fmt=record.fmt, write_dir=local_data_dir)
    print(f"Record saved to {local_data_dir}/")

# Find ECG channel (usually labeled as 'ECG' or similar)
ecg_channel_idx = None
for i, sig_name in enumerate(record.sig_name):
    if 'ECG' in sig_name.upper():
        ecg_channel_idx = i
        break

if ecg_channel_idx is None:
    print("Available channels:", record.sig_name)
    ecg_channel_idx = 0  # Use first channel if ECG not found
    print(f"ECG channel not found, using channel {ecg_channel_idx}: {record.sig_name[ecg_channel_idx]}")

# Extract ECG signal and sampling rate
ecg_signal = record.p_signal[:, ecg_channel_idx]
sampling_rate = record.fs

# Use first 1000 samples for analysis
ecg_signal = ecg_signal[0:1000]

# Create time array in seconds
time_array = np.arange(len(ecg_signal)) / sampling_rate

print("Length of signal", len(ecg_signal))
print(f"Duration of signal: {len(ecg_signal) / sampling_rate:.2f} seconds")
# Remove NaN values if any
ecg_signal = ecg_signal[~np.isnan(ecg_signal)]

# Process ECG signal to detect R-peaks first
signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)


# Perform wave delineation separately
_, waves_peak = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
waves_delineated = nk.ecg_delineate(ecg_signal, waves_peak, sampling_rate=sampling_rate, method="dwt")

# The delineation returns a tuple, get the second element which contains the waves info
if isinstance(waves_delineated, tuple):
    waves_info = waves_delineated[1]  # Get the waves dictionary
else:
    waves_info = waves_delineated

# Create a wave type array - initialize with "baseline"
wave_types = np.zeros(len(ecg_signal), dtype=int)  # 0 = baseline

# Assign wave types based on delineation using the info dictionary
# P wave = 1, Q wave = 2, R wave = 3, S wave = 4, T wave = 5

# Use R peaks from the peak detection
if "ECG_R_Peaks" in waves_peak:
    for peak in waves_peak["ECG_R_Peaks"]:
        if peak < len(wave_types):
            # Mark a small region around the R peak
            start = max(0, peak - int(0.01 * sampling_rate))
            end = min(len(wave_types) - 1, peak + int(0.01 * sampling_rate))
            wave_types[start:end+1] = 3  # R wave

# Use the info dictionary from ecg_process for other waves
if "ECG_P_Peaks" in info and info["ECG_P_Peaks"] is not None:
    p_peaks = [peak for peak in info["ECG_P_Peaks"] if not np.isnan(peak)]
    for peak in p_peaks:
        peak = int(peak)
        if peak < len(wave_types):
            # Mark a small region around the P peak
            start = max(0, peak - int(0.02 * sampling_rate))
            end = min(len(wave_types) - 1, peak + int(0.02 * sampling_rate))
            wave_types[start:end+1] = 1  # P wave

if "ECG_Q_Peaks" in info and info["ECG_Q_Peaks"] is not None:
    q_peaks = [peak for peak in info["ECG_Q_Peaks"] if not np.isnan(peak)]
    for peak in q_peaks:
        peak = int(peak)
        if peak < len(wave_types):
            # Mark a small region around the Q peak
            start = max(0, peak - int(0.01 * sampling_rate))
            end = min(len(wave_types) - 1, peak + int(0.01 * sampling_rate))
            wave_types[start:end+1] = 2  # Q wave

if "ECG_S_Peaks" in info and info["ECG_S_Peaks"] is not None:
    s_peaks = [peak for peak in info["ECG_S_Peaks"] if not np.isnan(peak)]
    for peak in s_peaks:
        peak = int(peak)
        if peak < len(wave_types):
            # Mark a small region around the S peak
            start = max(0, peak - int(0.01 * sampling_rate))
            end = min(len(wave_types) - 1, peak + int(0.01 * sampling_rate))
            wave_types[start:end+1] = 4  # S wave

if "ECG_T_Peaks" in info and info["ECG_T_Peaks"] is not None:
    t_peaks = [peak for peak in info["ECG_T_Peaks"] if not np.isnan(peak)]
    for peak in t_peaks:
        peak = int(peak)
        if peak < len(wave_types):
            # Mark a small region around the T peak
            start = max(0, peak - int(0.02 * sampling_rate))
            end = min(len(wave_types) - 1, peak + int(0.02 * sampling_rate))
            wave_types[start:end+1] = 5  # T wave


# Create colored line plot
fig, ax = plt.subplots(figsize=(12, 6))

# Create points using time array instead of sample indices
points = np.array([time_array, ecg_signal]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Create a custom colormap for the wave types
colors = ['gray', 'green', 'blue', 'red', 'purple', 'orange']  # baseline, P, Q, R, S, T
cmap = ListedColormap(colors)

# Create a line collection
lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 5))
lc.set_array(wave_types[:-1])
line = ax.add_collection(lc)

# Set plot limits using time
ax.set_xlim(0, time_array[-1])
ax.set_ylim(np.min(ecg_signal)-0.1, np.max(ecg_signal)+0.1)

# Add zero line
ax.axhline(0, color='gray', linestyle='--', linewidth=1, label='Zero Line')

# Add labels and title
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Amplitude")
ax.set_title(f"ECG Signal with P-QRS-T Waves - {record_name}")

# Add a legend
from matplotlib.lines import Line2D
custom_lines = [
    Line2D([0], [0], color='gray', lw=2),
    Line2D([0], [0], color='green', lw=2),
    Line2D([0], [0], color='blue', lw=2),
    Line2D([0], [0], color='red', lw=2),
    Line2D([0], [0], color='purple', lw=2),
    Line2D([0], [0], color='orange', lw=2),
    Line2D([0], [0], color='gray', linestyle='--', lw=1)
]
ax.legend(custom_lines, ['Baseline', 'P Wave', 'Q Wave', 'R Wave', 'S Wave', 'T Wave', 'Zero Line'])

plt.tight_layout()
plt.show()

# Print summary information
print(f"Record: {record_name}")
print(f"Sampling rate: {sampling_rate} Hz")
print("P waves detected:", len([p for p in info.get("ECG_P_Peaks", []) if not np.isnan(p)]))
print("QRS complexes detected:", len(waves_peak.get("ECG_R_Peaks", [])))
print("T waves detected:", len([t for t in info.get("ECG_T_Peaks", []) if not np.isnan(t)]))

# Print timing information for detected peaks
if "ECG_R_Peaks" in waves_peak:
    r_peak_times = np.array(waves_peak["ECG_R_Peaks"]) / sampling_rate
    print(f"R peaks at times (s): {r_peak_times[:10]}...")  # Show first 10

if "ECG_P_Peaks" in info and info["ECG_P_Peaks"] is not None:
    p_peak_times = np.array([p for p in info["ECG_P_Peaks"] if not np.isnan(p)]) / sampling_rate
    if len(p_peak_times) > 0:
        print(f"P peaks at times (s): {p_peak_times[:10]}...")  # Show first 10

# Plot time differences (RR intervals) between R peaks
if "ECG_R_Peaks" in waves_peak and len(waves_peak["ECG_R_Peaks"]) > 1:
    r_peaks = np.array(waves_peak["ECG_R_Peaks"])
    rr_intervals = np.diff(r_peaks) / sampling_rate  # in seconds
    rr_times = r_peaks[1:] / sampling_rate  # time (s) of each RR interval

    plt.figure(figsize=(10, 4))
    plt.plot(rr_times, rr_intervals, marker='o')
    plt.xlabel("Time (s)")
    plt.ylabel("RR Interval (s)")
    plt.title("Time Differences Between R Peaks (RR Intervals)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Print RR interval statistics
    print(f"Mean RR interval: {np.mean(rr_intervals):.3f} seconds")
    print(f"Heart rate (average): {60 / np.mean(rr_intervals):.1f} BPM")
else:
    print("Not enough R peaks detected to plot RR intervals.")