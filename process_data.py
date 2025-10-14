import math
import pandas
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

# Plot control variables
PLOT_ECG_WITH_PEAKS = True
PLOT_RR_LINEAR = False
PLOT_RR_LOG = True
USE_NEUROKIT_RPEAK_DETECTION = True  # Set to True to use NeuroKit2's R-peak detection

# Constants
WINDOW_SIZE = 5
MIN_DISTANCE_MS = 100
HRV_WINDOW_SIZE = 20
HRV_HIGH_THRESHOLD = 0.1
HRV_LOW_THRESHOLD = 7e-3
HRV_LOW_COUNT_THRESHOLD = 5
HOUR_SECONDS = 3600
NORMAL_HR_MIN = 60
NORMAL_HR_MAX = 100
DOWNSAMPLE_FACTOR = 10
DYNAMIC_THRESHOLD_FACTOR = 2.5
HRV_WINDOW_SIZE_MEDIAN = 100


def load_data():
    df_preview = pandas.read_csv(
        'data.csv',
        quotechar='"',
        skipinitialspace=True,
        nrows=2
    )
    time0 = pandas.to_datetime(df_preview['TIME'].iloc[0])
    time1 = pandas.to_datetime(df_preview['TIME'].iloc[1])
    sampling_rate = 1 / (time1 - time0).total_seconds()

    df = pandas.read_csv(
        'data.csv',
        quotechar='"',
        skipinitialspace=True
    )

    num_samples = len(df)
    df['SECONDS_SINCE_START'] = np.arange(num_samples) / sampling_rate
    df.set_index('SECONDS_SINCE_START', inplace=True)

    print(df)

    print(f"sampling rate: {sampling_rate} hz")


    # Invert ECG
    df['ECG'] = -df['ECG']
    return df, sampling_rate

def load_healthy_data():
    import test_with_db
    ecg_signal, sampling_rate = test_with_db.load_data()
    df = pandas.DataFrame({
        'ECG': ecg_signal
    })
    df['SECONDS_SINCE_START'] = np.arange(len(df)) / sampling_rate
    df.set_index('SECONDS_SINCE_START', inplace=True)
    print(df)
    print(f"sampling rate: {sampling_rate} hz")
    return df, sampling_rate

df, sampling_rate = load_healthy_data()

window_size = round(sampling_rate * WINDOW_SIZE)

df['moving_median'] = df['ECG'].rolling(window=window_size, center=True).median()
df['standard dev'] = df['ECG'].rolling(window=window_size, center=True).std()

df['dynamic_threshold'] = df['moving_median'] + DYNAMIC_THRESHOLD_FACTOR * df['standard dev']

min_distance_ms = MIN_DISTANCE_MS
min_distance_samples = int(min_distance_ms * sampling_rate / 1000)

# R-peak detection - choose between custom implementation and NeuroKit2
if USE_NEUROKIT_RPEAK_DETECTION:
    # Use NeuroKit2's R-peak detection
    print("Using NeuroKit2 R-peak detection...")
    try:
        # NeuroKit2 ecg_peaks returns a tuple (signals, info)
        signals, info = nk.ecg_peaks(df['ECG'].values, sampling_rate=sampling_rate)
        # Extract R-peak indices from the signals dictionary
        r_peak_indices = np.where(signals['ECG_R_Peaks'] == 1)[0]
        r_peak_times = df.index[r_peak_indices]
        print(f"Found {len(r_peak_indices)} R-peaks with NeuroKit2")
    except Exception as e:
        print(f"NeuroKit2 R-peak detection failed: {e}")
        print("Falling back to custom R-peak detection...")
        USE_NEUROKIT_RPEAK_DETECTION = False

if not USE_NEUROKIT_RPEAK_DETECTION:
    # Use custom R-peak detection
    print("Using custom R-peak detection...")
    r_peak_indices = []
    prev_peak = -min_distance_samples
    values = df['ECG'].values
    thresholds = df['dynamic_threshold'].values

    check_offset = 1

    for i in range(check_offset, len(values) - check_offset):
        if (not np.isnan(thresholds[i]) and 
            values[i] > thresholds[i] and 
            values[i] >= values[i-check_offset] and 
            values[i] >= values[i+check_offset]):
            if i - prev_peak >= min_distance_samples:
                r_peak_indices.append(i)
                prev_peak = i

    r_peak_times = df.index[r_peak_indices]
    print(f"Found {len(r_peak_indices)} R-peaks with custom detection")

rr_intervals = np.diff(r_peak_times)
rr_interval_times = r_peak_times[1:]


# Calculate HRV using rolling standard deviation
hrv_window_size = HRV_WINDOW_SIZE
rr_series = pandas.Series(rr_intervals, index=rr_interval_times)
hrv = rr_series.rolling(window=hrv_window_size, center=True).std()

downsample_factor = DOWNSAMPLE_FACTOR
downsample_indices = np.arange(0, len(df), downsample_factor)

hrv = hrv.rolling(window=HRV_WINDOW_SIZE_MEDIAN, center=True).median()

if PLOT_ECG_WITH_PEAKS:
    plt.figure()
    plt.plot(df.index, df['ECG'], label='ECG')
    
    # Only plot threshold-related lines when using custom R-peak detection
    if not USE_NEUROKIT_RPEAK_DETECTION:
        plt.plot(df.index[downsample_indices], df['moving_median'].iloc[downsample_indices], 
                 label='Moving Median (10s)', color='orange')
        plt.plot(df.index[downsample_indices], df['dynamic_threshold'].iloc[downsample_indices], 
                 label='Dynamic Threshold', color='green', linestyle='--')
        plt.plot(df.index[downsample_indices], df['standard dev'].iloc[downsample_indices], 
                 label='Standard Deviation', color='blue', linestyle='--')

    plt.scatter(r_peak_times, df['ECG'].iloc[r_peak_indices],
                color='red', label='R peaks')

    plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Zero')
    plt.xlabel('Seconds since start')
    plt.ylabel('Amplitude')
    plt.title('ECG')
    plt.legend()

if PLOT_RR_LINEAR:
    plt.figure()
    plt.plot(rr_interval_times, rr_intervals, label='RR interval')

    # Add HRV to linear scale plot
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(hrv.index, hrv.values, color='purple', linewidth=2, label='HRV (30)', alpha=0.8)
    ax2.set_ylabel('HRV (s)', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.xlabel('Seconds since start')
    plt.ylabel('Interval (s)')
    plt.title('RR Intervals and Heart Rate Variability over Time')

if PLOT_RR_LOG:
    plt.figure()
    plt.plot(rr_interval_times, rr_intervals, label='RR interval')
    plt.yscale('log')

    normal_hr_min = NORMAL_HR_MIN
    normal_hr_max = NORMAL_HR_MAX
    rr_max_normal = 60 / normal_hr_min
    rr_min_normal = 60 / normal_hr_max

    plt.axhline(rr_max_normal, color='green', linestyle='--', alpha=0.7, label='60 bpm (1.0s)')
    plt.axhline(rr_min_normal, color='red', linestyle='--', alpha=0.7, label='100 bpm (0.6s)')

    # Create secondary y-axis for HRV
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(hrv.index, hrv.values, color='purple', linewidth=2, label='HRV (30) Log Scale', alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_ylabel('HRV (s) - Log Scale', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Add HRV threshold line and highlight regions below 3e-3s
    hrv_threshold = HRV_LOW_THRESHOLD
    ax2.axhline(hrv_threshold, color='red', linestyle=':', alpha=0.8, label='HRV Threshold (3e-3s)')

    low_hrv_mask = hrv < hrv_threshold
    if low_hrv_mask.any():
        y_min, y_max = ax1.get_ylim()
        low_hrv_regions = []
        in_region = False
        start_idx = None
        for i, is_low in enumerate(low_hrv_mask):
            if is_low and not in_region:
                start_idx = i
                in_region = True
            elif not is_low and in_region:
                low_hrv_regions.append((hrv.index[start_idx], hrv.index[i-1]))
                in_region = False
        
        # Handle case where last region extends to end
        if in_region:
            low_hrv_regions.append((hrv.index[start_idx], hrv.index[-1]))
        
        # Highlight each region
        for start_time, end_time in low_hrv_regions:
            ax1.axvspan(start_time, end_time, alpha=0.3, color='blue', label='Low HRV (<3e-3s)' if start_time == low_hrv_regions[0][0] else "")

    # Highlight regions where HRV > 0.12s in red
    hrv_threshold = HRV_HIGH_THRESHOLD
    high_hrv_mask = hrv > hrv_threshold
    ax2.axhline(hrv_threshold, color='orange', linestyle=':', alpha=0.8, label='HRV Threshold (0.2s)')
    if high_hrv_mask.any():
        y_min, y_max = ax1.get_ylim()
        high_hrv_regions = []
        in_region = False
        start_idx = None
        for i, is_high in enumerate(high_hrv_mask):
            if is_high and not in_region:
                start_idx = i
                in_region = True
            elif not is_high and in_region:
                high_hrv_regions.append((hrv.index[start_idx], hrv.index[i-1]))
                in_region = False
        
        # Handle case where last region extends to end
        if in_region:
            high_hrv_regions.append((hrv.index[start_idx], hrv.index[-1]))
        
        # Highlight each region
        for start_time, end_time in high_hrv_regions:
            ax1.axvspan(start_time, end_time, alpha=0.3, color='red', label='High HRV (>0.2s)' if start_time == high_hrv_regions[0][0] else "")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.xlabel('Seconds since start')
    plt.ylabel('Interval (s) - Log Scale')
    plt.title('RR Intervals and Heart Rate Variability over Time (Log Scale)')

plt.show()