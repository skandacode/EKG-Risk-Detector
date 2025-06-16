import pandas
import numpy as np
import matplotlib.pyplot as plt

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

window_size = 2000

df['moving_avg'] = df['ECG'].rolling(window=window_size, center=True).mean()
df['standard dev'] = df['ECG'].rolling(window=window_size, center=True).std()

df['dynamic_threshold'] = df['moving_avg'] + 1.5 * df['standard dev']

min_distance_ms = 100
min_distance_samples = int(min_distance_ms * sampling_rate / 1000)

r_peak_indices = []
prev_peak = -min_distance_samples
values = df['ECG'].values
thresholds = df['dynamic_threshold'].values

for i in range(1, len(values) - 1):
    if (not np.isnan(thresholds[i]) and 
        values[i] > thresholds[i] and 
        values[i] > values[i-1] and 
        values[i] > values[i+1]):
        if i - prev_peak >= min_distance_samples:
            r_peak_indices.append(i)
            prev_peak = i

# Add R peaks at the beginning and end of the signal
if r_peak_indices:
    # Add peak at beginning if not already there
    if r_peak_indices[0] > min_distance_samples:
        r_peak_indices.insert(0, 0)
    
    # Add peak at end if not already there
    if len(values) - 1 - r_peak_indices[-1] > min_distance_samples:
        r_peak_indices.append(len(values) - 1)

r_peak_times = df.index[r_peak_indices]

rr_intervals = np.diff(r_peak_times)
rr_interval_times = r_peak_times[1:]


downsample_factor = 10
downsample_indices = np.arange(0, len(df), downsample_factor)

plt.figure()
plt.plot(df.index, df['ECG'], label='ECG')
plt.plot(df.index[downsample_indices], df['moving_avg'].iloc[downsample_indices], 
         label='Moving Average (10s)', color='orange')
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

plt.figure()
plt.plot(rr_interval_times, rr_intervals, label='RR interval')
plt.xlabel('Seconds since start')
plt.ylabel('Interval (s)')
plt.title('RR Intervals over Time')

plt.figure()
plt.plot(rr_interval_times, rr_intervals, label='RR interval')
plt.yscale('log')
plt.xlabel('Seconds since start')
plt.ylabel('Interval (s) - Log Scale')
plt.title('RR Intervals over Time (Log Scale)')

plt.show()