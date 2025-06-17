import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Read data and calculate sampling rate
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

# Invert ECG
df['ECG'] = -df['ECG']

print(f"Sampling rate: {sampling_rate} Hz")
print(f"Total duration: {df.index[-1]:.2f} seconds")

# Parameters for Fourier analysis
window_duration = 2.0  # seconds
step_size = 2.0  # seconds
window_samples = int(window_duration * sampling_rate)
step_samples = int(step_size * sampling_rate)

# Calculate number of windows
total_samples = len(df)
num_windows = (total_samples - window_samples) // step_samples + 1

# Prepare arrays for Fourier analysis
window_times = []
frequency_data = []
magnitude_data = []

# Frequency array (same for all windows)
freqs = fftfreq(window_samples, 1/sampling_rate)
positive_freqs = freqs[:window_samples//2]

# Compute FFT for each window
for i in range(num_windows):
    start_idx = i * step_samples
    end_idx = start_idx + window_samples
    
    if end_idx > total_samples:
        break
    
    # Get window data
    window_data = df['ECG'].iloc[start_idx:end_idx].values
    window_time = df.index[start_idx + window_samples//2]  # Center time of window
    
    # Apply window function (Hamming) to reduce spectral leakage
    windowed_data = window_data * np.hamming(len(window_data))
    
    # Compute FFT
    fft_data = fft(windowed_data)
    magnitude = np.abs(fft_data[:window_samples//2])
    
    window_times.append(window_time)
    magnitude_data.append(magnitude)

window_times = np.array(window_times)
magnitude_data = np.array(magnitude_data)

# Create plots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Original ECG data
axes[0].plot(df.index, df['ECG'], label='ECG', linewidth=0.5)
axes[0].set_xlabel('Time (seconds)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('ECG Signal')
axes[0].grid(True, alpha=0.3)

# Plot 2: Spectrogram (frequency vs time)
freq_mesh, time_mesh = np.meshgrid(positive_freqs, window_times)
im = axes[1].pcolormesh(time_mesh, freq_mesh, magnitude_data, shading='auto', cmap='viridis')
axes[1].set_xlabel('Time (seconds)')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_title('Spectrogram (2-second windows)')
axes[1].set_ylim(0, 50)  # Focus on lower frequencies relevant to ECG
plt.colorbar(im, ax=axes[1], label='Magnitude')

# Plot 3: Average frequency spectrum
avg_magnitude = np.mean(magnitude_data, axis=0)
axes[2].plot(positive_freqs, avg_magnitude)
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Average Magnitude')
axes[2].set_title('Average Frequency Spectrum')
axes[2].set_xlim(0, 50)  # Focus on lower frequencies
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some statistics
print(f"\nFourier Analysis Results:")
print(f"Number of 2-second windows analyzed: {len(window_times)}")
print(f"Frequency resolution: {positive_freqs[1] - positive_freqs[0]:.2f} Hz")
print(f"Maximum frequency analyzed: {positive_freqs[-1]:.2f} Hz")

# Find dominant frequencies
dominant_freq_idx = np.argmax(avg_magnitude[1:]) + 1  # Skip DC component
dominant_freq = positive_freqs[dominant_freq_idx]
print(f"Dominant frequency (excluding DC): {dominant_freq:.2f} Hz")
