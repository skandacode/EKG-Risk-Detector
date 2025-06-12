import pandas
import numpy as np
import matplotlib.pyplot as plt

# Read metadata to get the date
with open('metadata.txt', 'r') as meta_file:
    for line in meta_file:
        if line.startswith('"Date recorded"'):
            date_str = line.split(',')[1].strip().strip('"')
            break

df = pandas.read_csv(
    'data.csv',
    quotechar='"',
    skipinitialspace=True
)

df['TIME'] = pandas.to_datetime(
    date_str + ' ' + df['TIME'],
    format='%Y-%m-%d %H:%M:%S.%f'
)
df.set_index('TIME', inplace=True)

print(df.head(100))

df=df.head(10000)


#print sampling rate
first_time_diff = (df.index[1] - df.index[0]).total_seconds()
sampling_rate = 1 / first_time_diff
print(f"sampling rate: {sampling_rate} hz")

first_chunk = df

# Invert ECG
first_chunk['ECG'] = -first_chunk['ECG']

# Plot ECG data with 0 line
plt.figure(figsize=(15, 6))
plt.plot(first_chunk.index, first_chunk['ECG'], 'b-', linewidth=0.8, label='ECG')
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='0 Voltage')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.title('ECG Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


derivative = np.diff(first_chunk['ECG']) / np.diff(first_chunk.index.astype(np.int64)) * 1e9

derivative_time = first_chunk.index[:-1]

plt.figure(figsize=(15, 6))
plt.plot(derivative_time, derivative, 'g-', linewidth=0.8, label='First Derivative')
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='0 Line')
plt.xlabel('Time')
plt.ylabel('Rate of Change (V/s)')
plt.title('First Derivative of ECG Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


second_derivative = np.diff(derivative) / np.diff(derivative_time.astype(np.int64)) * 1e9

second_derivative_time = derivative_time[:-1]

plt.figure(figsize=(15, 6))
plt.plot(second_derivative_time, second_derivative, 'r-', linewidth=0.8, label='Second Derivative')
plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='0 Line')
plt.xlabel('Time')
plt.ylabel('Rate of Change (V/s²)')
plt.title('Second Derivative of ECG Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()