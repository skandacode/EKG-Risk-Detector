import numpy as np
import matplotlib.pyplot as plt
import wfdb
import neurokit2 as nk
import os
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap


record_name = 'slp01a'
database = 'slpdb'
local_data_dir = 'data'
def load_data():
    os.makedirs(local_data_dir, exist_ok=True)
    local_record_path = os.path.join(local_data_dir, record_name)
    if os.path.exists(f"{local_record_path}.dat") and os.path.exists(f"{local_record_path}.hea"):
        print(f"Loading {record_name} from local storage...")
        record = wfdb.rdrecord(local_record_path)
    else:
        print(f"Downloading {record_name} from PhysioNet...")
        record = wfdb.rdrecord(record_name, pn_dir=database)
        wfdb.wrsamp(record_name, fs=record.fs, units=record.units, sig_name=record.sig_name, 
                    p_signal=record.p_signal, fmt=record.fmt, write_dir=local_data_dir)
        print(f"Record saved to {local_data_dir}/")
    ecg_channel_idx = None
    for i, sig_name in enumerate(record.sig_name):
        if 'ECG' in sig_name.upper():
            ecg_channel_idx = i
            break

    if ecg_channel_idx is None:
        print("Available channels:", record.sig_name)
        ecg_channel_idx = 0
        print(f"ECG channel not found, using channel {ecg_channel_idx}: {record.sig_name[ecg_channel_idx]}")


    ecg_signal = record.p_signal[:, ecg_channel_idx]
    sampling_rate = record.fs

    print("Length of signal", len(ecg_signal))
    print(f"Duration of signal: {len(ecg_signal) / sampling_rate:.2f} seconds")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(type(ecg_signal))
    print(type(sampling_rate))
    
    return ecg_signal, sampling_rate

if __name__ == "__main__":
    ecg_signal, sampling_rate = load_data()
    