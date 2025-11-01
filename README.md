# EKG Risk Detector

A cardiac event prediction system that analyzes EKG (electrocardiogram) data to detect potential cardiac events hours in advance using Heart Rate Variability (HRV) analysis.

## Overview

This project implements an algorithm that processes EKG signals to identify patterns associated with cardiac risk. By analyzing R-peak detection and Heart Rate Variability (HRV), the system can predict potential cardiac events before they occur, providing early warning indicators that could be crucial for patient monitoring and intervention.

## Key Features

- **R-Peak Detection**: Utilizes both NeuroKit2 library and custom algorithms to identify R-peaks in EKG signals
- **Heart Rate Variability (HRV) Analysis**: Calculates and monitors HRV to detect anomalies
- **Dynamic Threshold Detection**: Adapts to signal characteristics using moving median and standard deviation
- **Multi-Window Analysis**: Uses rolling windows for robust statistical analysis
- **Risk Prediction**: Identifies high and low HRV regions that indicate potential cardiac events
- **Visualization**: Comprehensive plotting of EKG signals, R-peaks, RR intervals, and HRV trends

## How It Works

### 1. Signal Processing
The algorithm processes EKG data through several stages:
- Loads EKG signal data (supports both custom CSV data and WFDB database formats)
- Applies moving median filtering to reduce noise
- Calculates dynamic thresholds based on local signal characteristics

### 2. R-Peak Detection
R-peaks are detected using:
- **Primary Method**: NeuroKit2's ECG peak detection algorithm
- **Fallback Method**: Custom detection using dynamic thresholds and minimum distance constraints

![R-Peak Detection](images/r_peaks.png)
*R-peaks detected in EKG signal with dynamic threshold overlay*

### 3. RR Interval Analysis
The time intervals between consecutive R-peaks (RR intervals) are calculated and analyzed:
- Detects abnormal heart rate patterns
- Tracks temporal variations in heart rhythm

### 4. Heart Rate Variability (HRV) Computation
HRV is computed as the standard deviation of RR intervals over a rolling window:
- **Low HRV** (< 7ms): Indicates reduced cardiac adaptability, associated with increased risk
- **High HRV** (> 100ms): May indicate arrhythmias or other irregularities
- Uses median filtering to smooth HRV trends and reduce noise

### 5. Risk Prediction
The algorithm identifies risk periods by monitoring:
- Sustained low HRV regions (blue shaded areas in plots)
- Abnormally high HRV regions (red shaded areas in plots)
- Patterns that precede cardiac events by hours

## Results

### Healthy Data Analysis
![Healthy Data Plot](images/healthy_data_plot.png)
*HRV analysis of healthy EKG data showing normal variability patterns*

### Unhealthy Data Analysis
![Unhealthy Data Plot](images/unhealthy_data_plot.png)
*HRV analysis of unhealthy EKG data showing abnormal patterns and risk regions*

The plots clearly show the difference in HRV patterns between healthy and at-risk individuals, with the unhealthy data showing characteristic periods of low HRV that precede cardiac events.

## Installation

### Prerequisites
- Python 3.7 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/skandacode/EKG-Risk-Detector.git
cd EKG-Risk-Detector
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages
- `matplotlib` - Visualization
- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `scipy` - Signal processing
- `wfdb` - Reading medical waveform data
- `neurokit2` - Biosignal processing
- `PyWavelets` - Wavelet transforms

## Usage

### Basic Usage

Run the main analysis script:
```bash
python process_data.py
```

### Configuration

The algorithm can be configured by modifying constants at the top of `process_data.py`:

```python
# Visualization options
PLOT_ECG_WITH_PEAKS = True      # Show EKG signal with detected R-peaks
PLOT_RR_LINEAR = False          # Show RR intervals on linear scale
PLOT_RR_LOG = True              # Show RR intervals on log scale (recommended)

# Algorithm options
USE_NEUROKIT_RPEAK_DETECTION = True  # Use NeuroKit2 for R-peak detection (Otherwise uses custom method)
USE_UNHEALTHY_DATA = True            # Toggle between healthy/unhealthy datasets

# Detection parameters
WINDOW_SIZE = 5                      # Moving median window (seconds)
MIN_DISTANCE_MS = 100                # Minimum distance between R-peaks (ms)
HRV_WINDOW_SIZE = 20                 # HRV calculation window size
HRV_HIGH_THRESHOLD = 0.1             # High HRV threshold (seconds)
HRV_LOW_THRESHOLD = 7e-3             # Low HRV threshold (seconds)
DYNAMIC_THRESHOLD_FACTOR = 2.5       # Multiplier for dynamic threshold
```

### Data Sources

The system supports two data input methods:

1. **Custom CSV Data** (`data.csv`):
   - Format: CSV with 'TIME' and 'ECG' columns
   - Used when `USE_UNHEALTHY_DATA = True`

2. **WFDB Database** (via `test_with_db.py`):
   - Downloads from PhysioNet SLPDB database
   - Used when `USE_UNHEALTHY_DATA = False`
   - Provides access to validated medical datasets

## Algorithm Parameters

### R-Peak Detection
- **Window Size**: 5 seconds for moving median calculation
- **Minimum Distance**: 100ms between peaks (prevents false detections)
- **Dynamic Threshold**: Mean + 2.5× standard deviation

### HRV Analysis
- **Primary Window**: 20 beats for HRV calculation
- **Smoothing Window**: 100 beats for median filtering
- **Low HRV Threshold**: 0.007s (7ms)
- **High HRV Threshold**: 0.1s (100ms)

### Heart Rate Ranges
- **Normal Range**: 60-100 bpm
- **Corresponding RR Intervals**: 0.6-1.0 seconds

## Clinical Significance

### Low HRV Regions
Extended periods of low HRV are associated with:
- Reduced parasympathetic activity
- Increased cardiac event risk
- Potential precursor to arrhythmias

### High HRV Regions
Abnormally high HRV may indicate:
- Irregular heart rhythms
- Potential arrhythmias

## Project Structure

```
EKG-Risk-Detector/
├── process_data.py          # Main analysis algorithm
├── test_with_db.py          # WFDB database interface
├── fourier_analysis.py      # Frequency domain analysis - Not being used
├── data.csv                 # Sample unhealthy EKG data
├── requirements.txt         # Python dependencies
├── plot_actual_data.ipynb   # Interactive data visualization
├── fouriertest.ipynb        # Fourier analysis experiments
├── data/                    # Downloaded WFDB records
│   └── slp01a.hea          # Header file for sleep database
└── images/                  # Plot images for documentation
    ├── r_peaks.png
    ├── healthy_data_plot.png
    └── unhealthy_data_plot.png
```


## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- NeuroKit2 library for biosignal processing
- PhysioNet for providing open medical datasets
- WFDB library for medical waveform data access
- AI assistance (GitHub Copilot) was used to help develop the code and create this documentation

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---
