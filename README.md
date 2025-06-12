# ECG Wave Delineation Visualizer

This project visualizes ECG signals with automatic delineation of P, Q, R, S, and T waves using Python. It can process ECG records from PhysioNet or plot your own ECG data from a CSV file.

## Features
- Plots RR intervals and provides summary statistics.
- **Plots actual ECG data from CSV files with R peak detection and RR interval analysis.**

## Requirements

- Python 3.7+
- [matplotlib](https://pypi.org/project/matplotlib/)
- [numpy](https://pypi.org/project/numpy/)
- [pandas](https://pypi.org/project/pandas/)

Install dependencies with:

```bash
pip install matplotlib numpy pandas
```

## Usage

1. Clone or download this repository.

### To plot your own ECG data from a CSV file:

```bash
python plot_actual_data.py
```

- Expects a `data.csv` file in the working directory.
- Ignores timestamps in the CSV and generates a time axis based on the sampling rate.
- Inverts the ECG signal for visualization.
- Detects R peaks using a threshold and minimum distance.
- Plots the ECG signal with detected R peaks and the RR intervals over time.
- Prints the sampling rate and the processed DataFrame.

**CSV Format:**  
Your `data.csv` should have at least the columns `TIME` (timestamp) and `ECG` (signal value).

Example:
```csv
TIME,ECG
18:00:05.699,661.022705
18:00:05.704,735.636780
18:00:05.709,752.442139
18:00:05.714,592.245361
18:00:05.719,637.751221
18:00:05.724,714.962036
18:00:05.729,579.380798
...
```

## Customization

- To analyze a different record, change the `record_name` and `database` variables in `test.py`.
- Adjust the R peak detection threshold or minimum distance in `plot_actual_data.py` as needed.
- For plotting your own data, ensure your CSV file is formatted with `TIME` and `ECG` columns.

## License

This project is for educational purposes.