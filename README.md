# ECG Wave Delineation Visualizer

This project visualizes ECG signals with automatic delineation of P, Q, R, S, and T waves using Python. It downloads ECG records from PhysioNet, processes them, and plots the results with color-coded wave segments.

## Features

- Downloads and caches ECG records from PhysioNet.
- Detects and delineates P, Q, R, S, and T waves using NeuroKit2.
- Visualizes ECG signals with colored segments for each wave type.
- Plots RR intervals and provides summary statistics.
- Supports plotting of actual ECG data from CSV files.

## Requirements

- Python 3.7+
- [wfdb](https://pypi.org/project/wfdb/)
- [neurokit2](https://pypi.org/project/neurokit2/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [numpy](https://pypi.org/project/numpy/)
- [pandas](https://pypi.org/project/pandas/)
- [PyWavelets](https://pypi.org/project/PyWavelets/)
- [plotly](https://pypi.org/project/plotly/)

Install dependencies with:

```bash
pip install wfdb neurokit2 matplotlib numpy pandas PyWavelets plotly
```

## Usage

1. Clone or download this repository.
2. To run the main ECG delineation and visualization script:

```bash
python test.py
```

The script will:
- Download the specified ECG record (default: `slp02a` from `slpdb`).
- Process and delineate the ECG signal.
- Display a plot with color-coded P, Q, R, S, and T waves.
- Show RR interval statistics.

### Plotting Actual Data

To plot your own ECG data from a CSV file (with metadata), use:

```bash
python plot_actual_data.py
```

This script expects `data.csv` and `metadata.txt` in the working directory.

## Customization

- To analyze a different record, change the `record_name` and `database` variables in `test.py`.
- Adjust the number of samples or visualization settings as needed.
- For plotting your own data, ensure your CSV and metadata files are formatted similarly to the provided examples.

## References

- [PhysioNet](https://physionet.org/)
- [NeuroKit2 Documentation](https://neuropsychology.github.io/NeuroKit/)
- [WFDB Python](https://wfdb.readthedocs.io/en/latest/)

## License

This project is for educational purposes.
