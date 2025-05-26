# ECG Wave Delineation Visualizer

This project visualizes ECG signals with automatic delineation of P, Q, R, S, and T waves using Python. It downloads ECG records from PhysioNet, processes them, and plots the results with color-coded wave segments.

## Features

- Downloads and caches ECG records from PhysioNet.
- Detects and delineates P, Q, R, S, and T waves using NeuroKit2.
- Visualizes ECG signals with colored segments for each wave type.
- Plots RR intervals and provides summary statistics.

## Requirements

- Python 3.7+
- [wfdb](https://pypi.org/project/wfdb/)
- [neurokit2](https://pypi.org/project/neurokit2/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [numpy](https://pypi.org/project/numpy/)

Install dependencies with:

```bash
pip install wfdb neurokit2 matplotlib numpy
```

## Usage

1. Clone or download this repository.
2. Run the main script:

```bash
python test.py
```

The script will:
- Download the specified ECG record (default: `slp02a` from `slpdb`).
- Process and delineate the ECG signal.
- Display a plot with color-coded P, Q, R, S, and T waves.
- Show RR interval statistics.

## Customization

- To analyze a different record, change the `record_name` and `database` variables in `test.py`.
- Adjust the number of samples or visualization settings as needed.

## References

- [PhysioNet](https://physionet.org/)
- [NeuroKit2 Documentation](https://neuropsychology.github.io/NeuroKit/)
- [WFDB Python](https://wfdb.readthedocs.io/en/latest/)

## License

This project is for educational purposes.
