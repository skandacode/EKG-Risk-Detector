import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from typing import Optional, Callable

class StreamingECGAnalyzer:
    def __init__(self, 
                 sampling_rate: float = 200.0,
                 buffer_size: int = 10000,
                 min_distance_ms: int = 100,
                 hrv_window_size: int = 13,
                 window_size: int = 2000,
                 hrv_threshold: float = 0.9,
                 hrv_low_threshold: float = 3e-3,
                 hrv_low_count_threshold: int = 5):
        
        # Configuration
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        self.min_distance_samples = int(min_distance_ms * sampling_rate / 1000)
        self.hrv_window_size = hrv_window_size
        self.window_size = window_size
        self.hrv_threshold = hrv_threshold
        self.hrv_low_threshold = hrv_low_threshold
        self.hrv_low_count_threshold = hrv_low_count_threshold
        
        # Use NumPy arrays with circular buffer indices for better performance
        self.ecg_buffer = np.zeros(buffer_size)
        self.time_buffer = np.zeros(buffer_size)
        self.buffer_index = 0
        self.buffer_full = False
        
        # Pre-allocate arrays for rolling statistics
        self.window_data = np.zeros(window_size)
        self.window_index = 0
        self.window_full = False
        
        # R-peak detection - keep deques for variable-length data
        self.r_peaks = deque(maxlen=1000)
        self.r_peak_times = deque(maxlen=1000)
        self.last_r_peak_sample = -self.min_distance_samples
        
        # HRV analysis - use NumPy for better performance
        self.rr_intervals = np.zeros(999)
        self.rr_times = np.zeros(999)
        self.hrv_buffer = np.zeros(hrv_window_size)
        self.rr_count = 0
        self.hrv_index = 0
        
        # Cache frequently used values
        self.current_moving_median = np.nan
        self.current_std = np.nan
        self.current_threshold = np.nan
        
        # Statistics
        self.total_samples = 0
        self.start_time = None
        
        # Callbacks
        self.r_peak_callback = None
        self.hrv_alert_callback = None
        self.streaming = False

    def add_sample(self, ecg_value: float, timestamp: float) -> None:
        """Add a single ECG sample to the stream - optimized version."""
        # Invert and store ECG value
        inverted_ecg = -ecg_value
        
        # Update circular buffer
        self.ecg_buffer[self.buffer_index] = inverted_ecg
        self.time_buffer[self.buffer_index] = timestamp
        
        # Update rolling statistics efficiently
        self._update_rolling_stats_fast(inverted_ecg)
        
        # Detect R-peaks
        detected = self._detect_r_peaks_fast()
        
        # Update indices
        self.buffer_index = (self.buffer_index + 1) % self.buffer_size
        if self.buffer_index == 0:
            self.buffer_full = True
        self.total_samples += 1
        
        if detected:
            self._check_alerts()

    def _update_rolling_stats_fast(self, new_value: float) -> None:
        """Efficiently update rolling statistics using median instead of mean."""
        if not self.window_full and self.window_index < self.window_size:
            # Building up the window
            self.window_data[self.window_index] = new_value
            self.window_index += 1
            
            if self.window_index == self.window_size:
                self.window_full = True
                
        elif self.window_full:
            # Replace old value with new value
            self.window_data[self.window_index % self.window_size] = new_value
            self.window_index += 1
        
        # Calculate current stats
        if self.window_full or self.window_index >= self.window_size:
            n = min(self.window_index, self.window_size)
            current_window = self.window_data[:n] if not self.window_full else self.window_data
            self.current_moving_median = np.median(current_window)
            self.current_std = np.std(current_window)
            self.current_threshold = self.current_moving_median + 1.5 * self.current_std
        else:
            self.current_moving_median = np.nan
            self.current_std = np.nan
            self.current_threshold = np.nan

    def _detect_r_peaks_fast(self) -> bool:
        """Optimized R-peak detection."""
        if self.total_samples < 2:
            return False
        
        # Get current and neighboring indices
        curr_idx = (self.buffer_index - 1) % self.buffer_size
        prev_idx = (self.buffer_index - 2) % self.buffer_size
        next_idx = self.buffer_index % self.buffer_size
        
        # Skip if we don't have enough data
        if not self.window_full or np.isnan(self.current_threshold):
            return False
        
        # Get values
        curr_val = self.ecg_buffer[curr_idx]
        prev_val = self.ecg_buffer[prev_idx]
        next_val = self.ecg_buffer[next_idx] if self.total_samples > 2 else curr_val
        
        # Check peak conditions
        is_peak = (curr_val > self.current_threshold and 
                  curr_val > prev_val and 
                  curr_val > next_val)
        
        # Check minimum distance
        if is_peak and (self.total_samples - self.last_r_peak_sample) >= self.min_distance_samples:
            self.r_peaks.append(curr_val)
            self.r_peak_times.append(self.time_buffer[curr_idx])
            self.last_r_peak_sample = self.total_samples
            
            # Trigger callback
            if self.r_peak_callback:
                self.r_peak_callback(self.time_buffer[curr_idx], curr_val)
            
            # Update HRV
            if len(self.r_peak_times) >= 2:
                self._update_hrv_fast()
                return True
        
        return False

    def _update_hrv_fast(self) -> None:
        """Optimized HRV calculation using NumPy arrays."""
        if len(self.r_peak_times) < 2:
            return
        
        # Calculate latest RR interval
        latest_rr = self.r_peak_times[-1] - self.r_peak_times[-2]
        
        # Store in circular buffer
        rr_idx = self.rr_count % len(self.rr_intervals)
        self.rr_intervals[rr_idx] = latest_rr
        self.rr_times[rr_idx] = self.r_peak_times[-1]
        self.rr_count += 1
        
        # Calculate HRV using efficient sliding window
        if self.rr_count >= self.hrv_window_size:
            # Get the most recent RR intervals for HRV calculation
            start_idx = max(0, self.rr_count - self.hrv_window_size)
            end_idx = self.rr_count
            
            if end_idx - start_idx == self.hrv_window_size:
                # Use circular buffer indices
                if self.rr_count <= len(self.rr_intervals):
                    # Haven't wrapped around yet
                    window_rr = self.rr_intervals[start_idx:end_idx]
                else:
                    # Need to handle wrap-around
                    n_recent = len(self.rr_intervals)
                    recent_start = (self.rr_count - self.hrv_window_size) % n_recent
                    recent_end = self.rr_count % n_recent
                    
                    if recent_start < recent_end:
                        window_rr = self.rr_intervals[recent_start:recent_end]
                    else:
                        window_rr = np.concatenate([
                            self.rr_intervals[recent_start:],
                            self.rr_intervals[:recent_end]
                        ])
                
                # Calculate HRV efficiently
                hrv = np.std(window_rr)
                
                # Store current HRV (could use a circular buffer here too for memory efficiency)
                self.current_hrv = hrv
                self.current_hrv_time = self.r_peak_times[-1]

    def _check_alerts(self) -> None:
        """Optimized alert checking."""
        if not hasattr(self, 'current_hrv') or not self.hrv_alert_callback:
            return
        
        # Check thresholds
        if self.current_hrv > self.hrv_threshold:
            self.hrv_alert_callback("HIGH_HRV", self.current_hrv_time, self.current_hrv)
        elif self.current_hrv < self.hrv_low_threshold:
            self.hrv_alert_callback("LOW_HRV", self.current_hrv_time, self.current_hrv)

    def get_current_stats(self) -> dict:
        """Get current streaming statistics - optimized."""
        stats = {
            'total_samples': self.total_samples,
            'buffer_usage': f"{min(self.total_samples, self.buffer_size)}/{self.buffer_size}",
            'r_peaks_detected': len(self.r_peaks),
            'current_hr': None,
            'current_hrv': None,
            'current_threshold': self.current_threshold,
            'streaming': getattr(self, 'streaming', False)
        }
        
        # Calculate HR from most recent RR interval
        if self.rr_count > 0:
            latest_rr_idx = (self.rr_count - 1) % len(self.rr_intervals)
            latest_rr = self.rr_intervals[latest_rr_idx]
            stats['current_hr'] = 60.0 / latest_rr if latest_rr > 0 else None
        
        if hasattr(self, 'current_hrv'):
            stats['current_hrv'] = self.current_hrv
        
        return stats

    def get_recent_data(self, n_samples: int = 1000) -> dict:
        """Get recent data for plotting - optimized."""
        if self.total_samples == 0:
            return {'ecg': [], 'time': [], 'threshold': []}
        
        # Determine how many samples we actually have
        available_samples = min(n_samples, self.total_samples, self.buffer_size)
        
        if not self.buffer_full:
            # Haven't filled buffer yet
            ecg_data = self.ecg_buffer[:self.buffer_index].copy()
            time_data = self.time_buffer[:self.buffer_index].copy()
        else:
            # Buffer is full, need to handle circular indexing
            if available_samples >= self.buffer_size:
                # Want all data
                start_idx = self.buffer_index
                indices = np.arange(start_idx, start_idx + self.buffer_size) % self.buffer_size
            else:
                # Want recent N samples
                start_idx = (self.buffer_index - available_samples) % self.buffer_size
                indices = np.arange(start_idx, start_idx + available_samples) % self.buffer_size
            
            ecg_data = self.ecg_buffer[indices]
            time_data = self.time_buffer[indices]
        
        return {
            'ecg': ecg_data,
            'time': time_data,
            'threshold': np.full(len(ecg_data), self.current_threshold),
            'moving_median': np.full(len(ecg_data), self.current_moving_median)
        }

    def set_callbacks(self, r_peak_callback: Optional[Callable] = None, 
                     hrv_alert_callback: Optional[Callable] = None) -> None:
        """Set callback functions for events."""
        self.r_peak_callback = r_peak_callback
        self.hrv_alert_callback = hrv_alert_callback


def example_r_peak_callback(timestamp: float, amplitude: float) -> None:
    """Example callback for R-peak detection."""
    return
    print(f"R-peak detected at {timestamp:.2f}s, amplitude: {amplitude:.3f}")


def example_hrv_alert_callback(alert_type: str, timestamp: float, hrv_value: float) -> None:
    """Example callback for HRV alerts."""
    print(f"HRV Alert [{alert_type}] at {timestamp:.2f}s: HRV = {hrv_value:.6f}")


if __name__ == "__main__":
    # Create analyzer
    analyzer = StreamingECGAnalyzer()
    
    # Set up callbacks
    analyzer.set_callbacks(
        r_peak_callback=example_r_peak_callback,
        hrv_alert_callback=example_hrv_alert_callback
    )
    #read data from a CSV file
    df = pd.read_csv('data.csv')
    for index, row in df.iterrows():
        ecg_value = row[' "ECG"']
        analyzer.add_sample(ecg_value, index / analyzer.sampling_rate)