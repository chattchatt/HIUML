"""
Feature Engineering Script for segment_04.csv
Creates comprehensive features from acoustic data in 150k segments
Optimized for Google Colab environment
"""

import numpy as np
import pandas as pd
from scipy import stats, signal, fft
from scipy.stats import skew, kurtosis
import librosa
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters
import warnings
import os
os.environ["TSFRESH_NO_Numba"] = "1"

warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = r"C:\Users\split_events\event_01.csv"
OUTPUT_FILE = r"C:\Users\feature_created\event_01.csv"
SEGMENT_SIZE = 150000
ROLLING_WINDOW = 1000
N_MFCC = 13
N_FFT_BINS = 10

print("=" * 80)
print("FEATURE ENGINEERING PIPELINE")
print("=" * 80)
print(f"Input file: {INPUT_FILE}")
print(f"Output file: {OUTPUT_FILE}")
print(f"Segment size: {SEGMENT_SIZE:,}")
print(f"Rolling window: {ROLLING_WINDOW:,}")
print(f"MFCC coefficients: {N_MFCC}")
print(f"FFT bins: {N_FFT_BINS}")
print("=" * 80)


def calculate_basic_statistics(data):
    """Calculate basic statistical features"""
    features = {}

    # Basic stats
    features['mean'] = np.mean(data)
    features['median'] = np.median(data)
    features['min'] = np.min(data)
    features['max'] = np.max(data)
    features['range'] = features['max'] - features['min']
    features['std'] = np.std(data)
    features['variance'] = np.var(data)

    # Quantiles
    features['q25'] = np.percentile(data, 25)
    features['q50'] = np.percentile(data, 50)
    features['q75'] = np.percentile(data, 75)
    features['iqr'] = features['q75'] - features['q25']

    # Higher moments
    features['skewness'] = skew(data)
    features['kurtosis'] = kurtosis(data)

    return features


def calculate_rolling_statistics(data, window=ROLLING_WINDOW):
    """Calculate rolling statistics features"""
    features = {}

    # Create pandas series for rolling calculations
    series = pd.Series(data)

    # Rolling mean
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    features['rolling_mean_mean'] = rolling_mean.mean()
    features['rolling_mean_std'] = rolling_mean.std()
    features['rolling_mean_min'] = rolling_mean.min()
    features['rolling_mean_max'] = rolling_mean.max()

    # Rolling min
    rolling_min = series.rolling(window=window, min_periods=1).min()
    features['rolling_min_mean'] = rolling_min.mean()
    features['rolling_min_std'] = rolling_min.std()

    # Rolling max
    rolling_max = series.rolling(window=window, min_periods=1).max()
    features['rolling_max_mean'] = rolling_max.mean()
    features['rolling_max_std'] = rolling_max.std()

    # Rolling quantiles
    rolling_q25 = series.rolling(window=window, min_periods=1).quantile(0.25)
    rolling_q50 = series.rolling(window=window, min_periods=1).quantile(0.50)
    rolling_q75 = series.rolling(window=window, min_periods=1).quantile(0.75)

    features['rolling_q25_mean'] = rolling_q25.mean()
    features['rolling_q50_mean'] = rolling_q50.mean()
    features['rolling_q75_mean'] = rolling_q75.mean()

    # Exponential weighted moving average
    ewma = series.ewm(span=window, min_periods=1).mean()
    features['ewma_mean'] = ewma.mean()
    features['ewma_std'] = ewma.std()
    features['ewma_min'] = ewma.min()
    features['ewma_max'] = ewma.max()

    return features


def calculate_signal_features(data):
    """Calculate signal processing features"""
    features = {}

    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(data)

    # Peak features
    peaks, properties = signal.find_peaks(data, height=0)
    features['num_peaks'] = len(peaks)
    features['peak_mean_height'] = np.mean(properties['peak_heights']) if len(peaks) > 0 else 0
    features['peak_max_height'] = np.max(properties['peak_heights']) if len(peaks) > 0 else 0
    features['peak_std_height'] = np.std(properties['peak_heights']) if len(peaks) > 0 else 0

    # Negative peaks
    neg_peaks, neg_properties = signal.find_peaks(-data, height=0)
    features['num_neg_peaks'] = len(neg_peaks)
    features['neg_peak_mean_height'] = np.mean(neg_properties['peak_heights']) if len(neg_peaks) > 0 else 0
    features['neg_peak_max_height'] = np.max(neg_properties['peak_heights']) if len(neg_peaks) > 0 else 0

    return features


def calculate_spectral_features(data, sr=4000000):
    """Calculate spectral features using librosa and scipy"""
    features = {}

    # Normalize data for librosa (expects float32 between -1 and 1)
    data_normalized = data.astype(np.float32)
    if np.max(np.abs(data_normalized)) > 0:
        data_normalized = data_normalized / np.max(np.abs(data_normalized))

    # Spectral centroid
    spectral_cent = librosa.feature.spectral_centroid(y=data_normalized, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_cent)
    features['spectral_centroid_std'] = np.std(spectral_cent)
    features['spectral_centroid_min'] = np.min(spectral_cent)
    features['spectral_centroid_max'] = np.max(spectral_cent)

    # Spectral rolloff
    spectral_roll = librosa.feature.spectral_rolloff(y=data_normalized, sr=sr)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_roll)
    features['spectral_rolloff_std'] = np.std(spectral_roll)
    features['spectral_rolloff_min'] = np.min(spectral_roll)
    features['spectral_rolloff_max'] = np.max(spectral_roll)

    # MFCC
    mfccs = librosa.feature.mfcc(y=data_normalized, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])

    # FFT features
    fft_values = np.abs(fft.fft(data))
    fft_freq = fft.fftfreq(len(data), 1/sr)

    # Get top N_FFT_BINS frequency magnitudes
    sorted_indices = np.argsort(fft_values)[::-1]
    for i in range(N_FFT_BINS):
        features[f'fft_magnitude_{i}'] = fft_values[sorted_indices[i]]
        features[f'fft_frequency_{i}'] = np.abs(fft_freq[sorted_indices[i]])

    return features


def calculate_tsfresh_features(data):
    """Calculate tsfresh features (lightweight version)"""
    df = pd.DataFrame({
        'id': [0] * len(data),
        'time': range(len(data)),
        'value': data
    })

    # 1) 기본적으로 가벼운 Efficient / Minimal 사용 (둘 중 하나 택1)
    # fc_params = MinimalFCParameters()
    fc_params = EfficientFCParameters()

    # 2) 그래도 Comprehensive 쓰고 싶으면, heavy 피처만 제거
    # fc_params = ComprehensiveFCParameters()
    # for bad_key in ["approximate_entropy", "sample_entropy"]:
    #     fc_params.pop(bad_key, None)

    features_df = extract_features(
        df,
        column_id='id',
        column_sort='time',
        column_value='value',
        default_fc_parameters=fc_params,
        disable_progressbar=True,
        n_jobs=1
    )

    features = features_df.iloc[0].to_dict()
    features = {f'tsfresh_{k}': v for k, v in features.items()}
    return features



def extract_all_features(segment_data):
    """Extract all features from a segment"""
    print("  - Extracting basic statistics...")
    features = calculate_basic_statistics(segment_data)

    print("  - Extracting rolling statistics...")
    features.update(calculate_rolling_statistics(segment_data))

    print("  - Extracting signal features...")
    features.update(calculate_signal_features(segment_data))

    print("  - Extracting spectral features...")
    features.update(calculate_spectral_features(segment_data))

    print("  - Extracting tsfresh features (this may take a while)...")
    features.update(calculate_tsfresh_features(segment_data))

    return features


def process_data():
    """Main processing function"""
    print("\n[1/5] Reading CSV file...")
    df = pd.read_csv(INPUT_FILE)
    print(f"  Total rows: {len(df):,}")

    acoustic_data = df['acoustic_data'].values
    time_to_failure = df['time_to_failure'].values
    total_rows = len(acoustic_data)

    # Calculate number of segments
    num_segments = int(np.ceil(total_rows / SEGMENT_SIZE))
    remainder = total_rows % SEGMENT_SIZE

    print(f"\n[2/5] Segmentation info:")
    print(f"  Total data points: {total_rows:,}")
    print(f"  Segment size: {SEGMENT_SIZE:,}")
    print(f"  Number of segments: {num_segments}")
    print(f"  Remainder: {remainder:,}")

    # Split data into segments (from back to front)
    print(f"\n[3/5] Creating segments...")
    segments = []

    if remainder == 0:
        # Perfect division
        for i in range(num_segments):
            start_idx = i * SEGMENT_SIZE
            end_idx = start_idx + SEGMENT_SIZE
            segments.append(acoustic_data[start_idx:end_idx])
    else:
        # Need padding for first segment
        # Take complete segments from the back
        complete_segments_start = remainder

        # First segment (with padding)
        first_segment = acoustic_data[:complete_segments_start]
        first_segment_mean = np.mean(first_segment)
        padding_needed = SEGMENT_SIZE - remainder
        padded_segment = np.concatenate([
            np.full(padding_needed, first_segment_mean),
            first_segment
        ])
        segments.append(padded_segment)
        print(f"  First segment padded with {padding_needed:,} values (mean: {first_segment_mean:.2f})")

        # Remaining complete segments
        for i in range(num_segments - 1):
            start_idx = complete_segments_start + i * SEGMENT_SIZE
            end_idx = start_idx + SEGMENT_SIZE
            segments.append(acoustic_data[start_idx:end_idx])

    print(f"  Created {len(segments)} segments")

    # Extract features for each segment
    print(f"\n[4/5] Extracting features from each segment...")
    all_segment_features = []

    for seg_idx, segment in enumerate(segments):
        print(f"\nProcessing segment {seg_idx + 1}/{len(segments)}...")
        segment_features = extract_all_features(segment)
        all_segment_features.append(segment_features)

    # Create output DataFrame with original data + features
    print(f"\n[5/5] Creating output DataFrame...")

    # Assign segment features to each row
    result_df = df.copy()

    # Add segment index column
    if remainder == 0:
        segment_indices = np.repeat(range(num_segments), SEGMENT_SIZE)
    else:
        segment_indices = np.concatenate([
            np.full(remainder, 0),
            np.repeat(range(1, num_segments), SEGMENT_SIZE)
        ])

    result_df['segment_id'] = segment_indices

    # Add all features to each row based on segment
    print("  Adding features to rows...")
    feature_names = list(all_segment_features[0].keys())

    for feature_name in feature_names:
        feature_values = []
        for seg_idx in segment_indices:
            feature_values.append(all_segment_features[seg_idx][feature_name])
        result_df[feature_name] = feature_values

    # Save to CSV
    print(f"\n[6/6] Saving to {OUTPUT_FILE}...")
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE!")
    print("=" * 80)
    print(f"Original columns: {len(df.columns)}")
    print(f"Total features added: {len(feature_names)}")
    print(f"Final columns: {len(result_df.columns)}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"File size: {len(result_df):,} rows x {len(result_df.columns)} columns")
    print("=" * 80)


if __name__ == "__main__":
    process_data()
