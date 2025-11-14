"""
Standardization Script for Acoustic Data
Applies StandardScaler to acoustic_data and saves to new CSV
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse
import os


def standardize_full_data(input_file, output_file):
    """
    전체 데이터를 한 번에 표준화

    Parameters:
    -----------
    input_file : str
        입력 CSV 파일 경로
    output_file : str
        출력 CSV 파일 경로
    """
    print("=" * 80)
    print("FULL DATA STANDARDIZATION")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")

    # CSV 읽기
    print("\n[1/3] Loading CSV file...")
    df = pd.read_csv(input_file)
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")

    # 원본 통계
    print("\n[2/3] Original acoustic_data statistics:")
    print(f"  Mean: {df['acoustic_data'].mean():.2f}")
    print(f"  Std: {df['acoustic_data'].std():.2f}")
    print(f"  Min: {df['acoustic_data'].min():.2f}")
    print(f"  Max: {df['acoustic_data'].max():.2f}")

    # 표준화
    print("\n[3/3] Applying StandardScaler...")
    scaler = StandardScaler()
    df['acoustic_data'] = scaler.fit_transform(
        df['acoustic_data'].values.reshape(-1, 1)
    ).flatten()

    # 표준화 후 통계
    print("\nStandardized acoustic_data statistics:")
    print(f"  Mean: {df['acoustic_data'].mean():.6f}")
    print(f"  Std: {df['acoustic_data'].std():.6f}")
    print(f"  Min: {df['acoustic_data'].min():.2f}")
    print(f"  Max: {df['acoustic_data'].max():.2f}")

    # 저장
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)

    print("\n" + "=" * 80)
    print("STANDARDIZATION COMPLETE!")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"File size: {len(df):,} rows x {len(df.columns)} columns")
    print("=" * 80)

    return df, scaler


def standardize_by_segments(input_file, output_file, segment_size=150000):
    """
    세그먼트별로 표준화 (각 세그먼트마다 별도의 scaler 적용)

    Parameters:
    -----------
    input_file : str
        입력 CSV 파일 경로
    output_file : str
        출력 CSV 파일 경로
    segment_size : int
        세그먼트 크기 (기본값: 150,000)
    """
    print("=" * 80)
    print("SEGMENT-WISE STANDARDIZATION")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Segment size: {segment_size:,}")

    # CSV 읽기
    print("\n[1/4] Loading CSV file...")
    df = pd.read_csv(input_file)
    total_rows = len(df)
    print(f"  Total rows: {total_rows:,}")

    # 세그먼트 개수 계산
    num_segments = int(np.ceil(total_rows / segment_size))
    remainder = total_rows % segment_size

    print(f"\n[2/4] Segmentation info:")
    print(f"  Number of segments: {num_segments}")
    print(f"  Remainder: {remainder:,}")

    # 결과 저장용 DataFrame
    result_df = df.copy()

    # 세그먼트 인덱스 생성
    if remainder == 0:
        segment_indices = np.repeat(range(num_segments), segment_size)
    else:
        segment_indices = np.concatenate([
            np.full(remainder, 0),
            np.repeat(range(1, num_segments), segment_size)
        ])

    result_df['segment_id'] = segment_indices

    # 세그먼트별 표준화
    print("\n[3/4] Standardizing each segment...")
    scalers = []

    for seg_id in range(num_segments):
        # 세그먼트 마스크
        mask = result_df['segment_id'] == seg_id
        segment_data = result_df.loc[mask, 'acoustic_data'].values

        # 표준화
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(
            segment_data.reshape(-1, 1)
        ).flatten()

        # 결과 저장
        result_df.loc[mask, 'acoustic_data'] = standardized_data
        scalers.append(scaler)

        print(f"  Segment {seg_id}: {len(segment_data):,} rows "
              f"(mean: {standardized_data.mean():.6f}, std: {standardized_data.std():.6f})")

    # 저장
    print(f"\n[4/4] Saving to {output_file}...")
    result_df.to_csv(output_file, index=False)

    print("\n" + "=" * 80)
    print("SEGMENT-WISE STANDARDIZATION COMPLETE!")
    print("=" * 80)
    print(f"Output file: {output_file}")
    print(f"File size: {len(result_df):,} rows x {len(result_df.columns)} columns")
    print(f"Added column: segment_id")
    print("=" * 80)

    return result_df, scalers


def main():
    parser = argparse.ArgumentParser(description='Standardize acoustic data and save to CSV')
    parser.add_argument('--input', '-i', type=str,
                        default=r"C:\Users\split_events\event_01.csv",
                        help='Input CSV file path')
    parser.add_argument('--output', '-o', type=str,
                        default=r"C:\Users\split_events\stand_event_01.csv",
                        help='Output CSV file path')
    parser.add_argument('--mode', '-m', type=str, choices=['full', 'segment'],
                        default='full',
                        help='Standardization mode: "full" (전체 한번에) or "segment" (세그먼트별)')
    parser.add_argument('--segment-size', '-s', type=int, default=150000,
                        help='Segment size for segment-wise standardization')

    args = parser.parse_args()

    # 입력 파일 존재 확인
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # 표준화 실행
    if args.mode == 'full':
        standardize_full_data(args.input, args.output)
    else:
        standardize_by_segments(args.input, args.output, args.segment_size)


if __name__ == '__main__':
    main()
