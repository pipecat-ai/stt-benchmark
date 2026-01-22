#!/usr/bin/env python3
"""Create a HuggingFace dataset from local PCM audio files and ground truth metadata.

Packages the .pcm audio files in stt_benchmark_data/audio/ together with the
transcription metadata from ground_truth.jsonl into a single HuggingFace dataset.

Usage:
    uv run python scripts/create_hf_dataset.py
    uv run python scripts/create_hf_dataset.py --output my_dataset
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from datasets import Audio, Dataset, Features, Value


def main():
    parser = argparse.ArgumentParser(
        description="Create HuggingFace dataset from PCM audio files and ground truth JSONL"
    )
    parser.add_argument(
        "--ground-truth",
        default="ground_truth.jsonl",
        help="Path to ground_truth.jsonl (default: ground_truth.jsonl)",
    )
    parser.add_argument(
        "--audio-dir",
        default="stt_benchmark_data/audio",
        help="Directory containing .pcm audio files (default: stt_benchmark_data/audio)",
    )
    parser.add_argument(
        "--output",
        default="stt_benchmark_dataset",
        help="Output directory for the dataset (default: stt_benchmark_dataset)",
    )
    args = parser.parse_args()

    ground_truth_path = Path(args.ground_truth)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output)

    if not ground_truth_path.exists():
        print(f"Error: {ground_truth_path} not found")
        sys.exit(1)

    if not audio_dir.exists():
        print(f"Error: {audio_dir} not found")
        sys.exit(1)

    # Parse ground truth JSONL
    header = None
    samples = []
    with open(ground_truth_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "header":
                header = record
            elif record.get("type") == "sample":
                samples.append(record)

    print(f"Found {len(samples)} samples in ground truth")
    if header:
        print(f"  Model: {header.get('model')}")
        print(f"  Run ID: {header.get('run_id')}")

    # Build dataset rows
    data = {
        "sample_id": [],
        "audio": [],
        "duration_seconds": [],
        "transcription": [],
    }

    skipped = 0
    for sample in samples:
        pcm_path = audio_dir / f"{sample['sample_id']}.pcm"
        if not pcm_path.exists():
            print(f"  Warning: {pcm_path.name} not found, skipping")
            skipped += 1
            continue

        # Read 16-bit signed PCM and convert to float32 [-1, 1]
        pcm_bytes = pcm_path.read_bytes()
        audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        data["sample_id"].append(sample["sample_id"])
        data["audio"].append({"array": audio_array, "sampling_rate": 16000})
        data["duration_seconds"].append(sample["duration_seconds"])
        data["transcription"].append(sample["transcription"])

    if skipped:
        print(f"  Skipped {skipped} samples (missing audio files)")

    # Define features
    features = Features(
        {
            "sample_id": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "duration_seconds": Value("float64"),
            "transcription": Value("string"),
        }
    )

    # Create dataset
    dataset = Dataset.from_dict(data, features=features)
    print(f"\nCreated dataset with {len(dataset)} samples")

    # Save to disk
    dataset.save_to_disk(str(output_dir))
    print(f"Saved to {output_dir}/")


if __name__ == "__main__":
    main()
