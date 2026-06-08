#!/usr/bin/env python3
"""
Summarize eating events and mean rewards across sweeps (CSV only).

Parses new run names like:
  20250920_115437_1_bao_vd_-0.006_lmp_-0.02_ltp_-0.01_fdr_10_wb_5_run_2

Saves:
  - runs_raw_metrics.csv (per run)
  - condition_summary_flat.csv (aggregated means/stds)
"""

from pathlib import Path
import re
import argparse
import pandas as pd
from datetime import datetime

# Regex for new sweep configs
RUN_REGEX = re.compile(
    r"bao_vd_(?P<vd>[-+]?\d*\.?\d+)_"
    r"lmp_(?P<lmp>[-+]?\d*\.?\d+)_"
    r"ltp_(?P<ltp>[-+]?\d*\.?\d+)_"
    r"fdr_(?P<fdr>\d+)_"
    r"wb_(?P<wb>\d+)_"
    r"run_(?P<run>\d+)"
)

# Regex to extract timestamp from folder name
TIMESTAMP_REGEX = re.compile(r"^(\d{8}_\d{6})")

def parse_timestamp(folder_name: str) -> datetime:
    """Extract timestamp from folder name and convert to datetime object."""
    match = TIMESTAMP_REGEX.match(folder_name)
    if not match:
        return None
    timestamp_str = match.group(1)
    try:
        return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        return None

def parse_metrics_file(p: Path) -> dict:
    vals = {}
    if not p.exists():
        return vals

    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Convert key to snake_case and lowercase
                key = key.lower().replace(" ", "_").replace("(", "").replace(")", "")

                # Handle special cases for mode_values which contains a list
                if "mode_values" in key:
                    try:
                        # Parse the list string
                        vals[key] = eval(value)
                    except:
                        vals[key] = value
                else:
                    # Try to convert to float, fallback to string
                    try:
                        vals[key] = float(value)
                    except ValueError:
                        vals[key] = value
    
    return vals

def extract_run_info(folder_name: str):
    m = RUN_REGEX.search(folder_name)
    if not m:
        return None
    return (
        float(m.group("vd")),
        float(m.group("lmp")),
        float(m.group("ltp")),
        int(m.group("fdr")),
        int(m.group("wb")),
        int(m.group("run")),
    )

def collect(base_dir: Path, date_prefix: str = None, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
    rows = []
    filtered_count = 0
    total_count = 0

    for exp_dir in base_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        total_count += 1

        # If date_prefix is specified, use that for filtering
        if date_prefix:
            if not exp_dir.name.startswith(date_prefix):
                filtered_count += 1
                continue
        else:
            # Otherwise use timestamp filtering
            folder_timestamp = parse_timestamp(exp_dir.name)
            if folder_timestamp is None:
                print(f"Warning: Could not parse timestamp from folder: {exp_dir.name}")
                continue

            # Apply timestamp filters
            if start_time and folder_timestamp < start_time:
                filtered_count += 1
                continue
            if end_time and folder_timestamp > end_time:
                filtered_count += 1
                continue

        info = extract_run_info(exp_dir.name)
        if not info:
            continue
        vd, lmp, ltp, fdr, wb, run = info
        metrics_path = exp_dir / "outputs" / "figures" / "performance_metrics.txt"
        if not metrics_path.exists():
            print("Skipping (no metrics):", exp_dir)
            continue
        print("Loading:", metrics_path)
        metrics = parse_metrics_file(metrics_path)
        if not metrics or "total_eating_events" not in metrics or metrics["total_eating_events"] is None:
            continue
        rows.append({
            "exp_dir": exp_dir.name,
            "vd": vd,
            "large_move_penalty": lmp,
            "large_turn_penalty": ltp,
            "food_detection_range": fdr,
            "walkerbots": wb,
            "run": run,
            **metrics,
        })

    if filtered_count > 0:
        print(f"Filtered out {filtered_count} runs based on filtering criteria.")
    print(f"Processing {len(rows)} runs out of {total_count} total folders.")

    if not rows:
        raise RuntimeError("No runs with metrics found.")
    return pd.DataFrame(rows)

def flatten_columns(df):
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            a, b = col
            new_cols.append(f"{a}_{b}" if b else str(a))
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",
        default="/n/holylfs06/LABS/krajan_lab/Lab/zfish/sonja_results/results/rmappo-MultiAgentForagingEnv-check/",
        help="Base directory containing experiment run folders.")
    parser.add_argument("--outdir",
        default="sweep_summary_csvs",
        help="Where to save CSVs.")
    parser.add_argument("--date-prefix",
        help="Date prefix to filter run folders (e.g., '20250922_'). If specified, start-time and end-time are ignored.")
    parser.add_argument("--start-time",
        type=str,
        help="Start timestamp (format: YYYYMMDD_HHMMSS, e.g., 20250923_000000). Only include runs starting from this time.")
    parser.add_argument("--end-time",
        type=str,
        help="End timestamp (format: YYYYMMDD_HHMMSS, e.g., 20250924_235959). Only include runs up to this time.")
    args = parser.parse_args()

    base_dir = Path(args.base).expanduser()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse timestamp arguments and create ID string
    start_time = None
    end_time = None
    id_string = args.date_prefix if args.date_prefix else ""

    if not args.date_prefix:
        # Only parse timestamps if date_prefix is not specified
        if args.start_time:
            try:
                start_time = datetime.strptime(args.start_time, "%Y%m%d_%H%M%S")
                print(f"Filtering runs starting from: {start_time}")
            except ValueError:
                raise ValueError(f"Invalid start-time format: {args.start_time}. Use YYYYMMDD_HHMMSS")

        if args.end_time:
            try:
                end_time = datetime.strptime(args.end_time, "%Y%m%d_%H%M%S")
                print(f"Filtering runs up to: {end_time}")
            except ValueError:
                raise ValueError(f"Invalid end-time format: {args.end_time}. Use YYYYMMDD_HHMMSS")

        # Create ID string from timestamps
        if args.start_time and args.end_time:
            id_string = f"{args.start_time}_{args.end_time}"
        elif args.start_time:
            id_string = f"{args.start_time}_onwards"
        elif args.end_time:
            id_string = f"until_{args.end_time}"
        else:
            id_string = "all_runs"

    df = collect(base_dir, args.date_prefix, start_time, end_time)
    print(df.columns)

    # Rename columns for consistency
    df = df.rename(columns={
        "average_total_episode_reward": "avg_total_episode_reward",
        "average_auc_success": "avg_auc_success",
        "average_auc_non-tracking": "avg_auc_nontracking",
    })

    group = (
        df.groupby(["vd", "large_move_penalty", "large_turn_penalty", "food_detection_range", "walkerbots"], as_index=True)
        .agg({
            "total_eating_events": ["mean", "std", "count"],
            "eating_events_per_episode": ["mean", "std"],
            "avg_total_episode_reward": ["mean", "std"],
            "avg_auc_success": ["mean", "std"],
            "avg_auc_nontracking": ["mean", "std"],
        })
        .sort_index()
    )

    group_flat = flatten_columns(group.reset_index())

    # Save CSVs
    df.to_csv(outdir / f"runs_raw_metrics_{id_string}.csv", index=False)
    group_flat.to_csv(outdir / f"condition_summary_flat_{id_string}.csv", index=False)

    print(f"\nSaved:\n  - {outdir/f'runs_raw_metrics_{id_string}.csv'}\n  - {outdir/f'condition_summary_flat_{id_string}.csv'}")

if __name__ == "__main__":
    main()
