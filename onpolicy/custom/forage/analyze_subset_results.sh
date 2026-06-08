#!/bin/bash

# Base directory
base_dir="/n/holylfs06/LABS/krajan_lab/Lab/zfish/sonja_results/results/rmappo-MultiAgentForagingEnv-check/"
raaghav_dir="/n/holylfs06/LABS/krajan_lab/Lab/zfish/ramalik/results/rmappo_MultiAgentForagingEnv-check/"
mkdir -p $raaghav_dir

# Find subdirectories matching the pattern and run the Python script
for dir in "$base_dir"*/; do
    # Extract just the directory name for pattern matching
    dir_name=$(basename "$dir")
    
    # Check if directory matches the pattern
    if [[ "$dir_name" =~ ^20250922_[0-9]{6}_1_bao_vd_[-+]?[0-9]*\.?[0-9]+_lmp_[-+]?[0-9]*\.?[0-9]+_ltp_[-+]?[0-9]*\.?[0-9]+_fdr_10_run_[0-9]+$ ]]; then
        echo "Processing directory: $dir_name"
    # Check if performance_metrics.txt exists and contains "Forward speed mean"
    if ! grep -q "^Forward speed mean" "${dir}/outputs/figures/performance_metrics.txt"; then
        # cp -r "$dir" "/n/holylfs06/LABS/krajan_lab/Lab/zfish/ramalik/results/rmappo_MultiAgentForagingEnv-check/"
        # new_dir="/n/holylfs06/LABS/krajan_lab/Lab/zfish/ramalik/results/rmappo_MultiAgentForagingEnv-check/$dir_name"
        python report_behavior_shs.py "${dir}/outputs" --update_performance_metrics_only
    fi
done