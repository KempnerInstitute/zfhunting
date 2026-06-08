#!/bin/bash

# Base directory
base_dir="/n/holylfs06/LABS/krajan_lab/Lab/zfish/ramalik/results/rmappo-MultiAgentForagingEnv-check/"

for dir in "$base_dir"*/; do
    # Extract just the directory name for pattern matching
    dir_name=$(basename "$dir")

    echo "Dir name: $dir_name"
    
    # Check if directory matches the pattern
    if [[ "$dir_name" =~ ^20250924_[0-9]{6}_1_final_bao_vd_[-+]?[0-9]*\.?[0-9]+_lmp_[-+]?[0-9]*\.?[0-9]+_ltp_[-+]?[0-9]*\.?[0-9]+_fdr_10_wb_[0-9]+_run_[0-9]+$ ]]; then
        echo "Processing directory: $dir_name"
        # Check if performance_metrics.txt exists and contains "Forward speed mean"
        
        python update_performance_metrics.py ${dir}/outputs
    fi
done