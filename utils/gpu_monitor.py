#!/usr/bin/env python3
"""
GPU Monitoring Script for SLURM jobs.
Logs GPU utilization and memory usage to a CSV file.
"""

import subprocess
import time
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor GPU utilization.")
    parser.add_argument(
        "--interval", 
        type=int, 
        default=5, 
        help="Polling interval in seconds [default: 5]"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="gpu_utilization_log.csv", 
        help="Output CSV file path"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting GPU monitoring every {args.interval}s. Logging to {args.output}")
    
    # Define the specific metrics we want from nvidia-smi
    query = "timestamp,index,name,utilization.gpu,memory.used,memory.total"
    
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            # Write standard CSV header
            f.write("timestamp,gpu_index,gpu_name,gpu_utilization_pct,memory_used_MiB,memory_total_MiB\n")
            
            while True:
                # Run nvidia-smi
                result = subprocess.run(
                    [
                        "nvidia-smi", 
                        f"--query-gpu={query}", 
                        "--format=csv,noheader,nounits"
                    ],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Write the output directly to the file and flush to ensure it saves immediately
                f.write(result.stdout)
                f.flush()
                
                time.sleep(args.interval)
                
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nMonitoring stopped safely.")

if __name__ == "__main__":
    main()

