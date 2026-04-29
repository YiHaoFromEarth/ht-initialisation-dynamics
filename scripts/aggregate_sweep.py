import gc
import os
import argparse
from pathlib import Path
from src.analysis import collect_sweep_metrics
# Limit internal library parallelism to 1 thread per worker
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch # Import torch AFTER setting these
torch.set_num_threads(1)

def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics from a hyperparameter sweep on HPC.")
    parser.add_argument("--sweep-dir", type=str, default="training_runs/mnist_mlp_sweep", help="Directory containing sweep subdirectories.")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers for aggregation.")
    parser.add_argument("--output-path", type=str, default="sweep_metrics.parquet", help="Path to save the aggregated metrics dataframe.")
    args = parser.parse_args()

    print("--- HPC Metric Aggregation Start ---")
    print(f"Target Directory: {args.sweep_dir}")
    print(f"Workers: {args.max_workers}")
    print(f"Output Path: {args.output_path}")

    # 2. Execution
    try:
        df = collect_sweep_metrics(
            sweep_dir=args.sweep_dir,
            max_workers=args.max_workers,
            output_path=args.output_path
        )

        print("\n--- Aggregation Complete ---")
        print(f"Final Dataframe Shape: {df.shape}")

    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")

    finally:
        # Final cleanup
        gc.collect()

if __name__ == "__main__":
    # This block is MANDATORY for multiprocessing to work on Linux/HPC
    main()
