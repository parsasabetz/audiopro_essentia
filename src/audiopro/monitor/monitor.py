"""
Module for system performance monitoring.
Enhanced docstrings and inline comments for clarity.
"""

# Standard library imports
import os
import logging
import threading
from typing import List

# Third-party imports
import numpy as np
import psutil

# Configure logging
logger = logging.getLogger(__name__)


def monitor_cpu_usage(
    cpu_usage_list: List[float],
    active_cores_list: List[int],
    stop_flag: threading.Event,
    max_samples: int = 1000,  # limit stored samples to 1000
) -> None:
    """
    Continuously collects per-core CPU usage and normalizes CPU load by core count.

    Args:
        cpu_usage_list: List to store CPU usage values (% of a single core).
        active_cores_list: List to track the number of active cores at each measurement.
        stop_flag: Event to signal this monitoring thread to stop.
        max_samples: Maximum number of samples to store in the lists.
    """
    cpu_count = psutil.cpu_count()
    while not stop_flag.is_set():
        try:
            # Single call with per-core usage
            per_cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            cpu_percent = sum(per_cpu_percent) / cpu_count
            active_cores = np.sum(np.array(per_cpu_percent) > 1)

            # Append new samples; remove oldest if size exceeds max_samples
            cpu_usage_list.append(cpu_percent)
            if len(cpu_usage_list) > max_samples:
                cpu_usage_list.pop(0)

            active_cores_list.append(active_cores)
            if len(active_cores_list) > max_samples:
                active_cores_list.pop(0)
        except Exception as e:
            logger.error(f"Error monitoring CPU usage: {str(e)}")


def print_performance_stats(
    start_time,
    end_time,
    cpu_usage_list,
    active_cores_list,
):
    """
    Print performance statistics or basic execution summary.

    This function provides either detailed performance metrics including CPU usage,
    memory consumption, and core utilization, or a simplified execution summary
    if performance monitoring was skipped.

    Args:
        start_time (float): Unix timestamp of process start
        end_time (float): Unix timestamp of process completion
        cpu_usage_list (List[float]): Collection of CPU usage measurements
        active_cores_list (List[int]): Collection of active core counts

    Note:
        - CPU measurements are filtered to remove statistical outliers
        - Memory usage is reported in GB
        - If monitoring was skipped (empty lists), only execution time is shown
    """
    execution_time = end_time - start_time
    
    if not cpu_usage_list or not active_cores_list:
        print("\n" + "=" * 50)
        print("EXECUTION SUMMARY")
        print("=" * 50)
        print(f"\nExecution Time: {execution_time:.4f} seconds")
        print("Performance monitoring was skipped")
        print("\n" + "=" * 50)
        return

    process = psutil.Process(os.getpid())

    # Vectorized filtering of outliers
    if cpu_usage_list:
        cpu_array = np.array(cpu_usage_list)
        mean_cpu = np.mean(cpu_array)
        std_cpu = np.std(cpu_array)
        filtered_measurements = cpu_array[
            (cpu_array >= mean_cpu - 2 * std_cpu)
            & (cpu_array <= mean_cpu + 2 * std_cpu)
        ]
        avg_cpu = (
            float(np.mean(filtered_measurements)) if filtered_measurements.size else 0
        )
        peak_cpu = (
            float(np.max(filtered_measurements)) if filtered_measurements.size else 0
        )
    else:
        avg_cpu = peak_cpu = 0

    # Memory Info
    memory_info = process.memory_info()
    memory_used = memory_info.rss / (1024**3)
    system_memory = psutil.virtual_memory()
    memory_total = system_memory.total / (1024**3)

    # Time Info
    execution_time = end_time - start_time

    # Calculate core usage statistics using NumPy
    active_cores_array = np.array(active_cores_list)
    avg_cores = float(np.mean(active_cores_array)) if active_cores_array.size else 0
    max_cores = int(np.max(active_cores_array)) if active_cores_array.size else 0

    # Print formatted output
    print("\n" + "=" * 50)
    print("PERFORMANCE METRICS")
    print("=" * 50)
    print(f"\nExecution Time: {execution_time:.4f} seconds")

    print("\nProgram CPU Statistics:")
    print(f"├── Average CPU Usage per Core: {avg_cpu:.2f}%")
    print(f"├── Peak CPU Usage per Core: {peak_cpu:.2f}%")
    print(f"├── Average Active Cores: {avg_cores:.1f}")
    print(f"├── Peak Active Cores: {max_cores}")
    print(f"└── System CPU Cores: {psutil.cpu_count()}")

    print("\nProgram Memory Statistics:")
    print(f"├── Program Memory Usage: {memory_used:.4f} GB")
    print(f"└── System Total Memory: {memory_total:.4f} GB")

    print("\n" + "=" * 50)
