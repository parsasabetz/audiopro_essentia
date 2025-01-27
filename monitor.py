"""
Module for system performance monitoring.
Enhanced docstrings and inline comments for clarity.
"""

import time
import psutil
import GPUtil
import numpy as np
import logging
import os
import threading
from typing import List

logger = logging.getLogger(__name__)

def monitor_cpu_usage(process: psutil.Process, 
                     cpu_usage_list: List[float],
                     active_cores_list: List[int],
                     stop_flag: threading.Event) -> None:
    """
    Continuously collects per-core CPU usage and normalizes CPU load by core count.

    Args:
        process: Reference to the current process object.
        cpu_usage_list: List to store CPU usage values (% of a single core).
        active_cores_list: List to track the number of active cores at each measurement.
        stop_flag: Event to signal this monitoring thread to stop.
    """
    cpu_count = psutil.cpu_count()
    while not stop_flag.is_set():
        # Get per-CPU percentage
        per_cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        active_cores = sum(1 for cpu in per_cpu_percent if cpu > 1)
        active_cores_list.append(active_cores)

        cpu_percent = process.cpu_percent() / cpu_count
        if 0 <= cpu_percent <= 100:
            cpu_usage_list.append(cpu_percent)
        time.sleep(0.1)

def print_performance_stats(start_time, end_time, cpu_usage_list, active_cores_list):
    """
    Prints a summary of CPU, memory, and GPU usage, along with execution time.

    Args:
        start_time: Timestamp at start of processing.
        end_time: Timestamp at end of processing.
        cpu_usage_list: Normalized CPU usage data.
        active_cores_list: List of active core counts over time.
    """
    process = psutil.Process(os.getpid())

    # Filter outliers (values beyond 2 standard deviations)
    if cpu_usage_list:
        mean_cpu = np.mean(cpu_usage_list)
        std_cpu = np.std(cpu_usage_list)
        filtered_measurements = [x for x in cpu_usage_list 
                               if (mean_cpu - 2 * std_cpu) <= x <= (mean_cpu + 2 * std_cpu)]

        avg_cpu = np.mean(filtered_measurements) if filtered_measurements else 0
        peak_cpu = np.max(filtered_measurements) if filtered_measurements else 0
    else:
        avg_cpu = peak_cpu = 0

    # Memory Info
    memory_info = process.memory_info()
    memory_used = memory_info.rss / (1024 ** 3)
    system_memory = psutil.virtual_memory()
    memory_total = system_memory.total / (1024 ** 3)

    # GPU Info
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_info.append({
            "name": gpu.name,
            "load": f"{gpu.load*100:.1f}%",
            "memory_used": f"{gpu.memoryUsed:.1f}MB",
            "memory_total": f"{gpu.memoryTotal:.1f}MB"
        })

    # Time Info
    execution_time = end_time - start_time

    # Calculate core usage statistics
    avg_cores = np.mean(active_cores_list) if active_cores_list else 0
    max_cores = max(active_cores_list) if active_cores_list else 0

    # Print formatted output
    print("\n" + "="*50)
    print("PERFORMANCE METRICS")
    print("="*50)
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

    if gpu_info:
        print("\nGPU Statistics:")
        for idx, gpu in enumerate(gpu_info):
            print(f"GPU {idx}:")
            print(f"├── Name: {gpu["name"]}")
            print(f"├── Load: {gpu["load"]}")
            print(f"├── Memory Used: {gpu["memory_used"]}")
            print(f"└── Memory Total: {gpu["memory_total"]}")
    print("\n" + "="*50)