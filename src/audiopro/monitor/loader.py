# typing imports
from typing import Callable, Tuple

# Standard library imports
import importlib


def load_monitor_functions() -> Tuple[Callable, Callable]:
    """
    Dynamically imports the monitor module and returns two functions from it.

    This function imports the `monitor.monitor` module from the `audiopro` package
    and retrieves two functions: `monitor_cpu_usage` and `print_performance_stats`.
    These functions are returned as a tuple.

    Returns:
        tuple[Callable, Callable]: A tuple containing the `monitor_cpu_usage` and
        `print_performance_stats` functions from the `monitor.monitor` module.
    """

    monitor_module = importlib.import_module(".monitor", package="audiopro")
    return (monitor_module.monitor_cpu_usage, monitor_module.print_performance_stats)
