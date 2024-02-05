import time
import numpy as np
from typing import Callable
import functools
from scipy.ndimage import generic_filter


class Calendar:
    def __init__(self, year: int = 2000, month: int = 1) -> None:
        self.year: int = year
        self.month: int = month

    def __str__(self) -> str:
        return f"{self.year}-{self.month}"

    def __add__(self, other) -> "Calendar":
        if isinstance(other, int):
            self.month += other

        elif isinstance(other, Calendar):
            self.year += other.year
            self.month += other.month

        elif isinstance(other, tuple):
            self.year += other[0]
            self.month += other[1]

        else:
            raise ValueError(f"Type{type(other)} is not supported.")

        if self.month > 12:
            self.year += 1
            self.month -= 12

        return self

    def __sub__(self, other) -> "Calendar":
        if isinstance(other, int):
            self.month -= other

        elif isinstance(other, Calendar):
            self.year -= other.year
            self.month -= other.month

        elif isinstance(other, tuple):
            self.year -= other[0]
            self.month -= other[1]

        else:
            raise ValueError(f"Type{type(other)} is not supported.")

        if self.month < 1:
            self.year -= 1
            self.month += 12

        return self

    def get_year(self) -> int:
        return self.year

    def set_year(self, year: int) -> None:
        self.year = year

    def get_month(self) -> int:
        return self.month

    def set_month(self, month: int) -> None:
        self.month = month


def timer(func: Callable):
    """Decorator to measure the total execution time of a function over multiple calls."""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        wrapper_timer.total_time += end_time - start_time
        return value

    wrapper_timer.total_time = 0
    return wrapper_timer


def softmax(lst):
    # Convert the list to a NumPy array for efficient computation
    arr = np.array(lst)
    # Exponentiate each element
    exp_arr = np.exp(arr - np.max(arr))  # Subtracting max for numerical stability
    # Calculate the sum of exponentiated values
    sum_exp_arr = np.sum(exp_arr)
    # Divide each exponentiated value by the sum to get probabilities
    probs = exp_arr / sum_exp_arr
    return probs


def nanmean_pooling(data, size=10):
    return generic_filter(data, np.nanmean, size=size, mode="reflect")


def nanmax_pooling(data, size=10):
    return generic_filter(data, np.nanmax, size=size, mode="reflect")


def resize_with_pooling(data, new_size=(5, 5), method="mean"):
    """
    Resize a matrix to new_size using pooling.

    Parameters:
    - data: 2D numpy array to pool.
    - new_size: tuple, the size of the output matrix.
    - method: str, either 'max' or 'mean' for the type of pooling.

    Returns:
    - Pooled and resized matrix.
    """
    output = np.zeros(new_size)
    step_x = data.shape[0] / new_size[0]
    step_y = data.shape[1] / new_size[1]

    for i in range(new_size[0]):
        for j in range(new_size[1]):
            block = data[
                int(i * step_x) : int((i + 1) * step_x),
                int(j * step_y) : int((j + 1) * step_y),
            ]

            if method == "max":
                # Use np.nanmax for max pooling to ignore np.nan values
                output[i, j] = np.nanmax(block)
            elif method == "mean":
                # Use np.nanmean for mean pooling to ignore np.nan values
                output[i, j] = np.nanmean(block)

    return output
