import time
import numpy as np


class Timer:
    def __init__(self, year: int = 2000, month: int = 1) -> None:
        self.year: int = year
        self.month: int = month

    def __str__(self) -> str:
        return f"{self.year}-{self.month}"

    def __add__(self, other) -> "Timer":
        if isinstance(other, int):
            self.month += other

        elif isinstance(other, Timer):
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

    def __sub__(self, other) -> "Timer":
        if isinstance(other, int):
            self.month -= other

        elif isinstance(other, Timer):
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
