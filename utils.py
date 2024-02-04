import time

# a Timer class to keep track of the time
# record the year and month
# implement str, add and sub methods
# implement a property to get the current month
# implement a setter to set the current month
# implement a deleter to delete the current month


class Timer:
    def __init__(self, year: int = 2000, month: int = 1):
        self.year = year
        self.month = month

    def __str__(self):
        return f"{self.year}-{self.month}"

    def __add__(self, other):
        if isinstance(other, int):
            self.month += other

        elif isinstance(other, Timer):
            self.year += other.year
            self.month += other.month

        elif isinstance(other, tuple):
            self.year += other[0]
            self.month += other[1]

        else:
            raise ValueError("The other should be an integer.")

        if self.month > 12:
            self.year += 1
            self.month -= 12

        return self

    def __sub__(self, other):
        if isinstance(other, int):
            self.month -= other

        elif isinstance(other, Timer):
            self.year -= other.year
            self.month -= other.month

        elif isinstance(other, tuple):
            self.year -= other[0]
            self.month -= other[1]

        else:
            raise ValueError("The other should be an integer.")

        if self.month < 1:
            self.year -= 1
            self.month += 12

        return self

    def get_year(self):
        return self.year

    def set_year(self, year):
        self.year = year

    def get_month(self):
        return self.month

    def set_month(self, month):
        self.month = month
