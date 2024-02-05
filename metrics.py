# the metrics.py file contains the functions to calculate the metrics for the ecosystem

import time
from tqdm import tqdm
import numpy as np
from typing import Union, Literal, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ecosystem import Ecosystem


class Metrics:
    def __init__(self, parent: "Ecosystem"):
        self.parent = parent

        self.info = {}

    def _average_sex_ratio(self):
        sum_male = 0
        sum_female = 0
        for row in range(self.parent.width):
            for col in range(self.parent.height):
                info = self.parent.lamprey_world.describe(row, col)
                sum_male += info[2]
                sum_female += info[3]

        return sum_male / sum_female if sum_female != 0 else -1

    def _average_larval_adult_ratio(self):
        sum_larval = 0
        sum_adult = 0
        for row in range(self.parent.width):
            for col in range(self.parent.height):
                info = self.parent.lamprey_world.describe(row, col)
                sum_larval += info[0]
                sum_adult += info[1]

        return sum_larval / sum_adult if sum_adult != 0 else -1

    def _adult_lamprey_density(self):
        sum_adult_lamprey = 0
        for row in range(self.parent.width):
            for col in range(self.parent.height):
                info = self.parent.lamprey_world.describe(row, col)
                sum_adult_lamprey += info[0]

        return sum_adult_lamprey / (self.parent.width * self.parent.height)

    def _larval_lamprey_density(self):
        sum_larval_lamprey = 0
        for row in range(self.parent.width):
            for col in range(self.parent.height):
                info = self.parent.lamprey_world.describe(row, col)
                sum_larval_lamprey += info[1]

        return sum_larval_lamprey / (self.parent.width * self.parent.height)

    def _lamprey_density(self):
        return self._adult_lamprey_density() + self._larval_lamprey_density()

    def _prey_density(self):
        sum_prey = 0
        for row in range(self.parent.width):
            for col in range(self.parent.height):
                sum_prey += float(self.parent.prey_world[row][col])

        return sum_prey / (self.parent.width * self.parent.height)

    def _predator_density(self):
        sum_predator = 0
        for row in range(self.parent.width):
            for col in range(self.parent.height):
                sum_predator += float(self.parent.predator_world[row][col])

        return sum_predator / (self.parent.width * self.parent.height)

    def get_metrics(
        self,
        key: Union[
            Literal["lamprey_density", "prey_density", "predator_density"], None
        ] = None,
    ):
        if key is None:
            return self.info
        else:
            if key == "lamprey_density":
                res = self._lamprey_density()
            elif key == "prey_density":
                res = self._prey_density()
            elif key == "predator_density":
                res = self._predator_density()
            else:
                raise ValueError(f"Invalid key: {key}")

            self.info[key] = np.append(self.info.get(key, []), res)
            return res
