import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
import time
from pathlib import Path
import copy
import math
from abc import abstractmethod, ABC
import sys
from tqdm import tqdm


class Species:
    def __init__(self, number, reproduction_rate, death_rate, prey_rate):
        self.number = number
        self.reproduction_rate = reproduction_rate
        self.death_rate = death_rate
        self.prey_rate = prey_rate

    def __str__(self):
        return f"Number: {self.number}, Reproduction Rate: {self.reproduction_rate}, Death Rate: {self.death_rate}, Prey Rate: {self.prey_rate}"


# and the lamprey, the prey, and the predator are all subclasses of the species class
class LampreySpecies(Species):
    def __init__(self, number, reproduction_rate, death_rate, prey_rate):
        super().__init__(number, reproduction_rate, death_rate, prey_rate)


class PreySpecies(Species):
    def __init__(self, number, reproduction_rate, death_rate, prey_rate):
        super().__init__(number, reproduction_rate, death_rate, prey_rate)


class PredatorSpecies(Species):
    def __init__(self, number, reproduction_rate, death_rate, prey_rate):
        super().__init__(number, reproduction_rate, death_rate, prey_rate)


class LampreySpecies:
    def __init__(self):
        # age 0-4: larval lampreys, no gender
        # age 5-7: adult lampreys, the first element male, the second female
        self.population = {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: [0, 0],
            6: [0, 0],
            7: [0, 0],
        }

    def __setitem__(self, key, value):
        self.population[key] = value

    def __getitem__(self, key):
        return self.population[key]

    def __str__(self):
        return ",".join(map(str, self.population.values()))

    def __repr__(self):
        s = str(
            (
                self.population[0]
                + self.population[1]
                + self.population[2]
                + self.population[3]
                + self.population[4],
                self.population[5][0] + self.population[6][0] + self.population[7][0],
                self.population[5][1] + self.population[6][1] + self.population[7][1],
            )
        )
        return str(self.population)

    def describe(self):
        # adult_cnt = self.population[5] + self.population[6] + self.population[7]
        larval_cnt = (
            self.population[0]
            + self.population[1]
            + self.population[2]
            + self.population[3]
            + self.population[4]
        )
        male_cnt = self.population[5][0] + self.population[6][0] + self.population[7][0]
        female_cnt = (
            self.population[5][1] + self.population[6][1] + self.population[7][1]
        )
        adult_cnt = male_cnt + female_cnt

        return adult_cnt, larval_cnt, male_cnt, female_cnt
