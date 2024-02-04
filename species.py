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


# A base class for species
class BaseSpecies:
    def __init__(self, content, born_rate, death_rate, prey_rate):
        """A base class for species

        the instance of the species class corresponds to a cell in the world,
        it should handle the basic attributes of a species, such as reproduce, die, born, etc.

        Args:
            number (_type_): _description_
            born_rate (_type_): _description_
            death_rate (_type_): _description_
            prey_rate (_type_): _description_
        """
        self.content = content
        self.born_rate = born_rate
        self.death_rate = death_rate
        self.prey_rate = prey_rate

    def __str__(self):
        return str(self.content)

    def __repr__(self):
        return f"Content: {self.content}, Reproduction Rate: {self.born_rate}, Death Rate: {self.death_rate}, Prey Rate: {self.prey_rate}"

    @abstractmethod
    def die(self):
        """only natural death, not predation or spawning death"""
        pass

    @abstractmethod
    def born(self):
        pass

    # Comparison methods
    def __lt__(self, other):
        if isinstance(other, BaseSpecies):
            return self.content < other.content
        else:  # Assuming other is int or float
            return self.content < other

    def __le__(self, other):
        if isinstance(other, BaseSpecies):
            return self.content <= other.content
        else:  # Assuming other is int or float
            return self.content <= other

    def __eq__(self, other):
        if isinstance(other, BaseSpecies):
            return self.content == other.content
        else:  # Assuming other is int or float
            return self.content == other

    def __ne__(self, other):
        if isinstance(other, BaseSpecies):
            return self.content != other.content
        else:  # Assuming other is int or float
            return self.content != other

    def __gt__(self, other):
        if isinstance(other, BaseSpecies):
            return self.content > other.content
        else:  # Assuming other is int or float
            return self.content > other

    def __ge__(self, other):
        if isinstance(other, BaseSpecies):
            return self.content >= other.content
        else:  # Assuming other is int or float
            return self.content >= other

    # calculation methods:
    def __add__(self, other):
        if isinstance(other, BaseSpecies):
            new_content = self.content + other.content
            # return self.content + other.content
        else:  # Assuming other is int or float
            new_content = self.content + other
            # return self.content + other
        return self.__class__(content=new_content)

    def __sub__(self, other):
        if isinstance(other, BaseSpecies):
            new_content = self.content - other.content
            # return self.content - other.content
        else:  # Assuming other is int or float
            new_content = self.content - other
            # return self.content - other
        return self.__class__(content=new_content)

    def __mul__(self, other):
        if isinstance(other, BaseSpecies):
            new_content = self.content * other.content
            # return self.content * other.content
        else:
            new_content = self.content * other
            # return self.content * other
        return self.__class__(content=new_content)

    def __div__(self, other):
        if isinstance(other, BaseSpecies):
            new_content = self.content / other.content
        else:
            new_content = self.content / other
        return new_content

    def __radd__(self, other):
        if isinstance(other, BaseSpecies):
            new_content = self.content + other.content
        else:  # Assuming other is int or float
            new_content = self.content + other
        return new_content

    def __rsub__(self, other):
        if isinstance(other, BaseSpecies):
            new_content = self.content - other.content
        else:
            new_content = self.content - other
        return new_content

    def __rmul__(self, other):
        if isinstance(other, BaseSpecies):
            new_content = self.content * other.content
        else:
            new_content = self.content * other
        return new_content

    def __truediv__(self, other):
        if isinstance(other, BaseSpecies):
            new_content = self.content / other.content
        else:
            new_content = self.content / other
        return new_content

    def __int__(self):
        return int(self.content)

    def __float__(self):
        return float(self.content)

    def __bool__(self):
        return bool(self.content)

    def __abs__(self):
        return abs(self.content)


class PreySpecies(BaseSpecies):
    def __init__(self, content, born_rate=0.1, death_rate=0.08, prey_rate=0.1):
        if isinstance(content, BaseSpecies):
            content = content.content
        super().__init__(content, born_rate, death_rate, prey_rate)

    def die(self):
        self.content = self.content * (1 - self.death_rate)

    def born(self):
        self.content = self.content * (1 + self.prey_rate)


class PredatorSpecies(BaseSpecies):
    def __init__(self, content, born_rate=0.1, death_rate=0.08, prey_rate=0.1):
        if isinstance(content, BaseSpecies):
            content = content.content
        super().__init__(content, born_rate, death_rate, prey_rate)

    def die(self):
        self.content = self.content * (1 - self.death_rate)

    def born(self):
        self.content = self.content * (1 + self.prey_rate)


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
