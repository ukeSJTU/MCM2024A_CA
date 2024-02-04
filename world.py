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

from species import LampreySpecies, PreySpecies, PredatorSpecies


# define a base class world:
class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.iter = 0  # the number of iterations the world has gone through

        self.matrix = [[0 for _ in range(self.width)] for _ in range(self.height)]

        # the name is default to the name of class
        self.name = self.__class__.__name__

    def __str__(self):
        return f"{self.name}: \n" + f"{self.debug()}"

    def __repr__(self) -> str:
        pass

    def debug(self, start_row=0, end_row=5, start_col=0, end_col=5):
        """
        Debug the current world.
        Show part of the world.
        """
        s = ""
        for row in self.matrix[start_row:end_row]:
            s += " ".join(map(str, row[start_col:end_col])) + "\n"
        return s

    # a method that should be implemented by the subclass
    # @abstractmethod
    def step(self):
        pass

    # @abstractmethod
    def show(self, save: bool, show: bool, filename: str):
        pass

    def simulate(self, n: int = 10):
        for _ in range(n):
            self.step()
            # self.show()

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __getitem__(self, key):
        return self.matrix[key]


class LampreyWorld(World):
    def __init__(
        self,
        init_value: LampreySpecies,
        init_sex_ratio,
        width=100,
        height=100,
    ):
        super().__init__(width, height)
        # each element of the matrix of LampryWorld is a LampreyPop object

        self.matrix = [
            [copy.deepcopy(init_value) for _ in range(self.width)]
            for _ in range(self.height)
        ]
        self.sex_ratio = init_sex_ratio

        # use a mask to imply whether the cell is lake(1) or ocean(0)
        # init mask where the left half is ocean and the right half is lake
        self.mask = [
            [0 if col < self.width / 2 else 1 for col in range(self.width)]
            for _ in range(self.height)
        ]
        # define a 2d direction vector to show how to move from the lake to the sea

        self.adult_cnt = 0
        for row in range(self.height):
            for col in range(self.width):
                for age in [5, 6, 7]:
                    self.adult_cnt += sum(self.matrix[row][col][age])

        self.male_cnt = int(init_sex_ratio * self.adult_cnt)
        self.female_cnt = self.adult_cnt - self.male_cnt

        self.larval_death_rate = 0.7
        self.male_death_rate = 0.2
        self.female_death_rate = 0.2

    def show(self, save=False, show=False, filename="output.png"):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # convert the self.matrix element from instances of LampreyPop to a tuple of three integers
        # the three integers are numbers of larval lampreys, male lampreys and female lampreys in each cell
        # then scale the numbers to 0-255, so that the numbers are also colors for that pixel when plotting

        color_mat = []
        for row in range(self.height):
            color_row = []
            for col in range(self.width):
                adult_cnt, larval_cnt, male_cnt, female_cnt = self.describe(row, col)
                color_row.append(
                    (
                        larval_cnt,
                        male_cnt,
                        female_cnt,
                    )
                )
            color_mat.append(color_row)

        # scale
        color_mat = np.array(color_mat)
        color_mat = color_mat / color_mat.max() * 255

        ax.imshow(color_mat, cmap="viridis")
        if show:
            plt.show()
        return fig, color_mat

    def describe(self, row, col, key: str = "all"):
        adult_cnt, larval_cnt, male_cnt, female_cnt = self.matrix[row][col].describe()
        return adult_cnt, larval_cnt, male_cnt, female_cnt


class PreyWorld(World):
    def __init__(
        self,
        init_value,
        width=100,
        height=100,
    ):
        super().__init__(width, height)
        # self.init = init_value
        self.matrix = [
            [random.uniform(500, init_value) for _ in range(self.width)]
            for _ in range(self.height)
        ]

        self.birth_rate = 2  # the birth rate of the prey of lampreys

    def show(self, save=False, show=False, filename="output.png"):
        plt.plot()
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(self.matrix, cmap="PuBu")
        if show:
            plt.show()
        return fig, self.matrix


class PredatorWorld(World):
    def __init__(
        self,
        init_value,
        width=100,
        height=100,
    ):
        super().__init__(width, height)
        # self.init = init_value
        self.matrix = [
            [init_value for _ in range(self.width)] for _ in range(self.height)
        ]

        self.birth_rate = 2  # the birth rate of the prey of lampreys

    def show(self, save=False, show=False, filename="output.png"):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(self.matrix, cmap="Reds")
        if show:
            plt.show()
        return fig
