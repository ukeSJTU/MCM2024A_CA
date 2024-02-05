import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
from typing import Union, Literal
import time
from pathlib import Path
import copy
from scipy.ndimage import uniform_filter, maximum_filter
import math
from abc import abstractmethod, ABC
import sys
from tqdm import tqdm
from matplotlib.colors import Normalize
import seaborn as sns


from utils import Calendar, nanmax_pooling, nanmean_pooling, resize_with_pooling
from species import LampreySpecies, PreySpecies, PredatorSpecies
from world import LampreyWorld, PreyWorld, PredatorWorld, Terrain

# from rules import rulesets
import rules

K = 48


class Ecosystem:
    def __init__(
        self,
        lamprey_world: LampreyWorld,
        prey_world: PreyWorld,
        predator_world: PredatorWorld,
        terrain: Terrain,
        pool_size: int = 10,
        pool_method: Literal["max", "mean"] = "max",
        output_dir: Path = Path("./output") / str(int(time.time())),
        calendar: Calendar = Calendar(2000, 1),
    ):
        self.lamprey_world = lamprey_world
        self.prey_world = prey_world
        self.predator_world = predator_world
        self.terrain = terrain

        self.pool_size = pool_size  # Adjust based on desired granularity
        self.pool_method = pool_method
        self.output_dir = output_dir  # the directory to save the output

        self.output_prey_dir = self.output_dir / "prey"
        self.output_predator_dir = self.output_dir / "predator"
        self.output_lamprey_dir = self.output_dir / "lamprey"

        self.output_prey_dir.mkdir(parents=True, exist_ok=True)
        self.output_predator_dir.mkdir(parents=True, exist_ok=True)
        self.output_lamprey_dir.mkdir(parents=True, exist_ok=True)

        assert (
            self.lamprey_world.width
            == self.prey_world.width
            == self.predator_world.width
        ), "The width of the worlds should be the same"
        assert (
            self.lamprey_world.height
            == self.prey_world.height
            == self.predator_world.height
        ), "The height of the worlds should be the same"

        self.width = self.lamprey_world.width
        self.height = self.lamprey_world.height

        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 5))
        self.cbars = []

        self.iter = 0
        self.calendar = calendar

    def debug(self, n):
        """_summary_

        Args:
            n (int): rule index
        """
        print(f"Iteration {self.iter}, {self.calendar}, Rule: {n}")
        print(self.lamprey_world)
        print(self.prey_world)
        print(self.predator_world)

    def step(self):
        # every step is one month in the real world.
        # use self.timer to iterate over time, every step should be one month
        # there are multiple rules for the lampreys to consume food, reproduce and die
        # we assume that every lamprey lives 7 years. The first 4 years are larval lampreys, the last 3 years are adult lampreys
        # 1. every larval lamprey turns into adults in 4 years. Their gender is determined based on the amount of food in the cell
        # 2. every adult lamprey consumes 0.67 food in the cell they are in
        # 3. every adult lamprey has a chance to reproduce with another adult lamprey in the same cell. The number of larval lampreys they produce is k(122) * MP * (1-MP) * A, where MP is male percentage and A is the number of food in the cell.
        # 4. every adult lamprey has a chance to die. The probability of dying is 0.02
        # 5. every larval lamprey has a chance to die. The probability of dying is 0.39 (1 - 61%)
        # 6. the mated lampreys will die after they reproduce
        # 7. the prey of lampreys will reproduce every year. The number of prey they reproduce is 2 times the number of lampreys they eat

        rules.rulesets.apply(ecosystem=self)
        self.iter += 1
        self.calendar += 1

    def visualize(
        self, save: bool = False, show: bool = True, filename: Union[str, Path] = None
    ):
        # lamprey_fig, color_mat = self.lamprey_world.show()
        # prey_fig, prey_mat = self.prey_world.show()
        if save == False and show == False:
            return

        if filename is None:
            filename = f"{self.iter}.png"

        # prey_data = np.array(self.prey_world.matrix)
        # predator_data = np.array(self.predator_world.matrix)

        male_percentage_data = np.array(
            [
                [
                    (
                        cell.describe()[2] / cell.describe()[0]
                        if cell.describe()[0] != 0
                        else np.nan
                    )
                    for cell in row
                ]
                for row in self.lamprey_world.matrix
            ]
        )
        larval_adult_ratio_data = np.array(
            [
                [
                    (
                        cell.describe()[0] / cell.describe()[1]
                        if cell.describe()[1] != 0
                        else np.nan
                    )
                    for cell in row
                ]
                for row in self.lamprey_world.matrix
            ]
        )

        # male_percentage_data_

        prey_data = np.array(
            [[cell.content for cell in row] for row in self.prey_world.matrix]
        )
        predator_data = np.array(
            [[cell.content for cell in row] for row in self.predator_world.matrix]
        )

        # max/mean pool the data
        # pool_size = 10  # Adjust based on desired granularity
        male_percentage_data_pooled = resize_with_pooling(
            data=male_percentage_data,
            new_size=(self.pool_size, self.pool_size),
            method=self.pool_method,
        )
        male_percentage_mask = np.isnan(male_percentage_data_pooled)
        # male_percentage_data_pooled = self.apply_pooling(
        #     male_percentage_data, size=self.pool_size, method=self.pool_method
        # )

        larval_adult_ratio_data_pooled = resize_with_pooling(
            data=larval_adult_ratio_data,
            new_size=(self.pool_size, self.pool_size),
            method=self.pool_method,
        )
        larval_adult_ratio_mask = np.isnan(larval_adult_ratio_data_pooled)

        prey_data_pooled = resize_with_pooling(
            prey_data,
            new_size=(self.pool_size, self.pool_size),
            method=self.pool_method,
        )  # or 'mean'

        predator_data_pooled = resize_with_pooling(
            predator_data,
            new_size=(self.pool_size, self.pool_size),
            method=self.pool_method,
        )

        # Normalize the data to scale between 0 and 255
        norm = Normalize(
            vmin=0,
            vmax=max(
                male_percentage_data_pooled.max(),
                larval_adult_ratio_data_pooled.max(),
                prey_data_pooled.max(),
                predator_data_pooled.max(),
            ),
        )
        male_percentage_normalized = norm(male_percentage_data_pooled) * 255
        larval_adult_ratio_normalized = norm(larval_adult_ratio_data_pooled) * 255
        prey_normalized = norm(prey_data_pooled) * 255
        predator_normalized = norm(predator_data_pooled) * 255

        # sns.heatmap(
        #     male_percentage_data_pooled,
        #     ax=self.axes[0][0],
        #     square=True,
        #     cmap="coolwarm",
        #     center=1,
        #     # cbar_kws={"label": "Sex Ratio (Males/Females)"},
        # )
        self.axes[0][0].imshow(
            male_percentage_normalized,
            cmap="viridis",
        )
        self.axes[0][0].set_title("Male Percentage")
        self.axes[0][0].set_axis_off()

        self.axes[0][1].imshow(larval_adult_ratio_normalized, cmap="binary")
        self.axes[0][1].set_title("Larval/Adult Ratio")
        self.axes[0][1].set_axis_off()

        self.axes[1][0].imshow(prey_normalized, cmap="PuBu")
        self.axes[1][0].set_title("Prey World")
        self.axes[1][0].set_axis_off()

        self.axes[1][1].imshow(predator_normalized, cmap="PuBu")
        self.axes[1][1].set_title("Predator World")
        self.axes[1][1].set_axis_off()

        if save:
            self.fig.savefig(str(self.output_dir / filename))

        if show:
            plt.pause(0.1)

        return

    @classmethod
    def apply_pooling(self, data, size=5, method: Literal["max", "mean"] = "max"):
        """Apply pooling to reduce the size of the data matrix.

        Parameters:
        - data: 2D numpy array of data to pool.
        - size: Size of the pooling window.
        - method: 'max' for max pooling, 'mean' for mean pooling.
        """
        if method == "max":
            return maximum_filter(data, size=size)[::size, ::size]
        elif method == "mean":
            return uniform_filter(data, size=size)[::size, ::size]
        else:
            raise ValueError("Method must be 'max' or 'mean'.")
