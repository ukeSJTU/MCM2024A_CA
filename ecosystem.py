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


from utils import Timer
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
        output_dir: Path = Path("./output") / str(int(time.time())),
        timer: Timer = Timer(2000, 1),
    ):
        self.lamprey_world = lamprey_world
        self.prey_world = prey_world
        self.predator_world = predator_world
        self.terrain = terrain

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
        self.timer = timer

    def debug(self, n):
        """_summary_

        Args:
            n (int): rule index
        """
        print(f"Iteration {self.iter}, {self.timer}, Rule: {n}")
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
        self.timer += 1

    def visualize(
        self, save: bool = False, show: bool = True, filename: Union[str, Path] = None
    ):
        # lamprey_fig, color_mat = self.lamprey_world.show()
        # prey_fig, prey_mat = self.prey_world.show()

        if filename is None:
            filename = f"{self.iter}.png"

        # prey_data = np.array(self.prey_world.matrix)
        # predator_data = np.array(self.predator_world.matrix)

        prey_data = np.array(
            [[cell.content for cell in row] for row in self.prey_world.matrix]
        )
        predator_data = np.array(
            [[cell.content for cell in row] for row in self.predator_world.matrix]
        )

        # max/mean pool the data
        pool_size = 10  # Adjust based on desired granularity
        prey_data_pooled = self.apply_pooling(
            prey_data, size=pool_size, method="max"
        )  # or 'mean'
        predator_data_pooled = self.apply_pooling(
            predator_data, size=pool_size, method="max"
        )

        # Normalize the data to scale between 0 and 255
        norm = Normalize(
            vmin=0, vmax=max(prey_data_pooled.max(), predator_data_pooled.max())
        )
        prey_normalized = norm(prey_data) * 255
        predator_normalized = norm(predator_data) * 255

        prey_ax = sns.heatmap(prey_normalized, cmap="viridis", ax=self.axes[0, 0])
        predator_ax = sns.heatmap(
            predator_normalized, cmap="viridis", ax=self.axes[0, 1]
        )

        if save:
            prey_fig = prey_ax.get_figure()
            prey_fig.savefig(str(self.output_prey_dir / filename))

            predator_fig = predator_ax.get_figure()
            predator_fig.savefig(str(self.output_predator_dir / filename))

        if show:
            plt.tight_layout()
            plt.pause(0.1)

        return

        for ax in self.axes.flat:
            ax.clear()

        for cbar in self.cbars[::-1]:
            cbar.remove()
        self.cbars.clear()

        # Visualize the prey_world
        ax1 = self.axes[0, 0]
        prey_plot = ax1.imshow(prey_normalized, cmap="viridis", interpolation="nearest")
        ax1.set_title("Prey World")
        cbar = self.fig.colorbar(prey_plot, ax=ax1, fraction=0.046, pad=0.04)
        self.cbars.append(cbar)

        # Visualize the predator_world
        ax2 = self.axes[0, 1]
        predator_plot = ax2.imshow(
            predator_normalized, cmap="viridis", interpolation="nearest"
        )
        ax2.set_title("Predator World")
        cbar = self.fig.colorbar(predator_plot, ax=ax2, fraction=0.046, pad=0.04)
        self.cbars.append(cbar)

        if save:
            # Save the figures to the specified output directory
            self.fig.savefig(str(self.output_dir / filename))

        if show:
            # Display the figures
            plt.tight_layout()
            plt.pause(0.1)
            plt.close()

        return

        color_mat = []
        male_percentage_mat = []
        for row in range(self.height):
            color_row = []
            male_percentage_row = []
            for col in range(self.width):
                adult_cnt, larval_cnt, male_cnt, female_cnt = (
                    self.lamprey_world.describe(row, col)
                )
                color_row.append(
                    (
                        larval_cnt,
                        male_cnt,
                        female_cnt,
                    )
                )
                if adult_cnt == 0:
                    male_percentage_row.append(1)
                else:
                    male_percentage_row.append(male_cnt / adult_cnt)
            color_mat.append(color_row)
            male_percentage_mat.append(male_percentage_row)

        # scale
        color_mat = np.array(color_mat)
        scale_larval = 0
        scale_male = 0
        scale_female = 0

        try:
            scale_larval = np.max(color_mat[:, :, 0])
        except:
            pass

        try:
            scale_male = np.max(color_mat[:, :, 1])
        except:
            pass

        try:
            scale_female = np.max(color_mat[:, :, 2])
        except:
            pass

        # prevent scale_factor = inf
        scale_larval = max(1, scale_larval)
        scale_male = max(1, scale_male)
        scale_female = max(1, scale_female)

        print(
            f"scale_larval: {scale_larval}, scale_male: {scale_male}, scale_female: {scale_female}"
        )
        color_mat = np.array(
            [
                [
                    (
                        int(255 * a / scale_larval),
                        int(255 * b / scale_male),
                        int(255 * c / scale_female),
                    )
                    for (a, b, c) in row
                ]
                for row in color_mat
            ]
        ).astype(np.uint8)

        # add legends
        self.axes[0][0].imshow(
            color_mat,
            cmap="viridis",
        )
        self.axes[0][0].set_title("Lamprey World")
        self.axes[0][0].set_axis_off()
        # self.axes[0][0].legend(loc="upper right")

        self.axes[0][1].imshow(male_percentage_mat, cmap="binary")
        self.axes[0][1].set_title("MP")
        self.axes[0][1].set_axis_off()

        self.axes[1][0].imshow(self.prey_world.matrix, cmap="PuBu")
        self.axes[1][0].set_title("Prey World")
        self.axes[1][0].set_axis_off()

        self.axes[1][1].imshow(self.predator_world.matrix, cmap="PuBu")
        self.axes[1][1].set_title("Predator World")
        self.axes[1][1].set_axis_off()

        # plt.pause(0.01)
        # plt.show()

        # save every frame to a folder
        self.fig.savefig(self.output_dir / f"{self.iter}.png")

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
