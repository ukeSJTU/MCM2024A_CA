import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
from typing import Union, Literal, Tuple
import time
from pathlib import Path
import pandas as pd
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
from metrics import Metrics

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
        output_dir: Path,
        pool_size: Tuple[int, int],
        pool_method: Literal["max", "mean"] = "max",
        calendar: Calendar = Calendar(2000, 1),
        rulesets: rules.RuleSet = rules.rulesets,
        plot_iter: int = 12,
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

        self.metrics = Metrics(parent=self)
        self.rulesets = rulesets

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

        self.plot_iter = plot_iter

        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 5))
        self.cbars = []

        self.iter = 0
        self.calendar = calendar

        # use a dataframe to keep track of the values of each world in the ecosystem
        self.info_dict = {
            "iter": [],
            "n_lamprey": [],
            "n_prey": [],
            "n_predator": [],
            "n_adult_lamprey": [],
            "n_larval_lamprey": [],
            "n_male_lamprey": [],
            "n_female_lamprey": [],
            "0_age": [],
            "1_age": [],
            "2_age": [],
            "3_age": [],
            "4_age": [],
            "5_age": [],
            "6_age": [],
            "7_age": [],
        }

    def debug(self, n):
        """_summary_

        Args:
            n (int): rule index
        """
        print(f"Iteration {self.iter}, {self.calendar}, Rule: {n} ")
        print(self.lamprey_world)
        # print(self.prey_world)
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

        self.rulesets.apply(ecosystem=self)

        print(self.metrics.get_metrics("lamprey_density"))
        print(self.metrics.get_metrics("prey_density"))
        print(self.metrics.get_metrics("predator_density"))

        self.iter += 1
        self.calendar += 1

        if self.iter % self.plot_iter == 0:
            self.visualize(save=True, show=False)

        adult_lamprey = 0
        larval_lamprey = 0
        male_lamprey = 0
        female_lamprey = 0

        age_0_lamprey = 0
        age_1_lamprey = 0
        age_2_lamprey = 0
        age_3_lamprey = 0
        age_4_lamprey = 0
        age_5_lamprey = 0
        age_6_lamprey = 0
        age_7_lamprey = 0

        for row in range(self.width):
            for col in range(self.height):
                n_adult, n_larval, n_male, n_female = self.lamprey_world.describe(
                    row, col
                )
                adult_lamprey += n_adult
                larval_lamprey += n_larval
                male_lamprey += n_male
                female_lamprey += n_female

                age_0_lamprey += self.lamprey_world[row][col][0]
                age_1_lamprey += self.lamprey_world[row][col][1]
                age_2_lamprey += self.lamprey_world[row][col][2]
                age_3_lamprey += self.lamprey_world[row][col][3]
                age_4_lamprey += self.lamprey_world[row][col][4]
                age_5_lamprey += (
                    self.lamprey_world[row][col][5][0]
                    + self.lamprey_world[row][col][5][1]
                )
                age_6_lamprey += (
                    self.lamprey_world[row][col][6][0]
                    + self.lamprey_world[row][col][6][1]
                )
                age_7_lamprey += (
                    self.lamprey_world[row][col][7][0]
                    + self.lamprey_world[row][col][7][1]
                )

        n_prey = 0
        for row in range(self.width):
            for col in range(self.height):
                n_prey += float(self.prey_world[row][col])

        n_predator = 0
        for row in range(self.width):
            for col in range(self.height):
                n_predator += float(self.predator_world[row][col])

        self.info_dict["iter"].append(self.iter)
        self.info_dict["n_lamprey"].append(adult_lamprey + larval_lamprey)
        self.info_dict["n_prey"].append(n_prey)
        self.info_dict["n_predator"].append(n_predator)
        self.info_dict["n_adult_lamprey"].append(adult_lamprey)
        self.info_dict["n_larval_lamprey"].append(larval_lamprey)
        self.info_dict["n_male_lamprey"].append(male_lamprey)
        self.info_dict["n_female_lamprey"].append(female_lamprey)
        self.info_dict["0_age"].append(age_0_lamprey)
        self.info_dict["1_age"].append(age_1_lamprey)
        self.info_dict["2_age"].append(age_2_lamprey)
        self.info_dict["3_age"].append(age_3_lamprey)
        self.info_dict["4_age"].append(age_4_lamprey)
        self.info_dict["5_age"].append(age_5_lamprey)
        self.info_dict["6_age"].append(age_6_lamprey)
        self.info_dict["7_age"].append(age_7_lamprey)

    def visualize(
        self, save: bool = False, show: bool = True, filename: Union[str, Path] = None
    ):
        # lamprey_fig, color_mat = self.lamprey_world.show()
        # prey_fig, prey_mat = self.prey_world.show()
        if save == False and show == False:
            return

        if filename is None:
            vector_filename = f"{self.iter}.eps"
            bitmap_filename = f"{self.iter}.png"

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
            new_size=self.pool_size,
            method=self.pool_method,
        )
        male_percentage_mask = np.isnan(male_percentage_data_pooled)
        # male_percentage_data_pooled = self.apply_pooling(
        #     male_percentage_data, size=self.pool_size, method=self.pool_method
        # )

        larval_adult_ratio_data_pooled = resize_with_pooling(
            data=larval_adult_ratio_data,
            new_size=self.pool_size,
            method=self.pool_method,
        )
        larval_adult_ratio_mask = np.isnan(larval_adult_ratio_data_pooled)

        prey_data_pooled = resize_with_pooling(
            prey_data,
            new_size=self.pool_size,
            method=self.pool_method,
        )  # or 'mean'

        predator_data_pooled = resize_with_pooling(
            predator_data,
            new_size=self.pool_size,
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

        mp_fig, mp_ax = plt.subplots()
        sns.heatmap(
            male_percentage_data_pooled,
            ax=mp_ax,
            square=True,
            cmap="coolwarm",
            center=1,
            linewidths=1,
            linecolor="black",
            # cbar_kws={"label": "Sex Ratio (Males/Females)"},
        )
        mp_ax.set_title("Lamprey MP")

        lamprey_distribution_fig, lamprey_distribution_ax = plt.subplots()
        sns.heatmap(
            larval_adult_ratio_data_pooled,
            ax=lamprey_distribution_ax,
            square=True,
            cmap="coolwarm",
            center=1,
            linewidths=1,
            linecolor="black",
            # cbar_kws={"label": "Larval/Adult Ratio"},
        )
        lamprey_distribution_ax.set_title("Lamprey Dist")

        prey_distribution_fig, prey_distribution_ax = plt.subplots()
        sns.heatmap(
            prey_data_pooled,
            ax=prey_distribution_ax,
            square=True,
            cmap="Greens",
            linewidths=1,
            linecolor="black",
            # cbar_kws={"label": "Prey Density"},
        )
        prey_distribution_ax.set_title("Prey Dist")

        predator_distribution_fig, predator_distribution_ax = plt.subplots()
        sns.heatmap(
            predator_data_pooled,
            ax=predator_distribution_ax,
            square=True,
            cmap="Reds",
            linewidths=1,
            linecolor="black",
            # cbar_kws={"label": "Predator Density"},
        )
        predator_distribution_ax.set_title("Predator Dist")

        if save:
            mp_fig.savefig(str(self.output_lamprey_dir / ("MP" + vector_filename)))
            # self.fig.savefig(str(self.output_dir / filename))
            lamprey_distribution_fig.savefig(
                str(self.output_lamprey_dir / ("LampD" + vector_filename))
            )
            prey_distribution_fig.savefig(
                str(self.output_prey_dir / ("PreyD" + vector_filename))
            )
            predator_distribution_fig.savefig(
                str(self.output_predator_dir / ("PredD" + vector_filename))
            )

            mp_fig.savefig(str(self.output_lamprey_dir / ("MP" + bitmap_filename)))
            # self.fig.savefig(str(self.output_dir / filename))
            lamprey_distribution_fig.savefig(
                str(self.output_lamprey_dir / ("LampD" + bitmap_filename))
            )
            prey_distribution_fig.savefig(
                str(self.output_prey_dir / ("PreyD" + bitmap_filename))
            )
            predator_distribution_fig.savefig(
                str(self.output_predator_dir / ("PredD" + bitmap_filename))
            )

        if show:
            plt.pause(0.1)
        plt.close()
        return

    def save_metrics(self):
        # save the metrics to a csv file
        # and plot the metrics

        metrics = self.metrics.get_metrics()
        print(metrics)
        df = pd.DataFrame(metrics)
        df.to_csv(str(self.output_dir / "metrics.csv"), index=False)

        fig = plt.figure()

        # need to scale the data to the same scale
        metrics["lamprey_density"] = np.array(metrics["lamprey_density"]) / np.max(
            np.array(metrics["lamprey_density"])
        )
        metrics["prey_density"] = np.array(metrics["prey_density"]) / np.max(
            np.array(metrics["prey_density"])
        )
        metrics["predator_density"] = np.array(metrics["predator_density"]) / np.max(
            np.array(metrics["predator_density"])
        )

        # plot the metrics and save the fig
        plt.plot(metrics["lamprey_density"], label="Lamprey Density")
        plt.plot(metrics["prey_density"], label="Prey Density")
        plt.plot(metrics["predator_density"], label="Predator Density")
        plt.legend()
        fig.suptitle("Ecosystem Metrics")
        plt.xlabel("Iteration (Month)")
        plt.ylabel("Density")

        fig.savefig(str(self.output_dir / "metrics.png"))
        plt.show()

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
