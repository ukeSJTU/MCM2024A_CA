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

from utils import Timer
from species import LampreySpecies, PreySpecies, PredatorSpecies
from world import LampreyWorld, PreyWorld, PredatorWorld

K = 122


class Ecosystem:
    def __init__(
        self,
        lamprey_world: LampreyWorld,
        prey_world: PreyWorld,
        predator_world: PredatorWorld,
        output_dir: Path = Path("./output") / str(int(time.time())),
    ):
        self.lamprey_world = lamprey_world
        self.prey_world = prey_world
        self.predator_world = predator_world
        self.output_dir = output_dir  # the directory to save the output

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

        self.iter = 0

    def step(self):
        # every step is one year in the real world.
        # there are multiple rules for the lampreys to consume food, reproduce and die
        # we assume that every lamprey lives 7 years. The first 4 years are larval lampreys, the last 3 years are adult lampreys
        # 1. every larval lamprey turns into adults in 4 years. Their gender is determined based on the amount of food in the cell
        # 2. every adult lamprey consumes 0.67 food in the cell they are in
        # 3. every adult lamprey has a chance to reproduce with another adult lamprey in the same cell. The number of larval lampreys they produce is k(122) * MP * (1-MP) * A, where MP is male percentage and A is the number of food in the cell.
        # 4. every adult lamprey has a chance to die. The probability of dying is 0.02
        # 5. every larval lamprey has a chance to die. The probability of dying is 0.39 (1 - 61%)
        # 6. the mated lampreys will die after they reproduce
        # 7. the prey of lampreys will reproduce every year. The number of prey they reproduce is 2 times the number of lampreys they eat

        # every year, adults lampreys consume food:

        # print(self.lamprey_world)
        # increase the age of every lamprey
        for row in range(self.height):
            for col in range(self.width):
                for age in range(7, 0, -1):
                    if age != 5:
                        # print("Before:", self.lamprey_world[row][col][age])
                        self.lamprey_world[row][col][age] = copy.deepcopy(
                            self.lamprey_world[row][col][age - 1]
                        )
                        # print("After", self.lamprey_world[row][col][age])

                    # 4-year-old becomes adult
                    else:
                        """
                        如果周围环境中平均每年有≥245条(365*0.67)salmonid，认为⻝物丰富，雄
                        性占⽐偏向56%。平均每年＜245条(365*0.67)salmonid，认为⻝物短缺，雄性占⽐偏向78%
                        """
                        if self.prey_world[row][col] >= 245:
                            sex_ratio = 0.56 * random.uniform(0.95, 1.05)
                        else:
                            sex_ratio = 0.78 * random.uniform(0.95, 1.05)
                        self.lamprey_world[row][col][5][0] = int(
                            self.lamprey_world[row][col][4] * sex_ratio
                        )
                        self.lamprey_world[row][col][5][1] = int(
                            self.lamprey_world[row][col][4] * (1 - sex_ratio)
                        )

        # print(self.lamprey_world)
        self.prey_world[row][col] = int(
            self.prey_world[row][col]
            - (
                self.lamprey_world[row][col][5][0]
                + self.lamprey_world[row][col][5][1]
                + self.lamprey_world[row][col][6][0]
                + self.lamprey_world[row][col][6][1]
                + self.lamprey_world[row][col][7][0]
                + self.lamprey_world[row][col][7][1]
            )
            * 0.67
            * 300
            * random.uniform(0.95, 1.05)
        )

        # print(self.lamprey_world)
        # reproduce
        for row in range(self.height):
            for col in range(self.width):

                # TODO better way to calc sex_ratio or MP
                adult_cnt, _, male_cnt, female_cnt = self.lamprey_world.describe(
                    row=row, col=col
                )

                if adult_cnt == 0:
                    continue

                MP = male_cnt / adult_cnt

                s = int(K * MP * (1 - MP) * self.prey_world[row][col])
                self.lamprey_world[row][col][0] += s

                # then we proportionally minus the number of adult lampreys fromage over 5, because mated lampreys die after reproduce

                for age in [5, 6, 7]:
                    for sex in [0, 1]:
                        self.lamprey_world[row][col][age][sex] = int(
                            0.4
                            * self.lamprey_world[row][col][age][
                                sex
                            ]  #! 0.4 is randomly picked
                        )

        # print(self.lamprey_world)
        # die
        for row in range(self.height):
            for col in range(self.width):
                for age in range(7, 0, -1):
                    # adult has a death rate of 40% and larval death rate is 60%
                    if age <= 4:
                        self.lamprey_world[row][col][age] = int(
                            self.lamprey_world[row][col][age]
                            * (1 - self.lamprey_world.larval_death_rate)
                        )
                    else:
                        self.lamprey_world[row][col][age][0] = int(
                            self.lamprey_world[row][col][age][0]
                            * (1 - self.lamprey_world.male_death_rate)
                        )

                        self.lamprey_world[row][col][age][1] = int(
                            self.lamprey_world[row][col][age][1]
                            * (1 - self.lamprey_world.female_death_rate)
                        )

        self.iter += 1
        print(f"Iteration {self.iter}")
        print(self.lamprey_world)

    def show(self):
        # lamprey_fig, color_mat = self.lamprey_world.show()
        # prey_fig, prey_mat = self.prey_world.show()

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
