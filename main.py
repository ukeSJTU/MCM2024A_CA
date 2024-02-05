import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import random
import time
from pathlib import Path
import pandas as pd
import copy
import math
from abc import abstractmethod, ABC
import sys
from tqdm import tqdm

from utils import Calendar
from species import LampreySpecies, PreySpecies, PredatorSpecies
from world import LampreyWorld, PreyWorld, PredatorWorld, Terrain
from ecosystem import Ecosystem

import warnings

# Convert specific warnings to exceptions
# warnings.filterwarnings("error", category=RuntimeWarning)
# filter RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.ion()
plt.tight_layout()

# Define the file where you want to redirect the output
output_file = open("output.txt", "w")

# Redirect the standard output (console) to the file
sys.stdout = output_file


# To restore the console output, you can reset sys.stdout to its original value
# sys.stdout = sys.__stdout__


random.seed(42)

t = time.time()
output_dir = Path("./output") / str(int(t))
output_dir.mkdir(parents=True, exist_ok=True)


# define a base class species
# the species class should handle the basic attributes of a species, such as
# the number of the species, the reproduction rate, the death rate, the prey rate,
# the migration process
# the species class should also handle the basic methods of a species, such as
# the reproduction process, the death process, the migration process
# the species class should also handle the basic interactions between species, such as
# the predation process, the competition process
# the species class should also handle the basic interactions between the species and the environment, such as


def main():
    WORLD_WIDTH = 100
    WORLD_HEIGHT = 100

    MIN_PREY_NUM = 100
    MAX_PREY_NUM = 400

    MIN_PREDATOR_NUM = 20
    MAX_PREDATOR_NUM = 40

    PREY_BORN_RATE = 0.2
    PREY_DEATH_RATE = 0.05
    PREY_PREY_RATE = 0.1

    PREDATOR_BORN_RATE = 0.1
    PREDATOR_DEATH_RATE = 0.091
    PREDATOR_PREY_RATE = 0.1

    N_ITER = 240

    POOLING = "max"  # or "mean", the pooling strategy for ecosystem visualization
    POOL_WIDTH_SIZE = 10  # 0.1 * WORLD_WIDTH
    POOL_HEIGHT_SIZE = 10  # 0.1 * WORLD_HEIGHT

    # Create the lamprey world
    init_lamprey = LampreySpecies()
    init_lamprey[0] = 100
    init_lamprey[1] = 120
    init_lamprey[2] = 140
    init_lamprey[3] = 80
    init_lamprey[4] = 160
    init_lamprey[5] = [50, 30]
    init_lamprey[6] = [18, 23]
    init_lamprey[7] = [20, 40]

    # {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: [40, 40], 6: [20, 20], 7: [0, 0]}
    lamprey_world = LampreyWorld(
        init_value=init_lamprey,
        init_sex_ratio=0.5,
        width=WORLD_WIDTH,
        height=WORLD_HEIGHT,
    )
    # lamprey_world = LampreyWorld(0, 100, 100, init_lamprey)
    # print(lamprey_world)

    # Create the food world
    prey_world = PreyWorld(
        init_value_range=(MIN_PREY_NUM, MAX_PREY_NUM),
        width=WORLD_WIDTH,
        height=WORLD_HEIGHT,
        born_rate=PREY_BORN_RATE,
        death_rate=PREY_DEATH_RATE,
        prey_rate=PREY_PREY_RATE,
    )
    # print(prey_world)

    # Create the predator world
    predator_world = PredatorWorld(
        init_value_range=(MIN_PREDATOR_NUM, MAX_PREDATOR_NUM),
        width=WORLD_WIDTH,
        height=WORLD_HEIGHT,
        born_rate=PREDATOR_BORN_RATE,
        death_rate=PREDATOR_DEATH_RATE,
        prey_rate=PREDATOR_PREY_RATE,
    )

    terrain = Terrain(
        width=WORLD_WIDTH,
        height=WORLD_HEIGHT,
    )

    ecosystem = Ecosystem(
        lamprey_world=lamprey_world,
        prey_world=prey_world,
        predator_world=predator_world,
        terrain=terrain,
        output_dir=output_dir,
        pool_size=(POOL_WIDTH_SIZE, POOL_HEIGHT_SIZE),
        pool_method=POOLING,
        plot_iter=6,
    )

    n_iter = N_ITER
    step_time = 0
    vis_time = 0

    for i in tqdm(range(n_iter)):
        a = time.time()
        ecosystem.step()
        b = time.time()
        step_time += b - a
        # ecosystem.visualize(save=True, show=True)
        c = time.time()
        vis_time += c - b

    print(step_time)
    print(vis_time)

    ecosystem.save_metrics()

    print(ecosystem.rulesets)

    info_df = pd.DataFrame(ecosystem.info_dict)
    info_df.to_csv(output_dir / f"{str(int(t))}_info.csv", index=False)


if __name__ == "__main__":
    main()
    output_file.close()
