# the metrics.py file contains the functions to calculate the metrics for the ecosystem

import time
from tqdm import tqdm

# from main import main
from ecosystem import Ecosystem


class Metrics:
    def __init__(self):
        pass

    def calculate_metrics(self, ecosystem, n_iter):
        step_time = 0
        vis_time = 0

        for i in tqdm(range(n_iter)):
            a = time.time()
            ecosystem.step()
            b = time.time()
            step_time += b - a
            ecosystem.visualize(save=True, show=True)
            c = time.time()
            vis_time += c - b

        print(step_time)
        print(vis_time)
