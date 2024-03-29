from typing import Callable, List, Union, TYPE_CHECKING
import random
import math
import copy
from utils import softmax, timer, calculate_male_percentage
from species import LampreySpecies, PreySpecies

if TYPE_CHECKING:
    from ecosystem import Ecosystem


class Rule:
    def __init__(self, condition: Callable, action: Callable, description: str = None):
        """An atomic rule

        Args:
            condition (Callable): when true, apply the action
            action (Callable): action to take when condition is met
        """
        self.condition = condition
        self.action = timer(action)

        if description is None:
            self.description = action.__name__
        self.description = description

    def get_time(self):
        return self.action.total_time


class RuleSet:
    def __init__(self, rules: List[Rule]):
        self.rules = rules
        self.time = [0 for _ in range(len(rules))]

    def apply(self, ecosystem: "Ecosystem"):
        for i, rule in enumerate(self.rules):
            if rule.condition(ecosystem):
                rule.action(ecosystem)
                self.time[i] += rule.get_time()

    def __str__(self):
        s = "\n".join(
            [
                f"Rule {i}: {self.time[i]}s | {self.rules[i].description}"
                for i in range(len(self.rules))
            ]
        )
        return s


def condition_which_month(n: Union[int, List[int]]) -> Callable[["Ecosystem"], bool]:
    """
    Returns a condition function that checks if the current month matches `n`, where `n` can be an integer or a list of integers.

    Args:
        n (Union[int, List[int]]): The month(s) to check against.

    Returns:
        Callable[[Ecosystem], bool]: A function that takes an Ecosystem instance and returns True if the current month matches `n`.
    """

    def condition(ecosystem: "Ecosystem") -> bool:
        current_month = ecosystem.calendar.get_month()
        if isinstance(n, int):
            return current_month == n
        elif isinstance(n, list):
            return current_month in n
        else:
            raise ValueError("n should be either an int or a list of int")

    return condition


def action_lamprey_increase_age(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            for age in range(7, 0, -1):
                if age != 5:
                    # print("Before:", ecosystem.lamprey_world[row][col][age])
                    # increase the age of every lamprey
                    ecosystem.lamprey_world[row][col][age] = copy.deepcopy(
                        ecosystem.lamprey_world[row][col][age - 1]
                    )
                    # print("After", ecosystem.lamprey_world[row][col][age])

                # 4-year-old becomes adult
                else:
                    """
                    如果周围环境中平均每年有≥245条(365*0.67)salmonid，认为⻝物丰富，雄
                    性占⽐偏向56%。平均每年＜245条(365*0.67)salmonid，认为⻝物短缺，雄性占⽐偏向78%
                    """
                    male_percentage = calculate_male_percentage(
                        value=ecosystem.prey_world[row][col].content, use_random=True
                    )
                    # print(f"MP: {male_percentage} for {row}, {col}")
                    # if ecosystem.prey_world[row][col] >= 245:
                    #     sex_ratio = 0.56 * random.uniform(0.95, 1.05)
                    # else:
                    #     sex_ratio = 0.78 * random.uniform(0.95, 1.05)
                    ecosystem.lamprey_world[row][col][5][0] = math.ceil(
                        ecosystem.lamprey_world[row][col][4] * male_percentage
                    )
                    ecosystem.lamprey_world[row][col][5][1] = math.ceil(
                        ecosystem.lamprey_world[row][col][4] * (1 - male_percentage)
                    )
            ecosystem.lamprey_world[row][col][0] = 0
    ecosystem.debug(1)


def action_lamprey_spawn_and_die(ecosystem: "Ecosystem"):
    K = 48
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):

            # TODO better way to calc sex_ratio or MP
            adult_cnt, _, male_cnt, female_cnt = ecosystem.lamprey_world.describe(
                row=row, col=col
            )

            if adult_cnt == 0:
                continue

            MP = male_cnt / adult_cnt

            s = int(
                ecosystem.prey_world[row][col]
                * K
                * MP
                * (1 - MP)
                * adult_cnt
                * adult_cnt
                * 0.000001
            )
            # put the ecosystem.prey_world[x][y] at the front of the equation
            # becuase we only implement * operator for PreySpecies * int or PreySpecies * float
            print(f"MP: {MP}, s: {s}, prey: {ecosystem.prey_world[row][col]}")
            print(f"Before: {ecosystem.lamprey_world[row][col][0]}")
            ecosystem.lamprey_world[row][col][0] = (
                ecosystem.lamprey_world[row][col][0] + s
            )
            print(f"After: {ecosystem.lamprey_world[row][col][0]}")

            # then we proportionally minus the number of adult lampreys fromage over 5, because mated lampreys die after reproduce

            for age in [5, 6, 7]:
                for sex in [0, 1]:
                    ecosystem.lamprey_world[row][col][age][sex] = math.ceil(
                        0.9
                        * ecosystem.lamprey_world[row][col][age][
                            sex
                        ]  #! 0.4 is randomly picked
                    )
    ecosystem.debug(2)


def action_predation(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            ecosystem.prey_world[row][col] = PreySpecies(
                content=ecosystem.prey_world[row][col]
                - (
                    ecosystem.lamprey_world[row][col][5][0]
                    + ecosystem.lamprey_world[row][col][5][1]
                    + ecosystem.lamprey_world[row][col][6][0]
                    + ecosystem.lamprey_world[row][col][6][1]
                    + ecosystem.lamprey_world[row][col][7][0]
                    + ecosystem.lamprey_world[row][col][7][1]
                )
                * 0.001  # ecosystem.lamprey_world.prey_rate
                * ecosystem.calendar.get_days()
                # * random.uniform(0.95, 1.05)
            )

            # if all the food in this cell is gone, the lampreys die immediately
            if ecosystem.prey_world[row][col] < 0:
                ecosystem.prey_world[row][col] = PreySpecies(content=0)
                ecosystem.lamprey_world[row][col] = LampreySpecies()

    ecosystem.debug(4)


def action_lamprey_predated(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            for age in [5, 6, 7]:
                for sex in [0, 1]:
                    # here we multiply 0.16 to because we assume that the predator prey evenly for lampreys of different age or sex
                    n_lamprey_preyed_by_predator = int(
                        ecosystem.predator_world.prey_rate
                        * ecosystem.predator_world[row][col]
                        * 0.16
                    )
                    # print(n_lamprey_preyed_by_predator)
                    ecosystem.lamprey_world[row][col][age][sex] = int(
                        ecosystem.lamprey_world[row][col][age][sex]
                        - n_lamprey_preyed_by_predator
                    )
    ecosystem.debug(5)


# Rule 6: every month, lampreys may die
def action_lamprey_death(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            for age in range(7, -1, -1):
                # adult has a death rate of 40% and larval death rate is 60%
                if age <= 4:
                    ecosystem.lamprey_world[row][col][age] = math.ceil(
                        ecosystem.lamprey_world[row][col][age]
                        * (1 - ecosystem.lamprey_world.larval_death_rate)
                    )
                else:
                    ecosystem.lamprey_world[row][col][age][0] = math.ceil(
                        ecosystem.lamprey_world[row][col][age][0]
                        * (1 - ecosystem.lamprey_world.male_death_rate)
                    )

                    ecosystem.lamprey_world[row][col][age][1] = math.ceil(
                        ecosystem.lamprey_world[row][col][age][1]
                        * (1 - ecosystem.lamprey_world.female_death_rate)
                    )
    ecosystem.debug(6)


# Rule 7: every month, the prey world reproduces and dies
def action_prey_reproduce(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            ecosystem.prey_world[row][col].born()
            ecosystem.prey_world[row][col].die()
    ecosystem.debug(7)


def action_predator_reproduce(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            ecosystem.predator_world[row][col].born()
            ecosystem.predator_world[row][col].die()
    ecosystem.debug(8)


def action_prey_migration(ecosystem: "Ecosystem"):
    # the prey has a higher prob to move to neighrbor cells with less adult lampreys
    for row in range(1, ecosystem.height - 1, 1):
        for col in range(1, ecosystem.width - 1, 1):
            neighbors = [
                (row - 1, col - 1),
                (row - 1, col),
                (row - 1, col + 1),
                (row, col - 1),
                (row, col + 1),
                (row + 1, col - 1),
                (row + 1, col),
                (row + 1, col + 1),
            ]
            probs = []
            for neighbor in neighbors:
                """if (
                    0 <= neighbor[0] < ecosystem.height
                    and 0 <= neighbor[1] < ecosystem.width
                ):
                """
                n_row, n_col = neighbor
                cur_adult_cnt, _, _, _ = ecosystem.lamprey_world.describe(
                    row=row, col=col
                )
                neighbor_adult_cnt, _, _, _ = ecosystem.lamprey_world.describe(
                    row=n_row, col=n_col
                )

                prob = cur_adult_cnt - neighbor_adult_cnt
                probs.append(prob)

            # softamx to get the prob
            probs = list(softmax(probs))
            migration_rate = 0.1  # TODO the migration rate should be controled by the corresponding species
            for i, neighbor in enumerate(neighbors):
                n_row, n_col = neighbor
                ecosystem.prey_world[n_row][n_col] = ecosystem.prey_world[n_row][
                    n_col
                ] + int(ecosystem.prey_world[row][col] * probs[i] * migration_rate)
                ecosystem.prey_world[row][col] = ecosystem.prey_world[row][col] - int(
                    ecosystem.prey_world[row][col] * probs[i] * migration_rate
                )

    ecosystem.debug(9)


def action_predator_migration(ecosystem: "Ecosystem"):
    # the predator has a higher prob to move to neighbor cells with more adult lampreys
    for row in range(1, ecosystem.height - 1, 1):
        for col in range(1, ecosystem.width - 1, 1):
            neighbors = [
                (row - 1, col - 1),
                (row - 1, col),
                (row - 1, col + 1),
                (row, col - 1),
                (row, col + 1),
                (row + 1, col - 1),
                (row + 1, col),
                (row + 1, col + 1),
            ]
            probs = []
            for neighbor in neighbors:
                n_row, n_col = neighbor
                cur_adult_cnt, _, _, _ = ecosystem.lamprey_world.describe(
                    row=row, col=col
                )
                neighbor_adult_cnt, _, _, _ = ecosystem.lamprey_world.describe(
                    row=n_row, col=n_col
                )
                prob = neighbor_adult_cnt - cur_adult_cnt
                probs.append(prob)

            # softamx to get the prob
            probs = list(softmax(probs))
            migration_rate = 0.1
            for i, neighbor in enumerate(neighbors):
                n_row, n_col = neighbor
                ecosystem.prey_world[n_row][n_col] = ecosystem.prey_world[n_row][
                    n_col
                ] + int(ecosystem.prey_world[row][col] * probs[i] * migration_rate)
                ecosystem.prey_world[row][col] = ecosystem.prey_world[row][col] - int(
                    ecosystem.prey_world[row][col] * probs[i] * migration_rate
                )

    ecosystem.debug(10)


def action_larval_lamprey_migration(ecosystem: "Ecosystem"):
    # the larval lampreys migrate to the cells with more food
    pass


def action_adult_lamprey_migration(ecosystem: "Ecosystem"):
    # the adult lampreys migrate to the cells with more food(prey)
    for age in [5, 6, 7]:
        for row in range(1, ecosystem.height - 1, 1):
            for col in range(1, ecosystem.width - 1, 1):
                neighbors = [
                    (row - 1, col - 1),
                    (row - 1, col),
                    (row - 1, col + 1),
                    (row, col - 1),
                    (row, col + 1),
                    (row + 1, col - 1),
                    (row + 1, col),
                    (row + 1, col + 1),
                ]
                probs = []
                for neighbor in neighbors:
                    n_row, n_col = neighbor

                    cur_prey_cnt = ecosystem.prey_world[row][col].content
                    neighbor_prey_cnt = ecosystem.prey_world[n_row][n_col].content

                    prob = neighbor_prey_cnt - cur_prey_cnt
                    probs.append(prob)

                # softamx to get the prob
                probs = list(softmax(probs))
                migration_rate = 0.1
                for i, neighbor in enumerate(neighbors):
                    n_row, n_col = neighbor
                    for sex in [0, 1]:
                        n_migration = int(
                            ecosystem.lamprey_world[row][col][age][sex]
                            * probs[i]
                            * migration_rate
                        )
                        ecosystem.lamprey_world[n_row][n_col][age][sex] += n_migration

                        ecosystem.lamprey_world[row][col][age][sex] -= n_migration


def action_lamprey_increase_age_with_even_sex_ratio(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            for age in range(7, 0, -1):
                if age != 5:
                    # print("Before:", ecosystem.lamprey_world[row][col][age])
                    # increase the age of every lamprey
                    ecosystem.lamprey_world[row][col][age] = copy.deepcopy(
                        ecosystem.lamprey_world[row][col][age - 1]
                    )
                    # print("After", ecosystem.lamprey_world[row][col][age])

                # 4-year-old becomes adult
                else:
                    male_percentage = 0.5
                    ecosystem.lamprey_world[row][col][5][0] = int(
                        ecosystem.lamprey_world[row][col][4] * male_percentage
                    )
                    ecosystem.lamprey_world[row][col][5][1] = int(
                        ecosystem.lamprey_world[row][col][4] * (1 - male_percentage)
                    )
    ecosystem.debug(1)


# Rule 1: when month is 3, every lamprey grows up by 1 year and the 4-year-old larval lampreys grow into adult lampreys
# PS: we assume that every lamprey lives 7 years. The first 4 years are larval lampreys, the last 3 years are adult lampreys
#     here, we handle the case of metamorphosis
rule_grow_older = Rule(
    condition=condition_which_month(3),
    action=action_lamprey_increase_age,
    description="All lampreys grow older",
)

# Rule 2: when month is between 3-7, adult lampreys spawn and die
rule_lamprey_spawn_and_die = Rule(
    condition=condition_which_month([5]),
    action=action_lamprey_spawn_and_die,
    description="Adult lampreys spawn and die",
)

# Rule 4: every month, adult lampreys prey from the prey world
# every month, adult lampreys consume food
rule_predation = Rule(
    condition=condition_which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    action=action_predation,
    description="Adult lampreys prey onto the prey world",
)

rule_larval_lamprey_migration = Rule(
    condition=condition_which_month([1, 2, 3, 6, 7, 8, 9, 10, 11, 12]),
    action=action_larval_lamprey_migration,
    description="Larval lampreys migrate",
)

rule_adult_lamprey_migration = Rule(
    condition=condition_which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    action=action_adult_lamprey_migration,
    description="Adult lampreys migrate",
)

rule_lamprey_death = Rule(
    condition=condition_which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    action=action_lamprey_death,
    description="All lampreys die",
)

# Rule 5: every month, adult lampreys may be eaten by predator world
rule_lamprey_predated = Rule(
    condition=condition_which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    action=action_lamprey_predated,
    description="Adult lampreys may be preyed by predator world",
)

rule_predator_reproduce = Rule(
    condition=condition_which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    action=action_predator_reproduce,
    description="Predator world reproduces",
)

rule_prey_migration = Rule(
    condition=condition_which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    action=action_prey_migration,
    description="Prey world migrates",
)

rule_predator_migration = Rule(
    condition=condition_which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    action=action_predator_migration,
    description="Predator world migrates",
)

rule_prey_reproduce = Rule(
    condition=condition_which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
    action=action_prey_reproduce,
    description="Prey world reproduces",
)

rule_grow_older_even = Rule(
    condition=condition_which_month(3),
    action=action_lamprey_increase_age_with_even_sex_ratio,
    description="All lampreys grow older and the sex ratio is 1:1",
)

rulesets = RuleSet(
    [
        rule_grow_older,
        rule_lamprey_spawn_and_die,
        rule_predation,
        rule_lamprey_predated,
        rule_lamprey_death,
        rule_prey_reproduce,
        rule_predator_reproduce,
        rule_prey_migration,
        rule_predator_migration,
        rule_adult_lamprey_migration,
    ]
)
