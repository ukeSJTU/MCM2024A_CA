from typing import Callable, List, Union, TYPE_CHECKING
import random
import copy
from species import LampreySpecies, PreySpecies

if TYPE_CHECKING:
    from ecosystem import Ecosystem


class Rule:
    def __init__(self, condition: Callable, action: Callable):
        """An atomic rule

        Args:
            condition (Callable): when true, apply the action
            action (Callable): action to take when condition is met
        """
        self.condition = condition
        self.action = action


class RuleSet:
    def __init__(self, rules: List[Rule]):
        self.rules = rules

    def apply(self, ecosystem: "Ecosystem"):
        for rule in self.rules:
            if rule.condition(ecosystem):
                rule.action(ecosystem)


def which_month(n: Union[int, List[int]]):
    """
    Returns a condition function that checks if the current month matches `n`, where `n` can be an integer or a list of integers.

    Args:
        n (Union[int, List[int]]): The month(s) to check against.

    Returns:
        Callable[[Ecosystem], bool]: A function that takes an Ecosystem instance and returns True if the current month matches `n`.
    """

    def condition(ecosystem: "Ecosystem") -> bool:
        current_month = ecosystem.timer.get_month()
        if isinstance(n, int):
            return current_month == n
        elif isinstance(n, list):
            return current_month in n
        else:
            raise ValueError("n should be either an int or a list of int")

    return condition


def action_grow_older(ecosystem: "Ecosystem"):
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
                    if ecosystem.prey_world[row][col] >= 245:
                        sex_ratio = 0.56 * random.uniform(0.95, 1.05)
                    else:
                        sex_ratio = 0.78 * random.uniform(0.95, 1.05)
                    ecosystem.lamprey_world[row][col][5][0] = int(
                        ecosystem.lamprey_world[row][col][4] * sex_ratio
                    )
                    ecosystem.lamprey_world[row][col][5][1] = int(
                        ecosystem.lamprey_world[row][col][4] * (1 - sex_ratio)
                    )
    ecosystem.debug(1)


def action_spawn_and_die(ecosystem: "Ecosystem"):
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
            )  # put the ecosystem.prey_world[x][y] at the front of the equation
            # becuase we only implement * operator for PreySpecies * int or PreySpecies * float
            # print(f"MP: {MP}, s: {s}, prey: {ecosystem.prey_world[row][col]}")
            ecosystem.lamprey_world[row][col][0] += s

            # then we proportionally minus the number of adult lampreys fromage over 5, because mated lampreys die after reproduce

            for age in [5, 6, 7]:
                for sex in [0, 1]:
                    ecosystem.lamprey_world[row][col][age][sex] = int(
                        0.4
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
                * 0.001
                * 30
                * random.uniform(0.95, 1.05)
            )

            # if all the food in this cell is gone, the lampreys die immediately
            if ecosystem.prey_world[row][col] < 0:
                ecosystem.prey_world[row][col] = PreySpecies(content=0)
                ecosystem.lamprey_world[row][col] = LampreySpecies()

    ecosystem.debug(4)


# Rule 1: when month is 3, every lamprey grows up by 1 year and the 4-year-old larval lampreys grow into adult lampreys
# PS: we assume that every lamprey lives 7 years. The first 4 years are larval lampreys, the last 3 years are adult lampreys
#     here, we handle the case of metamorphosis
rule_grow_older = Rule(which_month(3), action_grow_older)

# Rule 2: when month is between 3-7, adult lampreys spawn and die
rule_spawn_and_die = Rule(which_month([3, 4, 5, 6, 7]), action_spawn_and_die)

# Rule 4: every month, adult lampreys prey from the prey world
# every month, adult lampreys consume food
rule_predation = Rule(
    which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), action_predation
)


def action_predated(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            ecosystem.lamprey_world[row][col][5][0] = int(
                ecosystem.lamprey_world[row][col][5][0]
                * (1 - ecosystem.predator_world[row][col] * 0.001)
            )
            ecosystem.lamprey_world[row][col][5][1] = int(
                ecosystem.lamprey_world[row][col][5][1]
                * (1 - ecosystem.predator_world[row][col] * 0.001)
            )
            ecosystem.lamprey_world[row][col][6][0] = int(
                ecosystem.lamprey_world[row][col][6][0]
                * (1 - ecosystem.predator_world[row][col] * 0.001)
            )
            ecosystem.lamprey_world[row][col][6][1] = int(
                ecosystem.lamprey_world[row][col][6][1]
                * (1 - ecosystem.predator_world[row][col] * 0.001)
            )
            ecosystem.lamprey_world[row][col][7][0] = int(
                ecosystem.lamprey_world[row][col][7][0]
                * (1 - ecosystem.predator_world[row][col] * 0.001)
            )
            ecosystem.lamprey_world[row][col][7][1] = int(
                ecosystem.lamprey_world[row][col][7][1]
                * (1 - ecosystem.predator_world[row][col] * 0.001)
            )
    ecosystem.debug(5)


# Rule 5: every month, adult lampreys may be eaten by predator world
rule_predated = Rule(
    which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), action_predated
)


# Rule 6: every month, lampreys may die
def action_death(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            for age in range(7, 0, -1):
                # adult has a death rate of 40% and larval death rate is 60%
                if age <= 4:
                    ecosystem.lamprey_world[row][col][age] = int(
                        ecosystem.lamprey_world[row][col][age]
                        * (1 - ecosystem.lamprey_world.larval_death_rate)
                    )
                else:
                    ecosystem.lamprey_world[row][col][age][0] = int(
                        ecosystem.lamprey_world[row][col][age][0]
                        * (1 - ecosystem.lamprey_world.male_death_rate)
                    )

                    ecosystem.lamprey_world[row][col][age][1] = int(
                        ecosystem.lamprey_world[row][col][age][1]
                        * (1 - ecosystem.lamprey_world.female_death_rate)
                    )
    ecosystem.debug(6)


rule_death = Rule(which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), action_death)


# Rule 7: every month, the prey world reproduces and dies
def action_prey_reproduce(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            ecosystem.prey_world[row][col].born()
            ecosystem.prey_world[row][col].die()
    ecosystem.debug(7)


rule_prey_reproduce = Rule(
    which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), action_prey_reproduce
)


def action_predator_reproduce(ecosystem: "Ecosystem"):
    for row in range(ecosystem.height):
        for col in range(ecosystem.width):
            ecosystem.predator_world[row][col].born()
            ecosystem.predator_world[row][col].die()
    ecosystem.debug(8)


rule_predator_reproduce = Rule(
    which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), action_predator_reproduce
)


def action_migrate(ecosystem: "Ecosystem"):
    ecosystem.prey_world.migrate()
    ecosystem.predator_world.migrate()
    ecosystem.debug(9)


rule_migrate = Rule(
    which_month([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), action_migrate
)


rulesets = RuleSet(
    [
        rule_grow_older,
        rule_spawn_and_die,
        rule_predation,
        rule_predated,
        rule_death,
        rule_prey_reproduce,
        rule_predator_reproduce,
        rule_migrate,
    ]
)
