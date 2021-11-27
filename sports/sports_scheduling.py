# coding:utf-8
"""Make the Binary Quadratic Model for sports scheduling problem.

Definitions and comments in this code are based on the following paper.

Title:
  SOLVING LARGE BREAK MINIMIZATION PROBLEMS IN A MIRRORED DOUBLE ROUND-ROBIN TOURNAMENT USING QUANTUM ANNEALING.
  (https://arxiv.org/pdf/2110.07239.pdf)

Author:
  Michiya Kuramata (Tokyo Institute of Technology, Tokyo, Japan)
  Ryota Katsuki (NTT DATA, Tokyo, Japan)
  Nakata Kazuhide (Tokyo Institute of Technology, Tokyo, Japan)
"""
import itertools

import numpy as np
import pandas as pd


def kirkman_schedule(n):
    """make RRT timetable using Kirkman cycle method.

    Reference:
        Kirkman TP. On a problem in combinations. Cambridge and Dublin Mathematical Journal. 1847; 2: 191–204.

    Args:
        n (int): 2n is the number of teams in MDRRT.

    Returns:
        pandas.DataFrame: RRT by Kirkman cycle method.
    """
    num_teams = 2 * n
    num_circle = num_teams - 1

    left_circle = sorted(range(num_circle), reverse=False)
    left_circle = 2 * left_circle

    right_circle = sorted(range(num_circle), reverse=True)
    right_circle = right_circle[len(right_circle) - 1:] + right_circle[:len(right_circle) - 1]
    right_circle = 2 * right_circle

    schedule_list = [[] for i in range(num_teams - 1)]
    for i in range(num_teams - 1):
        circle_a = left_circle[len(left_circle) - num_teams - i + 1:len(left_circle) - i]
        circle_b = right_circle[i:i + num_teams - 1]

        schedule_list[i].append([circle_a[0], num_circle])
        for j in range(1, len(circle_a[1:])):
            tmp = sorted([circle_a[j], circle_b[j]])
            if not (tmp in schedule_list[i]):
                schedule_list[i].append(tmp)

    schedule = np.zeros((num_teams, num_teams - 1), dtype=np.int)
    for i, *partial_schedule in enumerate(schedule_list):
        for match in partial_schedule[0]:
            team_a, team_b = match
            schedule[team_a][i] = team_b
            schedule[team_b][i] = team_a

    result = pd.DataFrame(schedule)
    result.index.name = 'team'
    result.columns.name = 'timeslot'

    return result


class MDRRT:
    """Manage the target MDRRT.

    Args:
        n (int): 2n is the number of teams in MDRRT.
        schedule (pandas.DataFrame or str, optional): (kirkman) schedule defined in advance. path of csv file.

    Attributes:
        n (int): 2n is the number of teams in MDRRT.
        num_teams (int): the number of teams.
        num_slots (int): the number of timeslots in MDRRT.
        schedule (pandas.DataFrame): (kirkman) schedule defined in advance.
        k_list (list[tuple[int]]): the pair of a team and its opponent. tuples of combinations 2n_C_2.
        slot_list (list[tuple[int]]): the pair of timeslots of a team and its opponent in k_list.
        ts_list (list[list[tuple[int]]]): indices defined in (3) in paper.
        unpack_ts_list (dict[tuple[int], list[int]]): keys are (t, s) and values are (k, the order of 4 tuples).
    """

    def __init__(self, n, schedule=None):
        self.n = n
        self.num_teams = n * 2
        self.num_slots = (self.num_teams - 1) * 2
        self.schedule = schedule
        self.k_list = []
        self.slot_list = []
        self.ts_list = []
        self.unpack_ts_dict = []

        if isinstance(schedule, pd.DataFrame):
            self.schedule = schedule
            self.setup_index()

        elif isinstance(schedule, str):
            self.read_csv(schedule)

    def setup_index(self):
        """setup indices for MDRRT"""
        self.k_list = self._make_k_list()
        self.slot_list = self._make_timeslot_list()
        self.ts_list = self._make_ts_list()
        self.unpack_ts_dict = self._make_unpack_ts_dict()

    def read_csv(self, fpath):
        """setup MDRRT based on csv file."""
        schedule = pd.read_csv(fpath)
        schedule.columns.name = 'timeslot'
        schedule.index.name = 'team'
        schedule.columns = schedule.columns.astype(int)

        self.schedule = schedule
        self.setup_index()

        return schedule

    def _make_k_list(self):
        return list(itertools.combinations(range(2 * self.n), 2))

    def _make_timeslot_list(self):
        slot_list = []
        for team, opponent in self.k_list:
            slot_a, slot_b = list(self.schedule.iloc[0, :][self.schedule.iloc[team, :] == opponent].index)
            slot = (slot_a, slot_b)
            slot_list.append(slot)
        return slot_list

    def _make_ts_list(self):
        ts_list = []
        for k in range(len(self.k_list)):
            t, t_prime = self.k_list[k]
            s, s_prime = self.slot_list[k]

            ts_list.append([(t, s), (t, s_prime), (t_prime, s), (t_prime, s_prime)])

        return ts_list

    def _make_unpack_ts_dict(self):
        unpack_ts_list = list(itertools.chain.from_iterable(self.ts_list))
        unpack_ts_dict = dict()
        for i in range(len(unpack_ts_list)):
            unpack_ts_dict[unpack_ts_list[i]] = [i // 4, i % 4]
        return unpack_ts_dict


def z_to_y(mdrrt, z):
    """convert z into y according to target mdrrt.

    This method achieve the conversion of y into z.
    This is (7) in the paper.

    Args:
        mdrrt (MDRRT): target MDRRT
        z (pyqubo.array.Array): decision variable z

    Returns:
        list<list<pyqubo.Binary>>: decision variable y
    """
    n = mdrrt.n
    S = mdrrt.num_slots

    y = []
    for t in range(2 * n):
        row = []
        for s in range(S):
            # index (t, s) is on row k and line j. (3) in the paper.
            k, j = mdrrt.unpack_ts_dict[t, s]

            # the conversion of z to y. (7) in the paper
            if j in [0, 3]:
                row.append(z[k])
            else:
                row.append(1 - z[k])
        y.append(row)
    return y


def make_objective_function(mdrrt, y):
    """make objective function using pyqubo and gurobipy.

    This function returns the objective function (9) in the paper.
    You don't need to include any constraints because y is written by z.

    Args:
        mdrrt (MDRRT): target MDRRT
        y (pyqubo.array.Array): decision variable y (defined in (7))

    Returns:
        pyqubo.Add: objective function
    """
    n = mdrrt.n
    S = mdrrt.num_slots

    break_term = 0
    for t in range(2 * n):
        for s in range(S - 1):
            break_term += y[t][s] * y[t][s + 1] + (1 - y[t][s]) * (1 - y[t][s + 1])
    return break_term
