# coding:utf-8

import gurobipy as gb

from .sports_scheduling import *


def trick_BMP(mdrrt, model_name='Trick_BMP', **kwargs):
    """Break Minimization Problem for MDRRT by our extension of Trick's method.

    Reference:
        Trick MA. A schedule-then-break approach to sports timetabling.
        In: International Conference on the Practice and Theory of Automated Timetabling. Springer; 2000. p. 242–253.


    Args:
        mdrrt (MDRRT): target MDRRT.
        model_name (str, optional): name for the problem.
        kwargs (dict, optional): parameters used in mathematical solver.

    Returns:
        gurobipy.Model: optimization model for MDRRT in gurobipy.
        list[list[gurobipy.Var]]: variables for MDRRT in gurobipy.

    """
    n = mdrrt.n
    Teams = list(range(2 * n))
    Time = list(range(4 * n - 2))
    TweenTime = list(range(4 * n - 3))
    schedule = mdrrt.schedule

    # key: (team1, teams), value: [1st scheduled timeslot, 2nd scheduled timeslot]
    slot = {}
    for i, j in list(itertools.combinations(Teams, 2)):
        slot[(i, j)] = list(schedule.iloc[i][schedule.iloc[i] == j].index)

    model = gb.Model(model_name)
    model.update()
    start = model.addVars(2 * n, vtype=gb.GRB.BINARY, name='start')
    model.update()
    tohome = model.addVars(2 * n, 4 * n - 2 - 1, vtype=gb.GRB.BINARY, name='tohome')
    model.update()
    toaway = model.addVars(2 * n, 4 * n - 2 - 1, vtype=gb.GRB.BINARY, name='toaway')
    model.update()

    def athome_generator(i, t, tohome, toaway, start):
        return start[i] + sum([tohome[i, t1] - toaway[i, t1] for t1 in range(t)])

    athome = []
    for i in Teams:
        athome_vec = []
        for t in Time:
            athome_vec.append(athome_generator(i, t, tohome, toaway, start))
        athome.append(athome_vec)

    model.addConstrs((athome[i][slot[(i, j)][0]] + athome[j][slot[(i, j)][0]] == 1 for i, j in
                      list(itertools.combinations(Teams, 2))), name='athome_Constrs_1')  # 相手か自分のホームで戦う。第1ラウンド
    model.update()
    model.addConstrs((athome[i][slot[(i, j)][1]] + athome[j][slot[(i, j)][1]] == 1 for i, j in
                      list(itertools.combinations(Teams, 2))), name='athome_Constrs_2')  # 相手か自分のホームで戦う。第2ラウンド
    model.update()
    model.addConstrs(
        (athome[i][slot[(i, j)][0]] == athome[j][slot[(i, j)][1]] for i, j in list(itertools.combinations(Teams, 2))),
        name='athome_Constrs_3')
    model.update()

    model.addConstrs((0 <= athome[i][t] for i in Teams for t in Time[1:]), name='athome_lwb')
    model.update()
    model.addConstrs((athome[i][t] <= 1 for i in Teams for t in Time[1:]), name='athome_upb')
    model.update()

    model.addConstrs((tohome[i, t] + toaway[i, t] <= 1 for i in Teams for t in TweenTime), name='tohometoaway_Constrs')
    model.update()

    model.addConstr(start[0] == 1, name='start_Constr')
    model.update()

    model.addConstrs((athome[i][t] - toaway[i, t] >= 0 for i in Teams for t in TweenTime), name='athometoaway_Constrs')
    model.update()

    model.addConstrs((athome[i][t] + tohome[i, t] <= 1 for i in Teams for t in TweenTime), name='athometohome_Constrs')
    model.update()

    for i, j, k in list(itertools.combinations(Teams, 3)):
        index = [[i, j], [j, k], [i, k]]
        slot_list = [slot[i, j][0], slot[j, k][0], slot[i, k][0]]
        slot_argsort = np.argsort(slot_list)

        first_idx = index[slot_argsort[0]]
        second_idx = index[slot_argsort[1]]
        third_idx = index[slot_argsort[2]]

        first_slot = slot_list[slot_argsort[0]]
        second_slot = slot_list[slot_argsort[1]]
        third_slot = slot_list[slot_argsort[2]]

        p = list(set(first_idx) & set(third_idx))[0]
        q = list(set(first_idx) & set(second_idx))[0]
        r = list(set(second_idx) & set(third_idx))[0]
        model.addConstr(sum([toaway[p, t] + tohome[p, t] for t in range(first_slot, third_slot)]) +
                        sum([toaway[q, t] + tohome[q, t] for t in range(first_slot, second_slot)]) +
                        sum([toaway[r, t] + tohome[r, t] for t in range(second_slot, third_slot)]) <=
                        (third_slot - first_slot) + (second_slot - first_slot) + (third_slot - second_slot) - 1,
                        name=f'triangle1_{i}_{j}_{k}')
        model.update()

    for i, j, k in list(itertools.combinations(Teams, 3)):
        index = [[i, j], [j, k], [i, k]]
        slot_list = [slot[i, j][1], slot[j, k][1], slot[i, k][1]]
        slot_argsort = np.argsort(slot_list)

        first_idx = index[slot_argsort[0]]
        second_idx = index[slot_argsort[1]]
        third_idx = index[slot_argsort[2]]

        first_slot = slot_list[slot_argsort[0]]
        second_slot = slot_list[slot_argsort[1]]
        third_slot = slot_list[slot_argsort[2]]

        p = list(set(first_idx) & set(third_idx))[0]
        q = list(set(first_idx) & set(second_idx))[0]
        r = list(set(second_idx) & set(third_idx))[0]
        model.addConstr(sum([toaway[p, t] + tohome[p, t] for t in range(first_slot, third_slot)]) +
                        sum([toaway[q, t] + tohome[q, t] for t in range(first_slot, second_slot)]) +
                        sum([toaway[r, t] + tohome[r, t] for t in range(second_slot, third_slot)]) <=
                        (third_slot - first_slot) + (second_slot - first_slot) + (third_slot - second_slot) - 1,
                        name=f'triangle2_{i}_{j}_{k}')
        model.update()

    model.setParam("MIPGapAbs", 1.99)
    model.update()

    for k, v in kwargs.items():
        model.setParam(k, v)
        model.update()

    model.setObjective(gb.quicksum([tohome[i, t] + toaway[i, t] for i in Teams for t in TweenTime]), gb.GRB.MAXIMIZE)
    model.update()

    model.optimize()
    return model, athome


def urdaneta_BMP(mdrrt, model_name='Urdaneta_BMP', **kwargs):
    """Break Minimization Problem for MDRRT by Urdaneta's method.

    References:
        Urdaneta HL, Yuan J, Siqueira AS. Alternative Integer linear and Quadratic Programming Formulations for HA- Assignment Problems.
        Proceeding Series of the Brazilian Society of Computational and Applied Mathematics. 2018;6(1).

    Args:
        mdrrt (MDRRT): target MDRRT.
        model_name (str, optional): name for the problem.
        kwargs (dict, optional): parameters used in mathematical solver.

    Returns:
        gurobipy.Model: optimization model for MDRRT in gurobipy.
        list[list[gurobipy.Var]]: variables for MDRRT in gurobipy.
    """
    n = mdrrt.n

    model = gb.Model(model_name)
    model.update()

    z = model.addVars(n * (2 * n - 1), vtype=gb.GRB.BINARY, name='z')
    model.update()

    y = z_to_y(mdrrt, z)

    gb_objective = make_objective_function(mdrrt, y)
    model.setObjective(gb_objective)
    model.update()

    for k, v in kwargs.items():
        model.setParam(k, v)
        model.update()

    model.update()
    model.optimize()
    return model, y
