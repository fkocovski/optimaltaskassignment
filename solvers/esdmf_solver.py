from gurobipy import *


def esdmf(a, p, w, batch_queue_length, number_of_users):
    """
ESDMF solver.
    :param a: when user i will be free.
    :param p: service times for task j by user i.
    :param w: how long has task j waited in the global queue.
    :param batch_queue_length: length of the batch queue when the method is called.
    :param number_of_users: number of users.
    :return: list of tasks to users assignment and its minimizing goal.
    """
    model = Model()
    model.setParam("OutputFlag", False)

    # xij
    x = [[model.addVar(vtype=GRB.BINARY, name="x{}{}".format(i, j)) for j in range(batch_queue_length)] for i in range(number_of_users)]

    # z_max
    z_max = model.addVar(vtype=GRB.CONTINUOUS, name="z_max")

    model.setObjective(z_max, GRB.MINIMIZE)

    for j in range(batch_queue_length):
        model.addConstr(
            quicksum(x[i][j] for i in range(number_of_users)) == 1)

    for i in range(number_of_users):
        for j in range(batch_queue_length):
            model.addConstr(a[i] + quicksum(
                (p[i][k] + w[k] * (k == j)) * x[i][k] for k in
                range(j + 1)) <= z_max)

    model.optimize()

    tasks_to_users_assignment = [None] * batch_queue_length

    for j in range(batch_queue_length):
        for i in range(number_of_users):
            if x[i][j].x > 0.5:
                tasks_to_users_assignment[j] = i
                if abs(x[i][j].x - 1) > 1e-6:
                    print("Threshold violated: {}, i: {}, j: {}".format(x[i][j].x, i, j))

    return tasks_to_users_assignment,z_max.x
