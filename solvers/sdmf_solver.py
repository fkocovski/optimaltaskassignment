from gurobipy import *


def sdmf(a, p, w, batch_queue_length, number_of_users):
    """
SDMF solver.
    :param a: when user i will be free.
    :param p: service times for task j by user i.
    :param w: how long has task j waited in the global queue.
    :param batch_queue_length: length of the batch queue when the method is called.
    :param number_of_users: number of users.
    :return: list of tasks to users assignment and its minimizing goal.
    """
    model = Model()
    model.setParam("OutputFlag", False)

    # xijk
    x = [[[model.addVar(vtype=GRB.BINARY, name="x{}{}{}".format(i, j, k)) for k in range(batch_queue_length)] for
          j in range(batch_queue_length)] for i in range(number_of_users)]

    # z_max
    z_max = model.addVar(vtype=GRB.CONTINUOUS, name="z_max")

    model.setObjective(z_max, GRB.MINIMIZE)

    for j in range(batch_queue_length):
        model.addConstr(
            quicksum(x[i][j][k] for k in range(batch_queue_length) for i in range(number_of_users)) == 1)

    for i in range(number_of_users):
        for k in range(batch_queue_length):
            if k > 0:
                model.addConstr(quicksum(x[i][j][k] for j in range(batch_queue_length)) <= quicksum(
                    x[i][j][k - 1] for j in range(batch_queue_length)))
            else:
                model.addConstr(quicksum(x[i][j][k] for j in range(batch_queue_length)) <= 1)

    for i in range(number_of_users):
        for k in range(batch_queue_length):
            model.addConstr(a[i] + quicksum(
                (p[i][j] + w[j] * (t == k)) * x[i][j][t] for j in range(batch_queue_length) for t in
                range(k + 1)) <= z_max)

    model.optimize()

    tasks_to_users_assignment = [None] * batch_queue_length

    for k in range(batch_queue_length):
        for j in range(batch_queue_length):
            for i in range(number_of_users):
                if x[i][j][k].x > 0.5:
                    tasks_to_users_assignment[j] = i
                    if abs(x[i][j][k].x - 1) > 1e-6:
                        print("Threshold violated: {}, i: {}, j: {}, k: {}".format(x[i][j][k].x, i, j, k))

    return tasks_to_users_assignment, z_max.x
