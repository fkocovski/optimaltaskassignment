import numpy as np
import simpy
from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.llqp.llqp_mc import LLQP_MC
from simulations import *

q_table = np.zeros((100, 100, NUMBER_OF_USERS))
epsilon = 0.3
gamma = 0.5
epochs = 10
policy_name = "LLQP_MC_NU{}_GI{}_SIM{}".format(NUMBER_OF_USERS, GENERATION_INTERVAL, SIM_TIME)

for i in range(epochs):
    env = simpy.Environment()

    file_policy = create_files("{}.csv".format(policy_name))

    policy = LLQP_MC(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, q_table, epsilon, gamma)

    start_event = acquisition_process(env, policy, SEED, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=SIM_TIME)

    new_q_table = LLQP_MC.update_q_table(policy)
    q_table = new_q_table

    file_policy.close()

    calculate_statistics(file_policy.name, outfile=True)
    evolution(file_policy.name, outfile=True)