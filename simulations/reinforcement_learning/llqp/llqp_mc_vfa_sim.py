import numpy as np
import simpy
from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.llqp.llqp_mc_vfa import LLQP_MC_VFA
from simulations import *

theta = np.zeros(NUMBER_OF_USERS ** 2)
gamma = 0.5
epochs = 100
alpha = 0.001
epsilon = 0.1
policy_name = "LLQP_MC_VFA_NU{}_GI{}_TRSD{}_SIM{}".format(NUMBER_OF_USERS, GENERATION_INTERVAL, SEED, SIM_TIME)

for i in range(epochs):
    env = simpy.Environment()

    policy_train = LLQP_MC_VFA(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, theta, epsilon, gamma, alpha)

    start_event = acquisition_process(env, policy_train, SEED, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=SIM_TIME)

    policy_train.update_theta()

epsilon = 0.0

env = simpy.Environment()

file_policy = create_files("{}.csv".format(policy_name))

policy = LLQP_MC_VFA(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, theta, epsilon, gamma, alpha)

start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)