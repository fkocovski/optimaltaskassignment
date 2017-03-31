import numpy as np
import simpy
from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.batch.k_batch_mc_vfa import K_BATCH_MC_VFA
from simulations import *

theta = np.zeros(2 * (NUMBER_OF_USERS ** 2))
gamma = 0.5
epochs = 10
initial_alpha = 1e-2
policy_name = "{}_BATCH_MC_VFA_NU{}_GI{}_SIM{}".format(1,NUMBER_OF_USERS, GENERATION_INTERVAL, SIM_TIME)

for i in range(epochs):
    env = simpy.Environment()

    epsilon = 0.1
    # alpha_disc = initial_alpha

    # epsilon = 1 / (i + 1)
    alpha_disc = initial_alpha / (i + 1)

    # this method only works with batch size 1
    policy_train = K_BATCH_MC_VFA(env, NUMBER_OF_USERS, WORKER_VARIABILITY,None,1, theta, epsilon, gamma,
                                  alpha_disc)

    start_event = acquisition_process(env,policy_train,i,GENERATION_INTERVAL,False,None,None,None)

    env.process(start_event.generate_tokens())

    env.run(until=SIM_TIME)

    K_BATCH_MC_VFA.update_theta(policy_train)

epsilon = 0.0

env = simpy.Environment()

file_policy = create_files("{}.csv".format(policy_name))

# this method only works with batch size 1
policy = K_BATCH_MC_VFA(env, NUMBER_OF_USERS, WORKER_VARIABILITY,file_policy,1, theta, epsilon, gamma,
                        initial_alpha)

start_event = acquisition_process(env,policy,1,GENERATION_INTERVAL,False,None,None,None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
