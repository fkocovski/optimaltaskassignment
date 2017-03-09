import numpy as np
import simpy

from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.others.wz_one_td_vfa_op import WZ_ONE_TD_VFA_OP
from simulations import *

wait_size = 2
theta = np.zeros((NUMBER_OF_USERS ** wait_size, NUMBER_OF_USERS + 2 * wait_size))
gamma = 0.5
alpha = 0.0001

env = simpy.Environment()

policy_train = WZ_ONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, theta, gamma, alpha, False,
                                wait_size)

start_event = initialize_process(env, policy_train)

env.process(start_event.generate_tokens())

env.run(until=5000)

env = simpy.Environment()

file_policy = create_files("WZ_ONE_TD_VFA_OP.csv")

policy = WZ_ONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, theta, gamma, alpha, True,
                          wait_size)

start_event = initialize_process(env, policy)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=False)
evolution(file_policy.name, outfile=False)
