import numpy as np
import simpy
from evaluation.matrix_composed_history import matrix_composed_history
from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.batch.k_batch_td_vfa_op import K_BATCH_TD_VFA_OP
from simulations import *

theta = np.zeros((NUMBER_OF_USERS,NUMBER_OF_USERS+1))
gamma = 0.5
alpha = 0.001
sim_time_training = SIM_TIME * 2
policy_name = "{}BATCH_TD_VFA_OP_NU{}_GI{}_TRSD{}_SIM{}".format(1,NUMBER_OF_USERS, GENERATION_INTERVAL, SEED, SIM_TIME)

env = simpy.Environment()

policy_train = K_BATCH_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, 1, theta, gamma, alpha, False)

start_event = acquisition_process(env, policy_train, SEED, GENERATION_INTERVAL, False, None, None, None)

env.process(start_event.generate_tokens())

env.run(until=sim_time_training)

env = simpy.Environment()

file_policy = create_files("{}.csv".format(policy_name))

policy = K_BATCH_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, 1, theta, gamma, alpha, True)

start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

# works only for 2 users
comp_history = policy.compose_history()

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
# works only for 2 users
matrix_composed_history(comp_history)