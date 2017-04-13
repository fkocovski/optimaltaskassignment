import numpy as np
import simpy
from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.others.wz_td_vfa_op import WZ_TD_VFA_OP
from simulations import *

theta = np.zeros((NUMBER_OF_USERS ** BATCH_SIZE, NUMBER_OF_USERS + BATCH_SIZE))
gamma = 0.5
alpha = 0.001
sim_time_training = SIM_TIME*100
policy_name = "{}_WZ_TD_VFA_OP_NU{}_GI{}_TRSD{}_SIM{}".format(BATCH_SIZE, NUMBER_OF_USERS, GENERATION_INTERVAL, SEED,
                                                             SIM_TIME)

env = simpy.Environment()

policy_train = WZ_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, theta, gamma, alpha, False, BATCH_SIZE,SEED)

start_event = acquisition_process(env, policy_train, SEED, GENERATION_INTERVAL, False, None, None, None)

env.process(start_event.generate_tokens())

env.run(until=sim_time_training)

env = simpy.Environment()

file_policy = create_files("{}.csv".format(policy_name))

policy = WZ_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, theta, gamma, alpha, True, BATCH_SIZE,1)

start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
