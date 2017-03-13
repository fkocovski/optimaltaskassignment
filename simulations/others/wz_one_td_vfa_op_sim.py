import numpy as np
import simpy

from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.others.wz_one_td_vfa_op import WZ_ONE_TD_VFA_OP
from simulations import *

theta = np.zeros((NUMBER_OF_USERS ** BATCH_SIZE, NUMBER_OF_USERS + 2 * BATCH_SIZE))
gamma = 0.5
alpha = 0.0001

env = simpy.Environment()

policy_train = WZ_ONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, theta, gamma, alpha, False,
                                BATCH_SIZE,seed=SEED)

start_event = acquisition_process(env, policy_train,seed=SEED,generation_interval=1.1)

env.process(start_event.generate_tokens())

env.run(until=10000)

env = simpy.Environment()

file_policy = create_files("WZ_ONE_TD_VFA_OP_BS{}_NU{}_GI{}_TRSD{}_SIM{}.csv".format(BATCH_SIZE,NUMBER_OF_USERS,GENERATION_INTERVAL,SEED,SIM_TIME))

policy = WZ_ONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, theta, gamma, alpha, True,
                          BATCH_SIZE)

start_event = acquisition_process(env, policy)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
