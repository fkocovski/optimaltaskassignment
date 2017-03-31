import numpy as np
import simpy
from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.llqp.llqp_td_pg_ac import LLQP_TD_PG_AC
from simulations import *

theta = np.zeros(NUMBER_OF_USERS ** 2)
w = np.zeros(NUMBER_OF_USERS)
gamma = 0.5
alpha = 0.001
beta = 0.001
policy_name = "LLQP_TD_PG_AC_NU{}_GI{}_SIM{}".format(NUMBER_OF_USERS, GENERATION_INTERVAL, SIM_TIME)

env = simpy.Environment()

file_policy = create_files("{}.csv".format(policy_name))

policy = LLQP_TD_PG_AC(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, w, theta, gamma, alpha,
                       beta, 1)

start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME*10)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
