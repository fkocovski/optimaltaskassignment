import numpy as np
import simpy

from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.llqp.llqp_td_tf_op import LLQP_TD_TF_OP
from simulations import *

theta = np.zeros((NUMBER_OF_USERS, NUMBER_OF_USERS))
gamma = 0.5
alpha = 0.001
sim_time_training = SIM_TIME*5

env = simpy.Environment()

policy_train = LLQP_TD_TF_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, theta, gamma, alpha, False)

start_event = acquisition_process(env, policy_train,SEED,GENERATION_INTERVAL,False,None,None,None)

env.process(start_event.generate_tokens())

env.run(until=sim_time_training)

env = simpy.Environment()

file_policy = create_files("{}_NU{}_GI{}_TRSD{}_SIM{}.csv".format(policy_train.name,NUMBER_OF_USERS,GENERATION_INTERVAL,SEED,SIM_TIME))

policy = LLQP_TD_TF_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, theta, gamma, alpha, True)

start_event = acquisition_process(env, policy,1,GENERATION_INTERVAL,False,None,None,None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
