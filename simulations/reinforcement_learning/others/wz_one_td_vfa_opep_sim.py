import numpy as np
import simpy
from evaluation.sigmoid import sigmoid
from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.others.wz_one_td_vfa_opep import WZ_ONE_TD_VFA_OPEP
from simulations import *

theta = np.zeros((NUMBER_OF_USERS ** BATCH_SIZE, NUMBER_OF_USERS + 2 * BATCH_SIZE))
gamma = 0.5
alpha = 0.0001
sim_time_training = SIM_TIME
sigmoid_param = 0.01
policy_name = "{}_WZ_ONE_TD_VFA_OPEP_NU{}_GI{}_TRSD{}_SIM{}".format(BATCH_SIZE, NUMBER_OF_USERS, GENERATION_INTERVAL, SEED,
                                                             SIM_TIME)

env = simpy.Environment()

policy_train = WZ_ONE_TD_VFA_OPEP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, theta, gamma, alpha, False,
                                BATCH_SIZE,sim_time_training,sigmoid_param,SEED)

start_event = acquisition_process(env, policy_train,SEED,GENERATION_INTERVAL,True,4,sim_time_training,sigmoid_param)

env.process(start_event.generate_tokens())

env.run(until=sim_time_training)

sigmoid(start_event.t,start_event.g,policy_train.t,policy_train.g,policy_name, outfile=True)

env = simpy.Environment()

file_policy = create_files("{}.csv".format(policy_name))


policy = WZ_ONE_TD_VFA_OPEP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy,theta, gamma, alpha, True,
                          BATCH_SIZE,None,None,1)

start_event = acquisition_process(env, policy,1,GENERATION_INTERVAL,False,None,None,None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
