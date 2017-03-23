import numpy as np
import simpy

from evaluation.statistics import calculate_statistics
from evaluation.subplot_evolution import evolution
from policies.reinforcement_learning.others import WZ_ONE_TD_VFA_OP
from simulations import *

theta = np.zeros((NUMBER_OF_USERS ** BATCH_SIZE, NUMBER_OF_USERS + 2 * BATCH_SIZE))
gamma = 0.5
alpha = 0.001
sim_time_training = SIM_TIME*5

env = simpy.Environment()

policy_train = WZ_ONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, theta, gamma, alpha, False,
                                BATCH_SIZE)

start_event = acquisition_process(env, policy_train,SEED,GENERATION_INTERVAL,False,None,None,None)

env.process(start_event.generate_tokens())

env.run(until=sim_time_training)

env = simpy.Environment()

file_policy = create_files("{}_BS{}_NU{}_GI{}_TRSD{}_SIM{}.csv".format(policy_train.name,BATCH_SIZE,NUMBER_OF_USERS,GENERATION_INTERVAL,SEED,SIM_TIME))


policy = WZ_ONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, theta, gamma, alpha, True,
                          BATCH_SIZE)

start_event = acquisition_process(env, policy,1,GENERATION_INTERVAL,False,None,None,None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

# TODO only for statistical checks. Remove when finished
print("Out of {} evaluations in {} cases there was a user actually free, i.e. in {:.2f}% of the cases!".format(policy.total_evals,policy.actually_free,policy.actually_free/policy.total_evals*100))

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
