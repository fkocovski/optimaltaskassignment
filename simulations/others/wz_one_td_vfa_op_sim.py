import numpy as np
import simpy

from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.others.wz_one_td_vfa_op import WZ_ONE_TD_VFA_OP
from simulations import *

# init theta and reinforcement learning variables
wait_size = 2
theta = np.zeros((NUMBER_OF_USERS**wait_size,NUMBER_OF_USERS+2*wait_size))
gamma = 0.5
alpha = 0.0001

# creates simulation environment
env = simpy.Environment()

# initialize policy
policy_train = WZ_ONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, None, None, theta, gamma, alpha,False,wait_size)

# initialize process
start_event = initialize_process(env,policy_train)

# calls generation tokens process
env.process(start_event.generate_tokens())

# runs simulation
env.run(until=500)

# creates simulation environment
env = simpy.Environment()

# open file and write header
# file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("WZ_ONE_TD_VFA_OP")
file_policy = create_files("WZ_ONE_TD_VFA_OP")

# initialize policy
policy = WZ_ONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, file_policy, None, theta, gamma, alpha,True,wait_size)

# initialize process
start_event = initialize_process(env,policy)

# calls generation tokens process
env.process(start_event.generate_tokens())

# runs simulation
env.run(until=SIM_TIME)

# close file
file_policy.close()
# file_statistics.close()

# calculate statistics and plots
calculate_statistics(file_policy.name, outfile=False)
# evolution(file_policy.name,outfile=False)