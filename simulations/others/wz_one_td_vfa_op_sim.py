import numpy as np
import simpy

from evaluation.plot import evolution
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

# start event
# start_event = StartEvent(env, GENERATION_INTERVAL)

# user tasks
# user_task = UserTask(env, policy_train, "User task 1", SERVICE_INTERVAL, TASK_VARIABILITY)

# connections
# connect(start_event, user_task)

# calls generation tokens process
env.process(start_event.generate_tokens())

# runs simulation
env.run(until=1000)

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("WZ_ONE_TD_VFA_OP")

# initialize policy
policy = WZ_ONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, file_policy, file_statistics, theta, gamma, alpha,True,wait_size)

# initialize process
start_event = initialize_process(env,policy)

# start event
# start_event_test = StartEvent(env, GENERATION_INTERVAL)

# user tasks
# user_task_test = UserTask(env, policy, "User task 1", SERVICE_INTERVAL, TASK_VARIABILITY)

# connections
# connect(start_event_test, user_task_test)

# calls generation tokens process
env.process(start_event.generate_tokens())

# runs simulation
env.run(until=SIM_TIME)

# close file
file_policy.close()
file_statistics.close()

# calculate statistics and plots
calculate_statistics(file_policy_name, outfile="{}.pdf".format(file_policy_name[:-4]))
evolution(file_statistics_name, outfile="{}.pdf".format(file_statistics_name[:-4]))