import numpy as np
import simpy

from elements.workflow_process_elements import StartEvent, UserTask, connect
from evaluation.plot import evolution
from evaluation.statistics import calculate_statistics
from policies.others.wz_td_vfa_op import WZ_TD_VFA_OP
from simulations import *

# init theta and reinforcement learning variables
wait_size = 2
theta = np.zeros((NUMBER_OF_USERS**wait_size,NUMBER_OF_USERS+wait_size))
gamma = 0.5
alpha = 0.0001

# creates simulation environment
env = simpy.Environment()

# initialize policy
policy_train = WZ_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, None, theta, gamma, alpha, False, wait_size)

# start event
start_event = StartEvent(env, GENERATION_INTERVAL)

# user tasks
user_task = UserTask(env, policy_train, "User task 1", SERVICE_INTERVAL, TASK_VARIABILITY)

# connections
connect(start_event, user_task)

# calls generation tokens process
env.process(start_event.generate_tokens())

# runs simulation
env.run(until=1000)

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("WZ_TD_VFA_OP")

# initialize policy
policy = WZ_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, file_statistics, theta, gamma, alpha, True, wait_size)

# start event
start_event_test = StartEvent(env, GENERATION_INTERVAL)

# user tasks
user_task_test = UserTask(env, policy, "User task 1", SERVICE_INTERVAL, TASK_VARIABILITY)

# connections
connect(start_event_test, user_task_test)

# calls generation tokens process
env.process(start_event_test.generate_tokens())

# runs simulation
env.run(until=SIM_TIME)

# close file
file_policy.close()
file_statistics.close()

# calculate statistics and plots
calculate_statistics(file_policy_name, outfile="{}.pdf".format(file_policy_name[:-4]))
evolution(file_statistics_name, outfile="{}.pdf".format(file_statistics_name[:-4]))