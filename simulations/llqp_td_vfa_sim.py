import simpy
import numpy as np
from evaluation.plot import evolution
from elements.workflow_process_elements import StartEvent, UserTask, connect
from evaluation.statistics import calculate_statistics
from policies.llqp_td_vfa import LLQP_TD_VFA
from simulations import *
import time

# init theta and reinforcement learning variables
theta = np.zeros(NUMBER_OF_USERS ** 2)
gamma = 0.9
alpha = 0.001
epsilon = 0.1

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("LLQP_TD_VFA")

# initialize policy
policy_train = LLQP_TD_VFA(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, file_policy, file_statistics, theta, epsilon, gamma, alpha)

# start event
start_event = StartEvent(env, GENERATION_INTERVAL)

# user tasks
user_task = UserTask(env, policy_train, "User task 1", SERVICE_INTERVAL, TASK_VARIABILITY)

# connections
connect(start_event, user_task)

# calls generation tokens process
env.process(start_event.generate_tokens())

# runs simulation
env.run(until=SIM_TIME)

# close file
file_policy.close()
file_statistics.close()

# calculate statistics and plots
calculate_statistics(file_policy_name, outfile="{}.pdf".format(file_policy_name[:-4]))
# evolution(file_statistics_name, outfile="{}.pdf".format(file_statistics_name[:-4]))
