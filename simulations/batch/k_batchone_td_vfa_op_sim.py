import numpy as np
import simpy

from elements.workflow_process_elements import StartEvent, UserTask, connect
from evaluation.plot import evolution
from evaluation.statistics import calculate_statistics
from policies.batch.k_batchone_td_vfa_op import K_BATCHONE_TD_VFA_OP
from simulations import *

# init theta and reinforcement learning variables
theta = np.zeros((NUMBER_OF_USERS,NUMBER_OF_USERS+1))
gamma = 0.5
alpha = 0.00001

# creates simulation environment
env = simpy.Environment()

# initialize policy
policy_train = K_BATCHONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, None, None,1, theta, gamma, alpha,False)

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
file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("K_BATCHONE_TD_VFA_OP")

# initialize policy
policy = K_BATCHONE_TD_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, file_policy, file_statistics,1, theta, gamma, alpha,True)

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