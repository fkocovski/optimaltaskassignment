import numpy as np
import simpy

from elements.workflow_process_elements import connect
from policies.reinforcement_learning.llqp import LLQP_PG_AVAC
from simulations import *

# init theta and reinforcement learning variables
theta = np.zeros(NUMBER_OF_USERS ** 2)
w = np.zeros(NUMBER_OF_USERS**2)
gamma = 0.9
alpha = 0.000001
beta = 0.000001

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("LLQP_PG_AVAC")

# initialize policy
policy = LLQP_PG_AVAC(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, file_statistics, w, theta, gamma, alpha,
                      beta)
# start event
start_event = StartEvent(env, GENERATION_INTERVAL)

# user tasks
user_task = UserTask(env, policy, "User task 1", SERVICE_INTERVAL, TASK_VARIABILITY)

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
# calculate_statistics(file_policy_name, outfile="{}.pdf".format(file_policy_name[:-4]))
# evolution(file_statistics_name, outfile="{}.pdf".format(file_statistics_name[:-4]))

