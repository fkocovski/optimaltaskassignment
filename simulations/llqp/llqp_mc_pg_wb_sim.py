import numpy as np
import simpy
from evaluation.plot import evolution

from elements.workflow_process_elements import connect
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.llqp import LLQP_MC_PG_WB
from simulations import *

# init theta and reinforcement learning variables
theta = np.zeros(NUMBER_OF_USERS ** 2)
w = np.zeros(NUMBER_OF_USERS)
gamma = 0.9
epochs = 2000
alpha = 0.01
beta = 0.01

for i in range(epochs):
    # creates simulation environment
    env = simpy.Environment()

    # initialize policy
    policy_train = LLQP_MC_PG_WB(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, None, w, theta, gamma, alpha, beta)

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

    # update theta
    LLQP_MC_PG_WB.learn(policy_train)

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("LLQP_MC_PG_WB")

# initialize policy
policy = LLQP_MC_PG_WB(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, file_statistics, w, theta, gamma, alpha,
                       beta)

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