import time

import numpy as np
import simpy

from elements.workflow_process_elements import StartEvent, UserTask, connect
from evaluation.plot import evolution
from evaluation.statistics import calculate_statistics
from policies.llqp.llqp_mc_vfa import LLQP_MC_VFA
from simulations import *

# init theta and reinforcement learning variables
theta = np.zeros(NUMBER_OF_USERS ** 2)
gamma = 0.9
epochs = 3000
initial_alpha = 0.001

# start of simulation
start = time.time()

for i in range(epochs):
    # creates simulation environment
    env = simpy.Environment()

    # fixed parameters
    epsilon = 0.1
    alpha_disc = initial_alpha

    # decay parameters
    # epsilon = 1 / (i + 1)
    # alpha_disc = initial_alpha / (i + 1)

    # initialize policy
    policy_train = LLQP_MC_VFA(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, None, None, theta, epsilon, gamma, alpha_disc)

    # start event
    start_event = StartEvent(env, GENERATION_INTERVAL)

    # user tasks
    user_task = UserTask(env, policy_train, "User task 1", SERVICE_INTERVAL, TASK_VARIABILITY)

    # connections
    connect(start_event, user_task)

    # start of simulation
    # start = time.time()

    # calls generation tokens process
    env.process(start_event.generate_tokens())

    # runs simulation
    env.run(until=SIM_TIME)

    # end of simulation
    end = time.time()

    # update theta
    LLQP_MC_VFA.update_theta(policy_train)

    # if i % 50 == 0:
        # end of simulation
        # end = time.time()
        # print("FINISHED TRAINING SESSION {} in {}".format(i,end-start))

# set epsilon to 0.0 to make test policy behave full greedy
epsilon = 0.0

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("LLQP_MC_VFA")

# initialize policy
policy = LLQP_MC_VFA(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, file_policy, file_statistics, theta, epsilon, gamma,
                     initial_alpha)

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

# if NUMBER_OF_USERS == 2:
    # value action for plot
    # value_action = policy.value_function()
    # qsa_values(value_action)
