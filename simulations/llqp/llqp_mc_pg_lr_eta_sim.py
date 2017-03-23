import numpy as np
import simpy

from elements.workflow_process_elements import connect
from evaluation.eta_plot import eta_plot
from policies.reinforcement_learning.llqp import LLQP_MC_PG_LR
from simulations import *

# init theta and reinforcement learning variables
theta = np.zeros(NUMBER_OF_USERS ** 2)
gamma = 0.9
epochs = 100
alpha = 0.000001
eta = 0.0

etas = []
list_of_rewards = []

while eta < 5:
    theta[0] = -eta
    theta[1] = eta
    theta[2] = eta
    theta[3] = -eta

    rewards = []
    for j in range(100000):
        # creates simulation environment
        env = simpy.Environment()

        # open file and write header
        # file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("LLQP_MC_PG_LR")

        # initialize policy
        policy = LLQP_MC_PG_LR(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, None, theta, gamma, alpha)

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

        avg_reward = np.mean(policy.jobs_lateness)

        rewards.append(avg_reward)
        # close file
        # file_policy.close()
        # file_statistics.close()

        # calculate statistics and plots
        # calculate_statistics(file_policy_name, outfile="{}.pdf".format(file_policy_name[:-4]))
        # evolution(file_statistics_name, outfile="{}.pdf".format(file_statistics_name[:-4]))
    list_of_rewards.append(rewards)
    etas.append(eta)

    eta += 0.5

eta_plot(etas,list_of_rewards,"eta_plot.pdf")