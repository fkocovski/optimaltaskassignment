import matplotlib.pyplot as plt
import numpy as np
import simpy

from elements.workflow_process_elements import connect
from policies.reinforcement_learning.llqp.llqp_pg_alpha import LLQP_PG_ALPHA
from simulations import *

# init theta and reinforcement learning variables
theta = np.zeros(NUMBER_OF_USERS ** 2)
w = np.zeros(NUMBER_OF_USERS**2)
gamma = 0.9
alpha = 0.000001
beta = 0.000001
eta = 0.0

etas = []
list_of_rewards = []

while eta < 5:
    theta[0] = -eta
    theta[1] = eta
    theta[2] = eta
    theta[3] = -eta

    rewards = []
    for i in range(100):
        # creates simulation environment
        env = simpy.Environment()

        # open file and write header
        # file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("LLQP_PG_AVAC")

        # initialize policy
        policy = LLQP_PG_ALPHA(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, None, w, theta, gamma, alpha,
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

        avg_reward = np.mean(policy.rewards)

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

avg_rewards = [np.mean(rwd) for rwd in list_of_rewards]
std_rewards = [np.std(rwd) for rwd in list_of_rewards]
plt.xlabel("eta value")
plt.ylabel("average reward")
plt.grid(True)
plt.errorbar(etas,avg_rewards,yerr=std_rewards,fmt="o",capsize=10,label="SD")
plt.plot(etas,avg_rewards,label="Mean lateness")
plt.legend()
plt.show()

