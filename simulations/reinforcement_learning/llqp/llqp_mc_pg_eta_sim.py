import numpy as np
import simpy
from evaluation.eta_plot import eta_plot
from policies.reinforcement_learning.llqp.llqp_mc_pg import LLQP_MC_PG
from simulations import *

theta = np.zeros(NUMBER_OF_USERS ** 2)
gamma = 0.5
epochs = 50
alpha = 0.0001
eta = 0.0
policy_name = "LLQP_MC_PG_ETA_NU{}_GI{}_SIM{}".format(NUMBER_OF_USERS, GENERATION_INTERVAL, SIM_TIME)

etas = []
list_of_rewards = []

while eta < 30:
    theta[0] = -eta
    theta[1] = eta
    theta[2] = eta
    theta[3] = -eta

    rewards = []
    for j in range(epochs):
        env = simpy.Environment()

        policy = LLQP_MC_PG(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, theta, gamma, alpha,1)

        start_event = acquisition_process(env, policy, j, GENERATION_INTERVAL, False, None, None, None)

        env.process(start_event.generate_tokens())

        env.run(until=SIM_TIME)

        avg_reward = np.mean(policy.jobs_lateness)

        rewards.append(avg_reward)

    list_of_rewards.append(rewards)
    etas.append(eta)

    eta += 1

eta_plot(etas,list_of_rewards,policy_name,outfile=True)