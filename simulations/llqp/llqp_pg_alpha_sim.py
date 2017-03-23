import numpy as np
import simpy
from evaluation.eta_plot import eta_plot
from policies.reinforcement_learning.llqp.llqp_pg_alpha import LLQP_PG_ALPHA
from simulations import *

theta = np.zeros(NUMBER_OF_USERS ** 2)
w = np.zeros(NUMBER_OF_USERS**2)
epochs = 100
gamma = 0.9
alpha = 0.000001
beta = 0.000001
eta = 0.0

policy_name = "LLQP_PG_ALPHA_ETA_NU{}_GI{}_TRSD{}_SIM{}".format(NUMBER_OF_USERS, GENERATION_INTERVAL, SEED, SIM_TIME)

etas = []
list_of_rewards = []

while eta < 5:
    theta[0] = -eta
    theta[1] = eta
    theta[2] = eta
    theta[3] = -eta

    rewards = []
    for j in range(epochs):
        env = simpy.Environment()

        policy = LLQP_PG_ALPHA(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None,w, theta, gamma, alpha,beta)

        start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

        env.process(start_event.generate_tokens())

        env.run(until=SIM_TIME)

        avg_reward = np.mean(policy.rewards)

        rewards.append(avg_reward)

    list_of_rewards.append(rewards)
    etas.append(eta)

    eta += 0.5

eta_plot(etas,list_of_rewards,policy_name,outfile=False)