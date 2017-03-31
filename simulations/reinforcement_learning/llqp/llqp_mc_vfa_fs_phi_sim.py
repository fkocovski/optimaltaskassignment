import numpy as np
import simpy
from evaluation.phi_plot import phi_plot
from policies.reinforcement_learning.llqp.llqp_mc_vfa_fs import LLQP_MC_VFA_FS
from simulations import *

theta = np.ones(NUMBER_OF_USERS ** 2)
gamma = 0.5
epochs = 100
alpha = 0.001
epsilon = 0.0
policy_name = "LLQP_MC_VFA_FS_ETA_NU{}_GI{}_SIM{}".format(NUMBER_OF_USERS, GENERATION_INTERVAL, SIM_TIME)

phis = []
list_of_rewards = []

for phi in np.linspace(4.5, 2*np.pi, 16):
    rewards = []
    for i in range(epochs):
        env = simpy.Environment()

        theta[0] = 0.0
        theta[1] = 0.0
        theta[2] = np.cos(phi)
        theta[3] = np.sin(phi)

        policy = LLQP_MC_VFA_FS(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, theta / np.linalg.norm(theta), epsilon, gamma, alpha)

        start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

        env.process(start_event.generate_tokens())

        env.run(until=SIM_TIME)

        avg_reward = np.mean(policy.rewards)

        rewards.append(avg_reward)

    list_of_rewards.append(rewards)
    phis.append(phi)


phi_plot(phis,list_of_rewards,policy_name,outfile=False)
