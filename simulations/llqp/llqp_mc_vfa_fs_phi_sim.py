import numpy as np
import simpy

from elements.workflow_process_elements import StartEvent, UserTask, connect
from evaluation.plot import evolution
from evaluation.statistics import calculate_statistics
from policies.llqp.llqp_mc_vfa_fs import LLQP_MC_VFA_FS
from simulations import *
from evaluation.phi_plot import phi_plot

# init theta and reinforcement learning variables
theta = np.ones(NUMBER_OF_USERS ** 2)
gamma = 0.9
epochs = 100
alpha = 0.001
epsilon = 0.0
phis = []
list_of_rewards = []

for phi in np.linspace(4.5, 2*np.pi, 16):
    rewards = []
    for i in range(epochs):
        # creates simulation environment
        env = simpy.Environment()

        # initialize policy
        # FIXME: normalization as method
        theta[0] = 0.0
        theta[1] = 0.0
        theta[2] = np.cos(phi)
        theta[3] = np.sin(phi)

        policy_train = LLQP_MC_VFA_FS(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, None, None, theta/np.linalg.norm(theta), epsilon, gamma, alpha)

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

        avg_reward = np.mean(policy_train.rewards)

        rewards.append(avg_reward)

    list_of_rewards.append(rewards)
    phis.append(phi)

    print("Finished with phi: {}".format(phi))

phi_plot(phis,list_of_rewards)
