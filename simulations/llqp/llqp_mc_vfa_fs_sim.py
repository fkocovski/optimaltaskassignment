import numpy as np
import simpy

from elements.workflow_process_elements import StartEvent, UserTask, connect
from evaluation.plot import evolution
from evaluation.statistics import calculate_statistics
from policies.llqp.llqp_mc_vfa_fs import LLQP_MC_VFA_FS
from simulations import *
from evaluation.composed_history import composed_history

# init theta and reinforcement learning variables
theta = np.zeros(NUMBER_OF_USERS ** 2)
gamma = 0.5
epochs = 10
alpha = 0.0001
# epsilon = 0.1

for i in range(epochs):
    # creates simulation environment
    env = simpy.Environment()

    # initialize policy
    policy_train = LLQP_MC_VFA_FS(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, None, None, theta, np.sqrt(1/(i+1)), gamma, alpha)

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

    # comp_history = policy_train.compose_history()

    # update theta
    # policy_train.update_theta()
    policy_train.regression_fit()
    if i % 1 == 0:
        print("Finished {}th train run".format(i))
        # composed_history(comp_history)


# set epsilon to 0.0 to make test policy behave full greedy
epsilon = 0.0

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("LLQP_MC_VFA_FS")

# initialize policy
policy = LLQP_MC_VFA_FS(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, file_policy, file_statistics, theta, epsilon, gamma,
                     alpha)

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

comp_history = policy.compose_history()

# close file
file_policy.close()
file_statistics.close()

# calculate statistics and plots
calculate_statistics(file_policy_name, outfile="{}.pdf".format(file_policy_name[:-4]))
evolution(file_statistics_name, outfile="{}.pdf".format(file_statistics_name[:-4]))
composed_history(comp_history)