import numpy as np
import simpy

from elements.workflow_process_elements import StartEvent, UserTask, connect
from evaluation.plot import evolution
from evaluation.statistics import calculate_statistics
from evaluation.composed_history import composed_history
from policies.llqp.llqp_mc_vfa_op import LLQP_MC_VFA_OP
from simulations import *

# init theta and reinforcement learning variables
theta = np.zeros(NUMBER_OF_USERS ** 2)
gamma = 0.5
epochs = 1000
alpha = 0.0001

for i in range(epochs):
    # creates simulation environment
    env = simpy.Environment()

    # initialize policy
    policy_train = LLQP_MC_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, None, None, theta, gamma, alpha,False)

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
    policy_train.update_theta()

    # print train episode
    if i % 10 == 0:
        print("Finished {}th train run".format(i))
        print(policy_train.theta)

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("LLQP_MC_VFA_OP")

# initialize policy
policy = LLQP_MC_VFA_OP(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, file_policy, file_statistics, theta, gamma, alpha,True)

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

# composed history data
comp_history = policy.compose_history()

# close file
file_policy.close()
file_statistics.close()

# calculate statistics and plots
calculate_statistics(file_policy_name, outfile="{}.pdf".format(file_policy_name[:-4]))
evolution(file_statistics_name, outfile="{}.pdf".format(file_statistics_name[:-4]))
composed_history(comp_history)