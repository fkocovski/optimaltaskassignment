import simpy
import numpy as np
from evaluation.plot import evolution
from elements.workflow_process_elements import StartEvent, UserTask, connect
from evaluation.statistics import calculate_statistics
from policies.llqp_mc_pg_wb import LLQP_MC_PG_WB
from simulations import *
import time

start = time.time()

# init theta and reinforcement learning variables
theta = np.zeros(NUMBER_OF_USERS ** 2)
w = np.zeros(NUMBER_OF_USERS)
gamma = 0.9
epochs = 1000
alpha = 0.01
beta = 0.01

for i in range(epochs):
    # creates simulation environment
    env = simpy.Environment()

    # decay parameters
    alpha_disc = alpha/(i+1)
    beta_disc = beta/(i+1)

    # initialize policy
    policy_train = LLQP_MC_PG_WB(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, None, None, w, theta, gamma, alpha_disc, beta_disc)

    # start event
    start_event = StartEvent(env, GENERATION_INTERVAL)

    # user tasks
    user_task = UserTask(env, policy_train, "User task 1", SERVICE_INTERVAL, TASK_VARIABILITY)

    # connections
    connect(start_event, user_task)

    # calls generation tokens process
    env.process(start_event.generate_tokens())

    # runs simulation
    print(theta)
    env.run(until=SIM_TIME)

    # update theta
    LLQP_MC_PG_WB.learn(policy_train)

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("LLQP_MC_PG_WB")

# initialize policy
policy = LLQP_MC_PG_WB(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, file_policy, file_statistics, w, theta, gamma, alpha,
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

end = time.time()

print(end-start)