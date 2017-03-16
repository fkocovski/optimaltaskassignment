import numpy as np
import simpy
from evaluation.plot import evolution

from elements.workflow_process_elements import connect
from evaluation.statistics import calculate_statistics
from policies.llqp.llqp_mc import LLQP_MC
from simulations import *

# init q_table and reinforcement learning variables
q_table = np.zeros((100, 100, NUMBER_OF_USERS))
epsilon = 0.3
gamma = 0.5
epochs = 50

for i in range(epochs):
    # creates simulation environment
    env = simpy.Environment()

    # open file and write header
    file_policy,file_statistics,file_policy_name,file_statistics_name = create_files("run{}_mc".format(i))

    # initialize policy
    policy = LLQP_MC(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, file_statistics, q_table, epsilon, gamma)

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

    # update q_table
    new_q_table = LLQP_MC.update_q_table(policy)
    q_table = new_q_table

    # close file
    file_policy.close()
    file_statistics.close()

    # calculate statistics and plots
    calculate_statistics(file_policy_name, outfile="{}.pdf".format(file_policy_name[:-4]))
    evolution(file_statistics_name, outfile="{}.pdf".format(file_statistics_name[:-4]))
