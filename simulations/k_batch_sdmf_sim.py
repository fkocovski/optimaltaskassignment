import simpy

from Policies.plot import evolution
from optimaltaskassignment.elements.workflow_process_elements import StartEvent, UserTask, connect
from optimaltaskassignment.evaluation.statistics import calculate_statistics
from optimaltaskassignment.policies.k_batch import KBatch
from optimaltaskassignment.simulations import *
from optimaltaskassignment.solvers.sdmf_solver import sdmf

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy,file_statistics,file_policy_name,file_statistics_name = create_files("batch_sdmf")

# initialize policy
policy = KBatch(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, BATCH_SIZE, sdmf, file_policy, file_statistics)

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

# close file
file_policy.close()
file_statistics.close()

# calculate statistics and plots
calculate_statistics(file_policy_name, outfile="{}.pdf".format(file_policy_name[:-4]))
evolution(file_statistics_name, outfile="{}.pdf".format(file_statistics_name[:-4]))
