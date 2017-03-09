import simpy

from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.batch.k_batchone import K_BATCHONE
from simulations import *
from solvers.st_solver import st

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy = create_files("{}BATCHONE_ST".format(BATCH_SIZE))

# initialize policy
policy = K_BATCHONE(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, BATCH_SIZE, st, file_policy, None)

# process initialization
start_event = initialize_process(env,policy)

# calls generation tokens process
env.process(start_event.generate_tokens())

# runs simulation
env.run(until=SIM_TIME)

# close file
file_policy.close()
# file_statistics.close()

# calculate statistics and plots
calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name,outfile=True)