import simpy

from evaluation.evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.batch.k_batchone import K_BATCHONE
from simulations import *
from solvers.msa_solver import msa

# creates simulation environment
env = simpy.Environment()

# open file and write header
file_policy,file_statistics,file_policy_name,file_statistics_name = create_files("{}batchone_msa".format(BATCH_SIZE))

# initialize policy
policy = K_BATCHONE(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, BATCH_SIZE, msa, file_policy, file_statistics)

# process initialization
start_event = initialize_process(env,policy)

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