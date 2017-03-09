import simpy

from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.batch.k_batchone import K_BATCHONE
from simulations import *
from solvers.st_solver import st

env = simpy.Environment()

file_policy = create_files("{}BATCHONE_ST".format(BATCH_SIZE))

policy = K_BATCHONE(env, NUMBER_OF_USERS, WORKER_VARIABILITY, BATCH_SIZE, st, file_policy)

start_event = acquisition_process(env, policy)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name,outfile=True)