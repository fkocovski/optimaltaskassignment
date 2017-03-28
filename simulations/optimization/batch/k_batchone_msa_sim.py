import simpy

from evaluation.statistics import calculate_statistics
from evaluation.subplot_evolution import evolution
from policies.optimization.batch.k_batchone import K_BATCHONE
from simulations import *
from solvers.msa_solver import msa

policy_name = "{}BATCHONE_MSA_NU{}_GI{}_SIM{}".format(BATCH_SIZE,NUMBER_OF_USERS, GENERATION_INTERVAL, SEED, SIM_TIME)

env = simpy.Environment()

file_policy = create_files("{}.csv".format(policy_name))

policy = K_BATCHONE(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, BATCH_SIZE, msa)

start_event = simple_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
