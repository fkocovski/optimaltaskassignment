import simpy

from evaluation.statistics import calculate_statistics
from evaluation.subplot_evolution import evolution
from policies.optimization.batch.k_batch import K_BATCH
from simulations import *
from solvers.esdmf_solver import esdmf

policy_name = "{}_BATCH_ESDMF_NU{}_GI{}_SIM{}".format(BATCH_SIZE,NUMBER_OF_USERS, GENERATION_INTERVAL, SIM_TIME)

env = simpy.Environment()

file_policy = create_files("{}.csv".format(policy_name))

policy = K_BATCH(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy,BATCH_SIZE, esdmf)

start_event = acquisition_process(env,policy,1,GENERATION_INTERVAL,False,None,None,None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
