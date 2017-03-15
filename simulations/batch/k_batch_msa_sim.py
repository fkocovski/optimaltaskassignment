import simpy

from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.batch.k_batch import K_BATCH
from simulations import *
from solvers.msa_solver import msa

env = simpy.Environment()

file_policy = create_files("{}BATCH_MSA_BS{}_NU{}_GI{}_SIM{}.csv".format(BATCH_SIZE,BATCH_SIZE,NUMBER_OF_USERS,GENERATION_INTERVAL,SIM_TIME))

policy = K_BATCH(env, NUMBER_OF_USERS, WORKER_VARIABILITY,file_policy, BATCH_SIZE, msa)

start_event = acquisition_process(env, policy,1,GENERATION_INTERVAL,False,None,None,None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name,outfile=True)
