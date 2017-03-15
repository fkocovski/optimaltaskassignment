import simpy

from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.llqp.llqp import LLQP
from simulations import *

env = simpy.Environment()

file_policy = create_files("LLQP_NU{}_GI{}_TRSD{}_SIM{}.csv".format(NUMBER_OF_USERS,GENERATION_INTERVAL,SEED,SIM_TIME))

policy = LLQP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy)

start_event = acquisition_process(env, policy,1,GENERATION_INTERVAL,False,None,None,None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
