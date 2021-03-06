import simpy

from evaluation.statistics import calculate_statistics
from evaluation.subplot_evolution import evolution
from policies.optimization.llqp.llqp import LLQP
from simulations import *

policy_name = "LLQP_NU{}_GI{}_SIM{}".format(NUMBER_OF_USERS, GENERATION_INTERVAL, SIM_TIME)

env = simpy.Environment()

file_policy = create_files("{}.csv".format(policy_name))

policy = LLQP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy)

start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

env.process(start_event.generate_tokens())

env.run(until=SIM_TIME)

file_policy.close()

calculate_statistics(file_policy.name, outfile=True)
evolution(file_policy.name, outfile=True)
