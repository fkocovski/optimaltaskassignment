import simpy
import numpy as np
from policies.mcd_llqp import LLQP
from simulations import *

RANDOM_STATE = np.random.RandomState(1)

states_actions = np.zeros((100, 100, 2))
returns = []
history = []

class monte_carlo(object):
    def __init__(self, env,states_actions):
        self.env = env
        self.policy = LLQP(self.env, NUMBER_OF_USERS, WORKER_VARAIBILITY, TASK_VARIABILITY, SERVICE_INTERVAL, states_actions)
        self.action = env.process(self.generate_tokens())

    def request_from_policy(self):
        policy_job, rwd, current_state = self.policy.request()
        returns.append(rwd)
        history.append(current_state)
        service_time = yield policy_job.request_event
        yield self.env.timeout(service_time)
        self.policy.release(policy_job)

    def generate_tokens(self):
        while True:
            exp_arrival = RANDOM_STATE.exponential(GENERATION_INTERVAL)
            yield self.env.timeout(exp_arrival)
            self.env.process(self.request_from_policy())






def update_q():
    for index,sa in enumerate(history):
        print(returns[index])
        states_actions[sa] += (1 / history.count(sa)) * (returns[index] - states_actions[sa])

for i in range(1):
    # creates simulation environment
    env = simpy.Environment()

    # open file and write header
    # file_policy, file_statistics, file_policy_name, file_statistics_name = create_files("mcd_llqp")

    # initialize policy
    # policy = LLQP(env, NUMBER_OF_USERS, WORKER_VARAIBILITY, TASK_VARIABILITY, SERVICE_INTERVAL,states_actions)

    # start event
    # start_event = StartEvent(env, GENERATION_INTERVAL)

    # user tasks
    # user_task = UserTask(env, policy, "User task 1", SERVICE_INTERVAL, TASK_VARIABILITY)

    # connections
    # connect(start_event, user_task)

    # calls generation tokens process
    # env.process(start_event.generate_tokens())

    # env.process(generate_tokens(env))

    mc = monte_carlo(env,states_actions)
    # runs simulation
    env.run(until=SIM_TIME)
    update_q()
    returns.clear()
    history.clear()
    # close file
    # file_policy.close()
    # file_statistics.close()

    # calculate statistics and plots
    # calculate_statistics(file_policy_name, outfile="{}.pdf".format(file_policy_name[:-4]))
    # evolution(file_statistics_name, outfile="{}.pdf".format(file_statistics_name[:-4]))
