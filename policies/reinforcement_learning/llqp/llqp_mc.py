import numpy as np
import randomstate.prng.pcg64 as pcg
from policies import *
from collections import deque


class LLQP_MC(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, q_table, epsilon, gamma):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.q_table = q_table
        self.epsilon = epsilon
        self.gamma = gamma
        self.EPSILON_GREEDY_RANDOM_STATE = pcg.RandomState(1)
        self.name = "LLQP_MC"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.history = []
        self.returns = []

    def request(self, user_task,token):
        llqp_job = super().request(user_task,token)

        self.evaluate(llqp_job)

        return llqp_job

    def release(self, llqp_job):
        super().release(llqp_job)

        user_to_release_index = llqp_job.assigned_user
        user_queue_to_free = self.users_queues[user_to_release_index]
        user_queue_to_free.popleft()

        if len(user_queue_to_free) > 0:
            next_llqp_job = user_queue_to_free[0]
            next_llqp_job.started = self.env.now
            next_llqp_job.assigned = self.env.now
            next_llqp_job.request_event.succeed(next_llqp_job.service_rate[user_to_release_index])

    def evaluate(self, llqp_job):
        busy_times = [None] * self.number_of_users
        for user_index, user_deq in enumerate(self.users_queues):
            if len(user_deq) > 0:
                leftmost_queue_element = user_deq[0]
                busy_times[user_index] = sum(job.service_rate[user_index] for job in user_deq)
                if leftmost_queue_element.is_busy(self.env.now):
                    busy_times[user_index] -= self.env.now - leftmost_queue_element.started
            else:
                busy_times[user_index] = 0

        # TODO: do not use fixed index for current_state
        current_state = [None] * self.number_of_users
        for i, a in enumerate(busy_times):
            current_state[i] = int(a)
        if self.EPSILON_GREEDY_RANDOM_STATE.rand() < self.epsilon:
            action = self.EPSILON_GREEDY_RANDOM_STATE.randint(0, 2)
        else:
            action = np.argmax(self.q_table[current_state[0], current_state[1]])

        self.update_policy_status(current_state[0], current_state[1], action)

        llqp_queue = self.users_queues[action]
        llqp_job.assigned_user = action
        llqp_job.assigned = self.env.now
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[action])


    def update_policy_status(self, user_one, user_two, action):
        if user_one < user_two and action == 0:
            self.returns.append(1)
        elif user_one > user_two and action == 1:
            self.returns.append(1)
        elif user_one == user_two:
            self.returns.append(1)
        else:
            self.returns.append(-1)

        self.history.append((user_one, user_two, action))

    def update_q_table(self):
        rewards = [None] * len(self.returns)
        for index, ret in enumerate(reversed(self.returns)):
            if index == 0:
                rewards[index] = ret
            else:
                rewards[-index] = ret + self.gamma * rewards[-index + 1]
        for index, sa in enumerate(self.history):
            self.q_table[sa] += (1 / self.history.count(sa)) * (rewards[index] - self.q_table[sa])

        return self.q_table
