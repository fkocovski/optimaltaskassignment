import numpy as np
import randomstate.prng.pcg64 as pcg
from policies import *
from collections import deque


class LLQP_TD_VFA(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, theta, epsilon, gamma,
                 alpha):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.theta = theta
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.RANDOM_STATE_ACTIONS = pcg.RandomState(1)
        self.name = "LLQP_TD_VFA"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.episode = 0

    def request(self, user_task, token):
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
        busy_times = self.get_busy_times()

        if self.RANDOM_STATE_ACTIONS.rand() < self.epsilon:
            action = self.RANDOM_STATE_ACTIONS.randint(0, self.number_of_users)
        else:
            action = max(range(self.number_of_users),
                         key=lambda action: self.action_value_approximator(busy_times, action))

        reward = busy_times[action] + llqp_job.service_rate[action]

        llqp_queue = self.users_queues[action]
        llqp_job.assigned_user = action
        llqp_job.assigned = self.env.now
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[action])

        busy_times_new = self.get_busy_times()

        if self.RANDOM_STATE_ACTIONS.rand() < self.epsilon:
            action_new = self.RANDOM_STATE_ACTIONS.randint(0, self.number_of_users)
        else:
            action_new = max(range(self.number_of_users),
                             key=lambda action_new: self.action_value_approximator(busy_times_new, action_new))

        self.episode += 1

        if self.episode % 1 == 0:
            self.update_theta(busy_times, action, busy_times_new, action_new, reward, True)
        else:
            self.update_theta(busy_times, action, busy_times_new, action_new, reward, False)

    def get_busy_times(self):
        busy_times = [None] * self.number_of_users
        for user_index, user_deq in enumerate(self.users_queues):
            if len(user_deq) > 0:
                busy_times[user_index] = sum(job.service_rate[user_index] for job in user_deq)
                if user_deq[0].is_busy(self.env.now):
                    busy_times[user_index] -= self.env.now - user_deq[0].started
            else:
                busy_times[user_index] = 0
        return busy_times

    def action_value_approximator(self, states, action):
        value = 0.0
        for i, busy_time in enumerate(states):
            value += busy_time * self.theta[i + action * self.number_of_users]
        return value

    def update_theta(self, old_state, old_action, new_state, new_action, reward, terminal):
        if terminal:
            self.theta += self.alpha * (
                -reward - self.action_value_approximator(old_state, old_action)) * self.gradient(old_state, old_action)
        else:
            self.theta += self.alpha * (-reward + self.gamma * self.action_value_approximator(new_state,
                                                                                              new_action) - self.action_value_approximator(
                old_state, old_action)) * self.gradient(old_state, old_action)

    def gradient(self, states, action):
        gradient_vector = np.zeros(self.number_of_users ** 2)
        for i, busy_time in enumerate(states):
            gradient_vector[i + action * self.number_of_users] = busy_time
        return gradient_vector
