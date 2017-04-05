import randomstate.prng.pcg64 as pcg
import numpy as np
from policies import *
from collections import deque


class K_BATCH_MC_VFA(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, batch_size, theta,
                 epsilon, gamma, alpha):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.batch_size = batch_size
        self.theta = theta
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.EPSILON_GREEDY_RANDOM_STATE = pcg.RandomState(1)
        self.name = "{}_BATCH_MC_VFA".format(self.batch_size)
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.batch_queue = []
        self.history = []
        self.rewards = []

    def request(self, user_task, token):
        k_batch_job = super().request(user_task, token)

        self.batch_queue.append(k_batch_job)

        if len(self.batch_queue) == self.batch_size:
            self.evaluate(k_batch_job)

        return k_batch_job

    def release(self, k_batch_job):
        super().release(k_batch_job)

        user_to_release_index = k_batch_job.assigned_user
        queue_to_pop = self.users_queues[user_to_release_index]
        queue_to_pop.popleft()

        if len(queue_to_pop) > 0:
            next_k_batch_job = queue_to_pop[0]
            next_k_batch_job.started = self.env.now
            next_k_batch_job.assigned = self.env.now
            next_k_batch_job.request_event.succeed(next_k_batch_job.service_rate[user_to_release_index])

    def evaluate(self, k_batch_job):
        state_space = self.state_space(k_batch_job)

        if self.EPSILON_GREEDY_RANDOM_STATE.rand() < self.epsilon:
            action = self.EPSILON_GREEDY_RANDOM_STATE.randint(0, self.number_of_users)
        else:
            action = max(range(self.number_of_users),
                         key=lambda action: self.q(state_space, action))

        self.history.append((state_space, action))
        self.rewards.append(state_space[action][action] + k_batch_job.service_rate[action])

        self.users_queues[action].append(k_batch_job)
        k_batch_job.assigned_user = action
        k_batch_job.assigned = self.env.now
        self.batch_queue.clear()
        if len(self.users_queues[action]) > 0:
            leftmost_k_batch_element = self.users_queues[action][0]
            if not leftmost_k_batch_element.is_busy(self.env.now):
                leftmost_k_batch_element.started = self.env.now
                leftmost_k_batch_element.request_event.succeed(leftmost_k_batch_element.service_rate[action])

    def q(self, states, action):
        features = self.features(states, action)
        q = np.dot(features[action], self.theta[action])
        return q

    def features(self, states, action):
        features = np.zeros((self.number_of_users, self.number_of_users + 1))
        features[action] = states[action]
        return features

    def update_theta(self):
        for i, (states, action) in enumerate(self.history):
            self.theta += self.alpha * (-self.discount_rewards(i) - self.q(states, action)) * self.features(states, action)

    def discount_rewards(self, time):
        g = 0.0
        for t, reward in enumerate(self.rewards[time:]):
            g += (self.gamma ** t) * reward
        return g

    def state_space(self, k_batch_job):
        p = [k_batch_job.service_rate[i] for i in range(self.number_of_users)]

        current_user_element = [None if len(queue) == 0 else queue[0] for queue in self.users_queues]
        a = [
            0 if current_user_element[i] is None else sum(job.service_rate[i] for job in self.users_queues[i]) for i
            in range(self.number_of_users)]

        for user_index, queue in enumerate(self.users_queues):
            if len(queue) > 0:
                if current_user_element[user_index].is_busy(self.env.now):
                    a[user_index] -= self.env.now - current_user_element[user_index].started

        state_space = np.zeros((self.number_of_users, self.number_of_users + 1))

        for i in range(self.number_of_users):
            state_space[i] = a + [p[i]]

        return state_space
