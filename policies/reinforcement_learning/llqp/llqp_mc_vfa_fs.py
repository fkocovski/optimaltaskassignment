import numpy as np
import randomstate.prng.pcg64 as pcg
from policies import *
from collections import deque


class LLQP_MC_VFA_FS(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, theta, epsilon, gamma,
                 alpha):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.theta = theta
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.RANDOM_STATE_ACTIONS = pcg.RandomState(1)
        self.name = "LLQP_MC_VFA_FS"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.history = []
        self.rewards = []

    def request(self, user_task,token):
        llqp_job = PolicyJob(user_task,token)
        llqp_job.request_event = self.env.event()
        llqp_job.arrival = self.env.now

        llqp_job.service_rate = [user_task.service_interval for _ in range(self.number_of_users)]

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
                         key=lambda action: self.q(busy_times, action))


        self.history.append((busy_times, action))
        self.rewards.append(busy_times[action] + llqp_job.service_rate[action])

        llqp_queue = self.users_queues[action]
        llqp_job.assigned_user = action
        llqp_job.assigned = self.env.now
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[action])

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

    def q(self, states, action):
        features = self.features(states, action)
        return np.dot(features, self.theta)

    def update_theta(self):
        for i, (states, action) in enumerate(self.history):
            self.theta += self.alpha * (self.discount_rewards(i) - self.q(states, action)) * self.features(states,action)

    def discount_rewards(self, time):
        g = 0.0
        tmp_rewards = self.rewards[time:]
        for t,rwd in enumerate(tmp_rewards):
            g += (self.gamma ** t) * rwd
        return -g

    def features(self, states, action):
        features = np.zeros(self.number_of_users ** 2)
        for act in range(self.number_of_users):
            features[act + action * self.number_of_users] = states[act]
        return features

    def normalize_theta(self):
        self.theta /= np.linalg.norm(self.theta)

    def compose_history(self):
        composed_history = []
        for i, (states, action) in enumerate(self.history):
            composed_history.append((states, action, self.discount_rewards(i)))
        return composed_history

    def regression_fit(self):
        s = []
        g = []
        for i, (states, action) in enumerate(self.history):
            s.append((states, action))
            g.append((self.discount_rewards(i), action))
        s_one = [s1[0] for s1 in s if s1[1] == 0]
        s_two = [s2[0] for s2 in s if s2[1] == 1]
        g_one = [g1[0] for g1 in g if g1[1] == 0]
        g_two = [g2[0] for g2 in g if g2[1] == 1]

        dependent_one = np.dot(np.array(s_one).T, np.array(g_one))
        dependent_two = np.dot(np.array(s_two).T, np.array(g_two))
        coefficient_one = np.dot(np.array(s_one).T, np.array(s_one))
        coefficient_two = np.dot(np.array(s_two).T, np.array(s_two))


        theta_one = np.linalg.solve(coefficient_one, dependent_one)
        theta_two = np.linalg.solve(coefficient_two, dependent_two)
        arr = np.concatenate((theta_one, theta_two))

        self.theta[:] = arr
