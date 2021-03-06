import randomstate.prng.pcg64 as pcg
import numpy as np
from policies import *


class K_BATCHONE_TD_VFA_OP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, batch_size, theta, gamma, alpha, greedy,
                 seed):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.batch_size = batch_size
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.greedy = greedy
        self.EPSILON_GREEDY_RANDOM_STATE = pcg.RandomState(seed)
        self.name = "{}_BATCHONE_TD_VFA_OP".format(self.batch_size)
        self.assigned_job_to_user = [None] * self.number_of_users
        self.batch_queue = []
        self.history = None

    def request(self, user_task, token):
        k_batchone_job = super().request(user_task, token)

        self.batch_queue.append(k_batchone_job)

        if len(self.batch_queue) >= self.batch_size:
            self.evaluate()

        return k_batchone_job

    def release(self, k_batchone_job):
        super().release(k_batchone_job)

        user_to_release_index = k_batchone_job.assigned_user
        self.assigned_job_to_user[user_to_release_index] = None

        if len(self.batch_queue) >= self.batch_size:
            self.evaluate()

    def evaluate(self):
        k_batchone_job = self.batch_queue[0]

        state_space = self.state_space(k_batchone_job)

        if self.greedy:
            action = max(range(self.number_of_users),
                         key=lambda action: self.q(state_space, action))
        else:
            action = self.EPSILON_GREEDY_RANDOM_STATE.randint(0, self.number_of_users)

        reward = state_space[action][action] + k_batchone_job.service_rate[action]

        if self.assigned_job_to_user[action] is None:
            self.batch_queue[0] = None
            self.assigned_job_to_user[action] = k_batchone_job
            k_batchone_job.assigned_user = action
            k_batchone_job.assigned = self.env.now
            k_batchone_job.started = self.env.now
            k_batchone_job.request_event.succeed(k_batchone_job.service_rate[action])
        self.batch_queue = [job for job in self.batch_queue if job is not None]

        if not self.greedy:
            if self.history is not None:
                self.update_theta(state_space)

        self.history = (state_space, action, reward)

    def state_space(self, k_batchone_job):
        p = [k_batchone_job.service_rate[i] for i in range(self.number_of_users)]

        current_user_element = [self.assigned_job_to_user[i] for i in range(self.number_of_users)]
        a = [0 if current_user_element[i] is None else current_user_element[i].will_finish() - self.env.now for i
             in range(self.number_of_users)]

        state_space = np.zeros((self.number_of_users, self.number_of_users + 1))

        for i in range(self.number_of_users):
            state_space[i] = a + [p[i]]

        return state_space

    def q(self, states, action):
        features = self.features(states, action)
        q = np.dot(features[action], self.theta[action])
        return q

    def update_theta(self,future_state):
        state,action,reward = self.history
        delta = -reward + self.gamma * (max(self.q(future_state, a) for a in range(self.number_of_users))) - self.q(
            state, action)
        self.theta += self.alpha * delta * self.features(state,action)

    def features(self, states, action):
        features = np.zeros((self.number_of_users, self.number_of_users + 1))
        features[action] = states[action]
        return features