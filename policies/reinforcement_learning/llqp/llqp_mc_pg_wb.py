import numpy as np
import randomstate.prng.pcg64 as pcg
from policies import *
from collections import deque


class LLQP_MC_PG_WB(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, w, theta, gamma, alpha,
                 beta):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.w = w
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.RANDOM_STATE_PROBABILITIES = pcg.RandomState(1)
        self.name = "LLQP_MC_PG_WB"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.beta = beta
        self.history = []
        self.jobs_lateness = []

    def request(self, user_task, token):
        llqp_job = super().request(user_task, token)

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
            next_llqp_job.started = self.env.now
            next_llqp_job.request_event.succeed(next_llqp_job.service_rate[user_to_release_index])

    def evaluate(self, llqp_job):
        busy_times = self.get_busy_times()

        probabilities = self.policy_probabilities(busy_times)

        chosen_action = self.RANDOM_STATE_PROBABILITIES.choice(self.number_of_users, p=probabilities)

        self.history.append((busy_times, chosen_action))
        self.jobs_lateness.append(busy_times[chosen_action] + llqp_job.service_rate[chosen_action])

        llqp_queue = self.users_queues[chosen_action]
        llqp_job.assigned_user = chosen_action
        llqp_job.assigned = self.env.now
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[chosen_action])

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

    def policy_probabilities(self, busy_times):
        probabilities = [None] * self.number_of_users
        for action in range(self.number_of_users):
            probabilities[action] = np.exp(self.action_value_approximator(busy_times, action)) / sum(
                np.exp(self.action_value_approximator(busy_times, a)) for a in range(self.number_of_users))
        return probabilities

    def action_value_approximator(self, states, action):
        value = 0.0
        for i, busy_time in enumerate(states):
            value += busy_time * self.theta[i + action * self.number_of_users]
        return value

    def state_value_approximator(self, states):
        value = 0.0
        for i, state in enumerate(states):
            value += self.w[i] * state
        return value

    def state_gradient(self, states):
        state_gradient = np.zeros(self.number_of_users)
        for i, state in enumerate(states):
            state_gradient[i] = state
        return state_gradient

    def learn(self):
        for i, (states, action) in enumerate(self.history):
            disc_reward = -self.discount_rewards(i)
            delta = disc_reward - self.state_value_approximator(states)
            self.w += self.beta * delta * self.state_gradient(states)
            self.theta += self.alpha * self.gamma ** i * delta * (self.features(states, action) - sum(
                self.policy_probabilities(states)[a] * self.features(states, a) for a in range(self.number_of_users)))

    def features(self, states, action):
        features = np.zeros(self.number_of_users ** 2)
        for act in range(self.number_of_users):
            features[act + action * self.number_of_users] = states[act]
        return features

    def discount_rewards(self, time):
        g = 0.0
        for t in range(time + 1):
            g += (self.gamma ** t) * self.jobs_lateness[t]
        return g
