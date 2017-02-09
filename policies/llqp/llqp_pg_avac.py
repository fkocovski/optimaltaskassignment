from policies import *
from collections import deque
import math

RANDOM_STATE_PROBABILITIES = np.random.RandomState(1)


class LLQP_PG_AVAC(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics, w, theta, gamma, alpha,
                 beta):
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.name = "LLQP_PG_AVAC"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.w = w
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def request(self, user_task):
        llqp_job = super().request(user_task)

        self.save_status()

        self.evaluate(llqp_job)

        self.save_status()

        return llqp_job

    def release(self, llqp_job):
        super().release(llqp_job)

        user_to_release_index = llqp_job.assigned_user

        user_queue_to_free = self.users_queues[user_to_release_index]

        self.save_status()

        user_queue_to_free.popleft()

        self.save_status()

        if len(user_queue_to_free) > 0:
            next_llqp_job = user_queue_to_free[0]
            next_llqp_job.started = self.env.now
            next_llqp_job.request_event.succeed(next_llqp_job.service_rate[user_to_release_index])

        self.save_status()

    def evaluate(self, llqp_job):
        busy_times = self.get_busy_times()

        probabilities = self.policy_probabilities(busy_times)

        chosen_action = RANDOM_STATE_PROBABILITIES.choice(self.number_of_users, p=probabilities)

        reward = busy_times[chosen_action] + llqp_job.service_rate[chosen_action]

        llqp_queue = self.users_queues[chosen_action]
        llqp_job.assigned_user = chosen_action
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[chosen_action])

        busy_times_new = self.get_busy_times()

        probabilities_new = self.policy_probabilities(busy_times_new)

        chosen_action_new = RANDOM_STATE_PROBABILITIES.choice(self.number_of_users, p=probabilities_new)

        print(self.theta)
        self.learn(busy_times, chosen_action, busy_times_new, chosen_action_new, reward)

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

    def policy_status(self):
        current_status = [0]
        for i in range(self.number_of_users):
            current_status.append(len(self.users_queues[i]))
        return current_status

    def learn(self, old_state, old_action, new_state, new_action, reward):
        delta = -reward + self.gamma * self.q_w(new_state, new_action) - self.q_w(old_state, old_action)
        self.theta += self.alpha * (self.features(old_state, old_action) - sum(
            self.policy_probabilities(old_state)[a] * self.features(old_state, a) for a in
            range(self.number_of_users))) * self.q_w(old_state, old_action)
        self.w += self.beta * delta * self.features(old_state, old_action)

    def policy_probabilities(self, busy_times):
        probabilities = [None] * self.number_of_users
        for action in range(self.number_of_users):
            probabilities[action] = math.exp(np.dot(self.features(busy_times, action), self.theta)) / sum(
                math.exp(np.dot(self.features(busy_times, a), self.theta)) for a in range(self.number_of_users))
        return probabilities

    def features(self, states, action):
        features = np.zeros(self.number_of_users ** 2)
        for act in range(self.number_of_users):
            features[act + action * self.number_of_users] = states[act]
        return features

    def q_w(self, states, action):
        features = self.features(states, action)
        return np.dot(features, self.w)
