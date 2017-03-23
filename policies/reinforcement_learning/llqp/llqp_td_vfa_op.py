import randomstate.prng.pcg64 as pcg
import numpy as np
from policies import *
from collections import deque


class LLQP_TD_VFA_OP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, theta, gamma,alpha,greedy):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.greedy = greedy
        self.RANDOM_STATE_ACTIONS = pcg.RandomState(1)
        self.name = "LLQP_TD_VFA_OP"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.history = None

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
        busy_times = self.get_busy_times()

        if self.greedy:
            action = max(range(self.number_of_users),
                         key=lambda action: self.q(busy_times, action))
        else:
            action = self.RANDOM_STATE_ACTIONS.randint(0, self.number_of_users)

        llqp_queue = self.users_queues[action]
        llqp_job.assigned_user = action
        llqp_queue.append(llqp_job)
        llqp_job.assigned = self.env.now
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[action])

        if not self.greedy:
            if self.history is not None:
                self.update_theta(busy_times)

            reward = busy_times[action] + llqp_job.service_rate[action]

            self.history = (busy_times, action,reward)

    def get_busy_times(self):
        busy_times = np.zeros(self.number_of_users)
        for user_index, user_deq in enumerate(self.users_queues):
            if len(user_deq) > 0:
                busy_times[user_index] = sum(job.service_rate[user_index] for job in user_deq)
                if user_deq[0].is_busy(self.env.now):
                    busy_times[user_index] -= self.env.now - user_deq[0].started
            else:
                busy_times[user_index] = 0
        return busy_times

    def q(self, states, action):
        q = np.dot(states, self.theta[action])
        return q

    def update_theta(self,new_busy_times):
        old_busy_times, old_action, reward = self.history
        delta = -reward + self.gamma * (max(self.q(new_busy_times, a) for a in range(self.number_of_users))) - self.q(old_busy_times, old_action)
        self.theta[old_action] += self.alpha * delta * old_busy_times
