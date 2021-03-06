import randomstate.prng.pcg64 as pcg
import numpy as np
from policies import *
from collections import deque


class WZ_TD_VFA_OP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, theta, gamma, alpha,
                 greedy, wait_size,seed):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.greedy = greedy
        self.wait_size = wait_size
        self.RANDOM_STATE_ACTIONS = pcg.RandomState(seed)
        self.name = "{}_WZ_TD_VFA_OP".format(self.wait_size)
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.batch_queue = []
        self.history = None

    def request(self, user_task, token):
        wz_job = super().request(user_task, token)

        self.batch_queue.append(wz_job)

        if len(self.batch_queue) == self.wait_size:
            self.evaluate()

        return wz_job

    def release(self, wz_job):
        super().release(wz_job)

        user_to_release_index = wz_job.assigned_user
        user_queue_to_free = self.users_queues[user_to_release_index]
        user_queue_to_free.popleft()

        if len(user_queue_to_free) > 0:
            next_wz_job = user_queue_to_free[0]
            next_wz_job.started = self.env.now
            next_wz_job.assigned = self.env.now
            next_wz_job.request_event.succeed(next_wz_job.service_rate[user_to_release_index])

    def evaluate(self):
        state_space, combinations, w,p,a = self.state_space()

        if self.greedy:
            action = max(range(self.number_of_users ** self.wait_size),
                         key=lambda a: self.q(state_space, a))
        else:
            action = self.RANDOM_STATE_ACTIONS.randint(0, self.number_of_users ** self.wait_size)

        for job_index, user_index in enumerate(combinations[action]):
            wz_queue = self.users_queues[user_index]
            wz_job = self.batch_queue[job_index]
            wz_job.assigned_user = user_index
            wz_job.assigned = self.env.now
            wz_queue.append(wz_job)
            leftmost = wz_queue[0]
            if not leftmost.is_busy(self.env.now):
                wz_job.started = self.env.now
                wz_job.request_event.succeed(wz_job.service_rate[user_index])
        self.batch_queue.clear()

        if not self.greedy:
            if self.history is not None:
                self.update_theta(state_space)

        self.history = (state_space, action, combinations, w,p,a)

    def state_space(self):
        w = [self.env.now - self.batch_queue[j].arrival for j in range(self.wait_size)]

        p = [[self.batch_queue[j].service_rate[i] for j in range(self.wait_size)] for i in
             range(self.number_of_users)]

        current_user_element = [None if len(queue) == 0 else queue[0] for queue in self.users_queues]
        a = [
            0 if current_user_element[i] is None else sum(job.service_rate[i] for job in self.users_queues[i]) for i
            in range(self.number_of_users)]

        for user_index, queue in enumerate(self.users_queues):
            if len(queue) > 0:
                if current_user_element[user_index].is_busy(self.env.now):
                    a[user_index] -= self.env.now - current_user_element[user_index].started

        state_space = np.zeros((self.number_of_users ** self.wait_size, self.number_of_users + self.wait_size))

        combinations = list(itertools.product(range(self.number_of_users), repeat=self.wait_size))

        for i, combination in enumerate(combinations):
            state_space[i] = a + [p[user_index][job_index] for job_index, user_index in enumerate(combination)]

        return state_space, combinations, w,p,a

    def q(self, states, action):
        q = np.dot(states[action], self.theta[action])
        return q

    def update_theta(self, new_state_space):
        old_state_space, old_action, old_combinations, w,p,a = self.history
        delta = -self.reward(old_state_space, old_action, old_combinations, w,p,a) + self.gamma * (
            max(self.q(new_state_space, a) for a in range(self.number_of_users ** self.wait_size))) - self.q(
            old_state_space, old_action)
        self.theta += self.alpha * delta * old_state_space[old_action]

    def reward(self, state_space, action, combinations, w,p,a):
        reward = 0.0
        busy_times = [a[i] for i in range(self.number_of_users)]
        for job_index, user_index in enumerate(combinations[action]):
            reward += w[job_index] + p[user_index][job_index] + busy_times[user_index]
            busy_times[user_index] += p[user_index][job_index]
        return reward
