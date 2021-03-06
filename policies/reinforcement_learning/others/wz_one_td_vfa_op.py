import randomstate.prng.pcg64 as pcg
import numpy as np
from policies import *

class WZ_ONE_TD_VFA_OP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, theta, gamma, alpha,
                 greedy, wait_size,seed):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.greedy = greedy
        self.wait_size = wait_size
        self.RANDOM_STATE_ACTIONS = pcg.RandomState(seed)
        self.name = "{}_WZ_ONE_TD_VFA_OP".format(self.wait_size)
        self.user_slot = [None] * self.number_of_users
        self.batch_queue = []
        self.history = None
        # TODO only for statistical checks. Remove when finished
        self.total_evals = 0
        self.actually_free = 0

    def request(self, user_task,token):
        wz_one_job = super().request(user_task,token)

        self.batch_queue.append(wz_one_job)

        if len(self.batch_queue) >= self.wait_size:
            self.evaluate()

        return wz_one_job

    def release(self, wz_one_job):
        super().release(wz_one_job)

        user_to_release_index = wz_one_job.assigned_user

        self.user_slot[user_to_release_index] = None

        if len(self.batch_queue) >= self.wait_size:
            self.evaluate()

    def evaluate(self):

        state_space, combinations,w,p,a = self.state_space()

        if self.greedy:
            action = max(range(self.number_of_users ** self.wait_size),
                         key=lambda a: self.q(state_space, a))
        else:
            action = self.RANDOM_STATE_ACTIONS.randint(0, self.number_of_users ** self.wait_size)

        for job_index, user_index in enumerate(combinations[action]):
            # TODO only for statistical checks. Remove when finished
            if self.greedy:
                self.total_evals += 1
            if self.user_slot[user_index] is None:
                wz_one_job = self.batch_queue[job_index]
                self.batch_queue[job_index] = None
                self.user_slot[user_index] = wz_one_job
                wz_one_job.assigned_user = user_index
                wz_one_job.assigned = self.env.now
                wz_one_job.started = self.env.now
                wz_one_job.request_event.succeed(wz_one_job.service_rate[user_index])
            # TODO only for statistical checks. Remove when finished
            else:
                if self.greedy:
                    if None in self.user_slot:
                        self.actually_free += 1

        self.batch_queue = [job for job in self.batch_queue if job is not None]

        if not self.greedy:
            if self.history is not None:
                self.update_theta(state_space)

        self.history = (state_space, action, combinations,w,p,a)

    def state_space(self):
        w = [self.env.now - self.batch_queue[j].arrival for j in range(self.wait_size)]

        p = [[self.batch_queue[j].service_rate[i] for j in range(self.wait_size)] for i in
             range(self.number_of_users)]

        a = [0 if self.user_slot[i] is None else self.user_slot[i].will_finish() - self.env.now for i
             in range(self.number_of_users)]

        state_space = np.zeros((self.number_of_users ** self.wait_size, self.number_of_users + 2 * self.wait_size))

        combinations = list(itertools.product(range(self.number_of_users), repeat=self.wait_size))

        for i, combination in enumerate(combinations):
            state_space[i] = w + a + [p[user_index][job_index] for job_index, user_index in enumerate(combination)]

        return state_space, combinations,w,p,a

    def q(self, states, action):
        q = np.dot(states[action], self.theta[action])
        return q

    def update_theta(self, new_state_space):
        old_state_space, old_action, old_combinations,w,p,a = self.history
        reward = self.reward(old_state_space, old_action, old_combinations,w,p,a)
        delta = -reward + self.gamma * (
        max(self.q(new_state_space, a) for a in range(self.number_of_users ** self.wait_size))) - self.q(
            old_state_space, old_action)
        self.theta[old_action] += self.alpha * delta * old_state_space[old_action]

    def reward(self, state_space, action, combinations,w,p,a):
        reward = 0.0
        busy_times = [a[i] for i in range(self.number_of_users)]
        for job_index, user_index in enumerate(combinations[action]):
            reward += w[job_index] + busy_times[user_index] + p[user_index][job_index]
            busy_times[user_index] += p[user_index][job_index]
        return reward
