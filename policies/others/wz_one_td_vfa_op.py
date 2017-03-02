from policies import *
import itertools


class WZ_ONE_TD_VFA_OP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics, theta, gamma, alpha,
                 greedy, wait_size):
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.greedy = greedy
        self.wait_size = wait_size
        self.name = "WZ_ONE_TD_VFA_OP"
        self.user_slot = [None] * self.number_of_users
        self.batch_queue = []
        self.history = None

    def request(self, user_task):
        wz_one_job = super().request(user_task)

        self.save_status()

        self.batch_queue.append(wz_one_job)

        self.save_status()

        if len(self.batch_queue) >= self.wait_size:
            self.evaluate()

        self.save_status()

        return wz_one_job

    def release(self, wz_one_job):
        super().release(wz_one_job)

        user_to_release_index = wz_one_job.assigned_user

        self.save_status()

        self.user_slot[user_to_release_index] = None

        self.save_status()

        if len(self.batch_queue) >= self.wait_size:
            self.evaluate()

        self.save_status()

    def evaluate(self):
        state_space, combinations = self.state_space()

        if self.greedy:
            action = max(range(self.number_of_users ** self.wait_size),
                         key=lambda a: self.q(state_space, a))
        else:
            action = RANDOM_STATE_ACTIONS.randint(0, self.number_of_users ** self.wait_size)

        for job_index, user_index in enumerate(combinations[action]):
            if self.user_slot[user_index] is None:
                wz_one_job = self.batch_queue[job_index]
                self.batch_queue[job_index] = None
                self.user_slot[user_index] = wz_one_job
                wz_one_job.assigned_user = user_index
                wz_one_job.started = self.env.now
                wz_one_job.request_event.succeed(wz_one_job.service_rate[user_index])
        self.batch_queue = [job for job in self.batch_queue if job is not None]

        if not self.greedy:
            if self.history is not None:
                self.update_theta(state_space)

        self.history = (state_space, action, combinations)

    def state_space(self):
        # wj
        w = [self.env.now - self.batch_queue[j].arrival for j in range(self.wait_size)]

        # pij
        p = [[self.batch_queue[j].service_rate[i] for j in range(self.wait_size)] for i in
             range(self.number_of_users)]

        # ai
        a = [0 if self.user_slot[i] is None else self.user_slot[i].will_finish() - self.env.now for i
             in range(self.number_of_users)]

        state_space = np.zeros((self.number_of_users ** self.wait_size,self.number_of_users + 2*self.wait_size))

        combinations = list(itertools.product(range(self.number_of_users),repeat=self.wait_size))

        for i, combination in enumerate(combinations):
            state_space[i] = w + a + [p[user_index][job_index] for job_index, user_index in enumerate(combination)]

        return state_space, combinations

    def policy_status(self):
        current_status = [len(self.batch_queue)]
        for i in range(self.number_of_users):
            if self.user_slot[i] is None:
                current_status.append(0)
            else:
                current_status.append(1)
        return current_status

    def q(self, states, action):
        q = np.dot(states[action], self.theta[action])
        return q

    def update_theta(self, new_state_space):
        old_state_space, old_action, old_combinations = self.history
        reward = self.reward(old_state_space, old_action, old_combinations)
        delta = -reward + self.gamma * (max(self.q(new_state_space, a) for a in range(self.number_of_users ** self.wait_size))) - self.q(old_state_space, old_action)
        self.theta[old_action] += self.alpha * delta * old_state_space[old_action]

    def reward(self, state_space, action, combinations):
        reward = 0.0
        busy_times = [state_space[action][self.wait_size + i] for i in range(self.number_of_users)]
        for job_index, user_index in enumerate(combinations[action]):
            reward += state_space[action][job_index] + busy_times[user_index] + state_space[action][2*self.wait_size + job_index]
            busy_times[user_index] += state_space[action][2 * self.wait_size + job_index]
        return reward
