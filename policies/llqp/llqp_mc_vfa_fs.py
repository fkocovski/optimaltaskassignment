from policies import *
from collections import deque


class LLQP_MC_VFA_FS(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics, theta, epsilon, gamma,
                 alpha):
        """
Initializes a MC policy with VFA.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param file_policy: file object to calculate policy related statistics.
        :param file_statistics: file object to draw the policy evolution.
        :param theta: weight vector for VFA.
        :param epsilon: parameter for the epsilon greedy approach.
        :param gamma: discounting factor for rewards.
        :param alpha: step size parameter for the gradient descent method.
        """
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.name = "LLQP_MC_VFA_FS"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.theta = theta
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.history = []
        self.rewards = []

    def request(self, user_task):
        """
Request method for MC policies. Creates a PolicyJob object and calls for the appropriate evaluation method.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        # super().request(user_task)

        llqp_job = PolicyJob(user_task)
        llqp_job.request_event = self.env.event()
        llqp_job.arrival = self.env.now

        llqp_job.service_rate = [user_task.service_interval for _ in range(self.number_of_users)]

        self.save_status()

        self.evaluate(llqp_job)

        self.save_status()

        return llqp_job

    def release(self, llqp_job):
        """
Release method for MC policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the released user's queue is not empty, it assigns the next policyjob to be worked.
        :param llqp_job: a policyjob object.
        """
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
        """
Evaluate method for MC policies. Creates a continuous state space which corresponds to the users busy times and follows and epsilon greedy policy approach to optimally choose the best user.
        :param llqp_job: a policyjob object to be assigned.
        """
        busy_times = self.get_busy_times()
        if RANDOM_STATE_ACTIONS.rand() < self.epsilon:
            action = RANDOM_STATE_ACTIONS.randint(0, self.number_of_users)
        else:
            action = max(range(self.number_of_users),
                         key=lambda action: self.q(busy_times, action))

        self.history.append((busy_times, action))
        self.rewards.append(busy_times[action] + llqp_job.service_rate[action])

        llqp_queue = self.users_queues[action]
        llqp_job.assigned_user = action
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[action])

    def get_busy_times(self):
        """
Calculates current busy times for users which represent the current state space.
        :return: list which indexes correspond to each user's busy time.
        """
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
        """
Evaluates the current state of the policy. Overrides parent method with MC specific logic.
        :return: returns a list where the first item is the global queue length (in MC always zero) and all subsequent elements are the respective user queues length.
        """
        current_status = [0]
        for i in range(self.number_of_users):
            current_status.append(len(self.users_queues[i]))
        return current_status

    def q(self, states, action):
        """
Value function approximator. Uses the policy theta weight vector and returns for action and states vector an approximated value.
        :param states: list of users busy time.
        :param action: chosen action corresponding to the states.
        :return: a single approximated value.
        """
        features = self.features(states, action)
        return np.dot(features, self.theta)

    def update_theta(self):
        """
MC method to learn based on its followed trajectory. Evaluates the history list in reverse and for each states-action pair updates its internal theta vector.
        """
        for i, (states, action) in enumerate(self.history):
            self.theta += self.alpha * (self.discount_rewards(i) - self.q(states, action)) * self.features(states,action)

    def discount_rewards(self, time):
        """
Discount rewards for one MC episode.
        """
        g = 0.0
        tmp_rewards = self.rewards[time:]
        for t,rwd in enumerate(tmp_rewards):
            g += (self.gamma ** t) * rwd
        return -g

    def features(self, states, action):
        """
Creates features vector for theta update function. For each action it creates a feature vector with busy times and the rest zeroes.
        :param states: busy times.
        :param action: chosen action.
        :return: vector full of zeroes except for action where the busy times are reported.
        """
        features = np.zeros(self.number_of_users ** 2)
        for act in range(self.number_of_users):
            features[act + action * self.number_of_users] = states[act]
        return features

    def normalize_theta(self):
        """
Normalizes theta vector.
        :return: normalized theta vector.
        """
        self.theta /= np.linalg.norm(self.theta)

    def compose_history(self):
        composed_history = []
        for i, (states, action) in enumerate(self.history):
            composed_history.append((states,action,self.discount_rewards(i)))
        return composed_history