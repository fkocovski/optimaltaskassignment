from policies import *
from collections import deque


class K_BATCH_MC_VFA_OPEP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics, batch_size, theta, gamma,alpha,greedy,epsilon):
        """
Initializes a MC policy with VFA.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param file_policy: file object to calculate policy related statistics.
        :param file_statistics: file object to draw the policy evolution.
        :param batch_size: batch size of global queue.
        :param theta: weight vector for VFA.
        :param gamma: discounting factor for rewards.
        :param alpha: step size parameter for the gradient descent method.
        :param greedy: boolean indicating whether the policy should use a greedy approach.
        :param epsilon: parameter for the epsilon greedy approach.
        """
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.batch_size = batch_size
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.greedy = greedy
        self.epsilon = epsilon
        self.name = "{}_BATCH_MC_VFA_OPEP".format(batch_size)
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.batch_queue = []
        self.history = []
        self.rewards = []

    def request(self, user_task):
        """
Request method for MC policies. Creates a PolicyJob object and calls for the appropriate evaluation method.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        k_batch_job = super().request(user_task)

        self.save_status()

        self.batch_queue.append(k_batch_job)

        self.save_status()

        if len(self.batch_queue) == self.batch_size:
            self.evaluate(k_batch_job)

        self.save_status()

        return k_batch_job

    def release(self, k_batch_job):
        """
Release method for MC policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the released user's queue is not empty, it assigns the next policyjob to be worked.
        :param llqp_job: a policyjob object.
        """
        super().release(k_batch_job)

        user_to_release_index = k_batch_job.assigned_user

        user_queue_to_free = self.users_queues[user_to_release_index]

        self.save_status()

        user_queue_to_free.popleft()

        self.save_status()

        if len(user_queue_to_free) > 0:
            next_k_batch_job = user_queue_to_free[0]
            next_k_batch_job.started = self.env.now
            next_k_batch_job.request_event.succeed(next_k_batch_job.service_rate[user_to_release_index])

        self.save_status()

    def evaluate(self, k_batch_job):
        """
Evaluate method for MC policies. Creates a continuous state space which corresponds to the users busy times and follows and epsilon greedy policy approach to optimally choose the best user.
        :param llqp_job: a policyjob object to be assigned.
        """
        state_space = self.state_space(k_batch_job)

        if self.greedy:
            action = max(range(self.number_of_users),
                         key=lambda action: self.q(state_space, action))
        else:
            rnd = np.random.rand()
            if rnd < self.epsilon:
                action = RANDOM_STATE_ACTIONS.randint(0, self.number_of_users)
            else:
                action = max(range(self.number_of_users),
                             key=lambda action: self.q(state_space, action))

        self.history.append((state_space, action))
        self.rewards.append(state_space[action][action] + k_batch_job.service_rate[action])

        user_queue = self.users_queues[action]
        k_batch_job.assigned_user = action
        user_queue.append(k_batch_job)
        # TODO: extend clear method of batch queue for bigger batch queue size
        self.batch_queue.clear()
        if len(self.users_queues[action]) > 0:
            leftmost_k_batch_element = user_queue[0]
            if not leftmost_k_batch_element.is_busy(self.env.now):
                k_batch_job.started = self.env.now
                k_batch_job.request_event.succeed(k_batch_job.service_rate[action])

    def state_space(self, k_batch_job):
        """
Calculates current busy times for users which represent the current state space.
        :return: list which indexes correspond to each user's busy time.
        """
        # wj
        w = [self.env.now - self.batch_queue[j].arrival for j in range(len(self.batch_queue))]

        # pi
        p = [k_batch_job.service_rate[i] for i in range(self.number_of_users)]

        # ai
        current_user_element = [None if len(queue) == 0 else queue[0] for queue in self.users_queues]
        a = [
            0 if current_user_element[i] is None else sum(job.service_rate[i] for job in self.users_queues[i]) for i
            in range(self.number_of_users)]

        for user_index, queue in enumerate(self.users_queues):
            if len(queue) > 0:
                if current_user_element[user_index].is_busy(self.env.now):
                    a[user_index] -= self.env.now - current_user_element[user_index].started

        state_space = np.zeros((self.number_of_users,self.number_of_users+1))


        for i in range(self.number_of_users):
            state_space[i] = a+[p[i]]

        return state_space

    def policy_status(self):
        """
Evaluates the current state of the policy. Overrides parent method with MC specific logic.
        :return: returns a list where the first item is the global queue length (in MC always zero) and all subsequent elements are the respective user queues length.
        """
        current_status = [len(self.batch_queue)]
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
        q = np.dot(features[action], self.theta[action])
        return q

    def update_theta(self):
        """
MC method to learn based on its followed trajectory. Evaluates the history list in reverse and for each states-action pair updates its internal theta vector.
        """
        for i, (states, action) in enumerate(self.history):
            delta = -self.rewards[i] + self.gamma*(max(self.q(states,a) for a in range(self.number_of_users))) - self.q(states,action)
            self.theta += self.alpha * delta*self.features(states,action)

    def features(self, states, action):
        """
Creates features vector for theta update function. For each action it creates a feature vector with busy times and the rest zeroes.
        :param states: busy times.
        :param action: chosen action.
        :return: vector full of zeroes except for action where the busy times are reported.
        """
        features = np.zeros((self.number_of_users,self.number_of_users+1))
        features[action] = states[action]
        return features

    def compose_history(self):
        """
Creates composed history array for per action 3d plot.
        :return: array with required information for 3d per action plot.
        """
        composed_history = []
        for i, (states, action) in enumerate(self.history):
            composed_history.append((states, action, -self.rewards[i]))
        return composed_history
