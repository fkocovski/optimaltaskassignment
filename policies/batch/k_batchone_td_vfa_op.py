from policies import *


class K_BATCHONE_TD_VFA_OP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics,batch_size,theta,gamma,alpha,greedy):
        """
Initializes a KBatchOne policy.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param batch_size: the batch size of the global queue.
        :param file_policy: file object to calculate policy related statistics.
        :param file_statistics: file object to draw the policy evolution.
        """
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.batch_size = batch_size
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.greedy = greedy
        self.name = "{}BATCHONE_TD_VFA_OP".format(self.batch_size)
        self.assigned_job_to_user = [None] * self.number_of_users
        self.batch_queue = []
        self.history = []

    def request(self, user_task):
        """
Request method for KBatchOne policies. Creates a PolicyJob object and calls for the appropriate evaluation method with the corresponding solver.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        k_batchone_job = super().request(user_task)

        self.save_status()

        self.batch_queue.append(k_batchone_job)

        self.save_status()

        if len(self.batch_queue) >= self.batch_size:
            self.evaluate()

        self.save_status()

        return k_batchone_job

    def release(self, k_batchone_job):
        """
Release method for KBatchOne policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the global queue's size is greater than the batch size, it calls the appropriate evaluation method again.
        :param k_batchone_job: a policyjob object.
        """
        super().release(k_batchone_job)

        self.save_status()

        user_to_release_index = k_batchone_job.assigned_user
        self.assigned_job_to_user[user_to_release_index] = None

        self.save_status()

        if len(self.batch_queue) >= self.batch_size:
            self.evaluate()

        self.save_status()

    def evaluate(self):
        """
Evaluate method for KBatchOne policies. Sets the required variables by the solver then calls the appropriate solver assigned and implements its returned solution.
        """
        k_batchone_job = self.batch_queue[0]

        state_space = self.state_space(k_batchone_job)

        if self.greedy:
            action = max(range(self.number_of_users),
                         key=lambda action: self.q(state_space, action))
        else:
            action = RANDOM_STATE_ACTIONS.randint(0, self.number_of_users)

        if self.assigned_job_to_user[action] is None:
            reward = state_space[action][action] + k_batchone_job.service_rate[action]
            self.history.append((state_space,action,reward))
            self.batch_queue[0] = None
            self.assigned_job_to_user[action] = k_batchone_job
            k_batchone_job.assigned_user = action
            k_batchone_job.started = self.env.now
            k_batchone_job.request_event.succeed(k_batchone_job.service_rate[action])
        self.batch_queue = [job for job in self.batch_queue if job is not None]

        if not self.greedy:
            if len(self.history) == 2:
                self.update_theta()

    def state_space(self,k_batchone_job):
        
        # pi
        p = [k_batchone_job.service_rate[i] for i in range(self.number_of_users)]

        # ai
        current_user_element = [self.assigned_job_to_user[i] for i in range(self.number_of_users)]
        a = [0 if current_user_element[i] is None else current_user_element[i].will_finish() - self.env.now for i
             in range(self.number_of_users)]

        state_space = np.zeros((self.number_of_users,self.number_of_users+1))

        for i in range(self.number_of_users):
            state_space[i] = a + [p[i]]

        return state_space

    def policy_status(self):
        """
Evaluates the current state of the policy. Overrides parent method with KBatchOne specific logic.
        :return: returns a list where the first item is the global queue length and all subsequent elements are the respective user queue length.
        """
        current_status = [len(self.batch_queue)]
        for i in range(self.number_of_users):
            if self.assigned_job_to_user[i] is None:
                current_status.append(0)
            else:
                current_status.append(1)
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
        state = self.history[0][0]
        action = self.history[0][1]
        reward = self.history[0][2]
        future_state = self.history[1][0]
        delta = -reward + self.gamma * (max(self.q(future_state, a) for a in range(self.number_of_users))) - self.q(
            state, action)
        self.theta += self.alpha * delta * self.features(state, action)
        self.history.clear()

    def features(self, states, action):
        """
Creates features vector for theta update function. For each action it creates a feature vector with busy times and the rest zeroes.
        :param states: busy times.
        :param action: chosen action.
        :return: vector full of zeroes except for action where the busy times are reported.
        """
        features = np.zeros((self.number_of_users, self.number_of_users + 1))
        features[action] = states[action]
        return features
