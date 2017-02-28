from policies import *
from collections import deque
import itertools


class WZ_TD_VFA_OP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics, theta, gamma, alpha,
                 greedy, wait_size):
        """
Initializes a WZ policy with VFA using TD and an OP as update method.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param file_policy: file object to calculate policy related statistics.
        :param file_statistics: file object to draw the policy evolution.
        :param theta: weight vector for VFA.
        :param gamma: discounting factor for rewards.
        :param alpha: step size parameter for the gradient descent method.
        :param greedy: boolean indicating whether the policy should use a greedy approach.
        """
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.name = "WZ_TD_VFA_OP"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.greedy = greedy
        self.wait_size = wait_size
        self.batch_queue = []
        self.history = None

    def request(self, user_task):
        """
Request method for MC policies. Creates a PolicyJob object and calls for the appropriate evaluation method.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        llqp_job = super().request(user_task)

        self.save_status()

        self.batch_queue.append(llqp_job)

        if len(self.batch_queue) == self.wait_size:
            self.evaluate()

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

    def evaluate(self):
        """
Evaluate method for MC policies. Creates a continuous state space which corresponds to the users busy times and follows and epsilon greedy policy approach to optimally choose the best user.
        :param llqp_job: a policyjob object to be assigned.
        """

        state_space, combinations = self.state_space()

        if self.greedy:
            action = max(range(self.number_of_users),
                         key=lambda action: self.q(state_space, action))
        else:
            action = RANDOM_STATE_ACTIONS.randint(0, self.number_of_users ** self.wait_size)

        for job_index, user_index in enumerate(combinations[action]):
            llqp_queue = self.users_queues[user_index]
            llqp_job = self.batch_queue[job_index]
            llqp_job.assigned_user = user_index
            llqp_queue.append(llqp_job)
            leftmost_llqp_queue_element = llqp_queue[0]
            if not leftmost_llqp_queue_element.is_busy(self.env.now):
                llqp_job.started = self.env.now
                llqp_job.request_event.succeed(llqp_job.service_rate[user_index])
        self.batch_queue.clear()

        if not self.greedy:
            if self.history is not None:
                self.update_theta(state_space)

        self.history = (state_space,action,combinations)

    def state_space(self):
        """
Calculates current busy times for users which represent the current state space.
        :return: list which indexes correspond to each user's busy time.
        """
        # wj
        w = [self.env.now - self.batch_queue[j].arrival for j in range(len(self.batch_queue))]

        # pij
        p = [[self.batch_queue[j].service_rate[i] for j in range(len(self.batch_queue))] for i in
             range(self.number_of_users)]

        # ai
        current_user_element = [None if len(queue) == 0 else queue[0] for queue in self.users_queues]
        a = [
            0 if current_user_element[i] is None else sum(job.service_rate[i] for job in self.users_queues[i]) for i
            in range(self.number_of_users)]

        for user_index, queue in enumerate(self.users_queues):
            if len(queue) > 0:
                if current_user_element[user_index].is_busy(self.env.now):
                    a[user_index] -= self.env.now - current_user_element[user_index].started

        state_space = np.zeros((self.number_of_users + self.wait_size, self.number_of_users ** self.wait_size))

        combinations = list(itertools.product(*[range(self.number_of_users)] * self.wait_size))

        for i, combination in enumerate(combinations):
            state_space[i] = a + [p[user_index][job_index] for job_index, user_index in enumerate(combination)]

        return state_space, combinations

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
        q = np.dot(states[action], self.theta[action])
        return q

    def update_theta(self, new_state_space):
        """
MC method to learn based on its followed trajectory. Evaluates the history list in reverse and for each states-action pair updates its internal theta vector.
        """
        old_state_space = self.history[0]
        old_action = self.history[1]
        old_combinations = self.history[2]
        delta = -self.reward(old_state_space,old_action,old_combinations) + self.gamma * (max(self.q(new_state_space, a) for a in range(self.number_of_users ** self.wait_size))) - self.q(old_state_space, old_action)
        self.theta[old_action] += self.alpha * delta * old_state_space[old_action]

    def reward(self,state_space,action,combinations):
        reward = 0.0
        for job_index,user_index in enumerate(combinations[action]):
            reward += state_space[action][user_index] + state_space[action][self.wait_size+job_index]
        print(state_space,action,combinations[action],reward)
        return reward