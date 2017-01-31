import numpy as np
from policies import *
from collections import deque


class MC(Policy):
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
        self.name = "LLQP"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.theta = theta
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.history = []
        self.jobs_lateness = []

    def request(self, user_task):
        """
Request method for MC policies. Creates a PolicyJob object and calls for the appropriate evaluation method.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        super().request(user_task)

        average_processing_time = RANDOM_STATE.gamma(
            user_task.service_interval ** 2 / user_task.task_variability,
            user_task.task_variability / user_task.service_interval)

        llqp_job = PolicyJob(user_task)
        llqp_job.request_event = self.env.event()
        llqp_job.arrival = self.env.now

        llqp_job.service_rate = [RANDOM_STATE.gamma(average_processing_time ** 2 / self.worker_variability,
                                                    self.worker_variability / average_processing_time) for
                                 _ in range(self.number_of_users)]

        if self.file_statistics is not None:
            self.save_status()

        self.evaluate(llqp_job)

        if self.file_statistics is not None:
            self.save_status()

        return llqp_job

    def release(self, llqp_job):
        """
Release method for MC policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the released user's queue is not empty, it assigns the next policyjob to be worked.
        :param llqp_job: a policyjob object.
        """
        super().release(llqp_job)
        self.save_job_lateness(llqp_job)
        user_to_release_index = llqp_job.assigned_user

        user_queue_to_free = self.users_queues[user_to_release_index]

        if self.file_statistics is not None:
            self.save_status()

        user_queue_to_free.popleft()

        if self.file_statistics is not None:
            self.save_status()

        if len(user_queue_to_free) > 0:
            next_llqp_job = user_queue_to_free[0]
            next_llqp_job.started = self.env.now
            next_llqp_job.request_event.succeed(next_llqp_job.service_rate[user_to_release_index])

        if self.file_statistics is not None:
            self.save_status()

    def evaluate(self, llqp_job):
        """
Evaluate method for MC policies. Creates a continuous state space which corresponds to the users busy times and follows and epsilon greedy policy approach to optimally choose the best user.
        :param llqp_job: a policyjob object to be assigned.
        """
        busy_times = [None] * self.number_of_users
        for user_index, user_deq in enumerate(self.users_queues):
            if len(user_deq) > 0:
                leftmost_queue_element = user_deq[0]
                busy_times[user_index] = sum(job.service_rate[user_index] for job in user_deq)
                if leftmost_queue_element.is_busy(self.env.now):
                    busy_times[user_index] -= self.env.now - leftmost_queue_element.started
            else:
                busy_times[user_index] = 0

        if RANDOM_STATE.rand() < self.epsilon:
            action = RANDOM_STATE.randint(0, self.number_of_users)
        else:
            action = max(range(self.number_of_users),
                         key=lambda action: self.action_value_approximator(busy_times, action))

        self.history.append((busy_times, action))

        llqp_queue = self.users_queues[action]
        llqp_job.assigned_user = action
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[action])

    def policy_status(self):
        """
Evaluates the current state of the policy. Overrides parent method with MC specific logic.
        :return: returns a list where the first item is the global queue length (in MC always zero) and all subsequent elements are the respective user queues length.
        """
        current_status = [0]
        for i in range(self.number_of_users):
            current_status.append(len(self.users_queues[i]))
        return current_status

    def action_value_approximator(self, states, action):
        """
Value function approximator. Uses the policy theta weight vector and returns for action and states vector an approximated value.
        :param states: list of users busy time.
        :param action: chosen action corresponding to the states.
        :return: a single approximated value.
        """
        value = 0.0
        for i, busy_time in enumerate(states):
            value += busy_time * self.theta[i + action * self.number_of_users]
        return value

    def update_theta(self):
        """
MC method to learn based on its followed trajectory. Evaluates the history list in reverse and for each states-action pair updates its internal theta vector.
        """
        avg_lateness = np.average(self.jobs_lateness)
        for i, (states, action) in enumerate(reversed(self.history)):
            self.theta += self.alpha * (
                self.gamma ** i * -avg_lateness - self.action_value_approximator(states, action)) * self.gradient(
                states,
                action)

    def save_job_lateness(self, policy_job):
        """
Evaluates and appends the job's lateness to a policy global queue.
        :param policy_job: policyjob object passed in each release method.
        """
        job_lateness = policy_job.finished - policy_job.started
        self.jobs_lateness.append(job_lateness)

    def gradient(self, states, action):
        """
For each states-action pair calculates the gradient descent to be used in the theta update function.
        :param states: list of users busy time.
        :param action: chosen action corresponding to the states.
        :return: gradient to be used for updating theta vector towards optimum.
        """
        gradient_vector = np.zeros(self.number_of_users ** 2)
        for i, busy_time in enumerate(states):
            gradient_vector[i + action * self.number_of_users] = busy_time
        return gradient_vector

    def value_function(self):
        """
Creates a list of approximated states-action values.
        :return: list of approximated states-action values to be used as input for trisurf plot.
        """
        value_action = []
        for i, (states, action) in enumerate(self.history):
            qsa_value = self.action_value_approximator(states, action)
            value_action.append((states, qsa_value, action))
        return value_action