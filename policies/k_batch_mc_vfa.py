import numpy as np
from policies import *
from collections import deque


class KBatchMcVfa(Policy):
    def __init__(self, env, number_of_users, worker_variability, batch_size, file_policy, file_statistics, theta,
                 epsilon, gamma, alpha):
        """
Initializes a KBatch policy.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param batch_size: the batch size of the global queue.
        :param file_policy: file object to calculate policy related statistics.
        :param file_statistics: file object to draw the policy evolution.
        """
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.batch_size = batch_size
        self.name = "{}Batch_mc_vfa".format(self.batch_size)
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.batch_queue = []
        self.theta = theta
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.history = []
        self.jobs_lateness = []
        self.jobs_service_times = []

    def request(self, user_task):
        """
Request method for KBatch policies. Creates a PolicyJob object and calls for the appropriate evaluation method with the corresponding solver.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        super().request(user_task)

        average_processing_time = RANDOM_STATE.gamma(
            user_task.service_interval ** 2 / user_task.task_variability,
            user_task.task_variability / user_task.service_interval)

        k_batch_job = PolicyJob(user_task)
        k_batch_job.request_event = self.env.event()
        k_batch_job.arrival = self.env.now
        k_batch_job.service_rate = [RANDOM_STATE.gamma(average_processing_time ** 2 / self.worker_variability,
                                                       self.worker_variability / average_processing_time) for
                                    _ in range(self.number_of_users)]

        self.save_status()

        self.batch_queue.append(k_batch_job)

        self.save_status()

        if len(self.batch_queue) == self.batch_size:
            self.evaluate(k_batch_job)

        self.save_status()

        return k_batch_job

    def release(self, k_batch_job):
        """
Release method for KBatch policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the released user's queue is not empty, it assigns the next policyjob to be worked.
        :param k_batch_job: a policyjob object.
        """
        super().release(k_batch_job)

        self.save_rewards(k_batch_job)

        self.save_status()

        user_to_release_index = k_batch_job.assigned_user
        queue_to_pop = self.users_queues[user_to_release_index]
        queue_to_pop.popleft()

        self.save_status()

        if len(queue_to_pop) > 0:
            next_k_batch_job = queue_to_pop[0]
            next_k_batch_job.started = self.env.now
            next_k_batch_job.request_event.succeed(next_k_batch_job.service_rate[user_to_release_index])

        self.save_status()

    def evaluate(self, k_batch_job):
        """
Evaluate method for KBatch policies. Sets the required variables by the solver then calls the appropriate solver assigned and implements its returned solution.
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

        states = a + p

        if RANDOM_STATE.rand() < self.epsilon:
            action = RANDOM_STATE.randint(0, self.number_of_users)
        else:
            action = max(range(self.number_of_users),
                         key=lambda action: self.action_value_approximator(states, action))

        self.history.append((states, action))

        self.users_queues[action].append(k_batch_job)
        k_batch_job.assigned_user = action
        self.batch_queue.clear()
        if len(self.users_queues[action]) > 0:
            leftmost_llpq_element = self.users_queues[action][0]
            if not leftmost_llpq_element.is_busy(self.env.now):
                leftmost_llpq_element.started = self.env.now
                leftmost_llpq_element.request_event.succeed(leftmost_llpq_element.service_rate[action])

    def action_value_approximator(self, states, action):
        """
Value function approximator. Uses the policy theta weight vector and returns for action and states vector an approximated value.
        :param states: list of users busy time.
        :param action: chosen action corresponding to the states.
        :return: a single approximated value.
        """
        value = 0.0
        print(states, action)
        for i, state_value in enumerate(states):
            print(state_value * self.theta[i + action * self.number_of_users * 2], state_value,
                  self.theta[i + action * self.number_of_users * 2])
            value += state_value * self.theta[i + action * self.number_of_users * 2]
        print("=====")
        return value

    def update_theta(self):
        """
MC method to learn based on its followed trajectory. Evaluates the history list in reverse and for each states-action pair updates its internal theta vector.
        """
        reward = np.average(self.jobs_lateness)
        for i, (states, action) in enumerate(reversed(self.history)):
            self.theta += self.alpha * (
                self.gamma ** i * -reward - self.action_value_approximator(states, action)) * self.gradient(
                states,
                action)

    def save_rewards(self, policy_job):
        """
Evaluates and appends the job's lateness to a policy global queue.
        :param policy_job: policyjob object passed in each release method.
        """
        job_lateness = policy_job.finished - policy_job.started
        self.jobs_lateness.append(job_lateness)
        self.jobs_service_times.append(policy_job.service_rate[policy_job.assigned_user])

    def gradient(self, states, action):
        """
For each states-action pair calculates the gradient descent to be used in the theta update function.
        :param states: list of users busy time.
        :param action: chosen action corresponding to the states.
        :return: gradient to be used for updating theta vector towards optimum.
        """
        gradient_vector = np.zeros(2 * (self.number_of_users ** 2))
        for i, state_value in enumerate(states):
            gradient_vector[i + action * self.number_of_users * 2] = state_value
        return gradient_vector

    def policy_status(self):
        """
Evaluates the current state of the policy. Overrides parent method with KBatch specific logic.
        :return: returns a list where the first item is the global queue length and all subsequent elements are the respective user queues length.
        """
        current_status = [len(self.batch_queue)]
        for i in range(self.number_of_users):
            current_status.append(len(self.users_queues[i]))
        return current_status
