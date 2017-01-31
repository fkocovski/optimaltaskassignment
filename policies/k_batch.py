from policies import *
from collections import deque


class KBatch(Policy):
    def __init__(self, env, number_of_users, worker_variability, batch_size, solver, file_policy, file_statistics):
        """
Initializes a KBatch policy.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param batch_size: the batch size of the global queue.
        :param solver: the solver used for the optimal task assignment.
        :param file_policy: file object to calculate policy related statistics.
        :param file_statistics: file object to draw the policy evolution.
        """
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.batch_size = batch_size
        self.solver = solver
        self.name = "{}Batch".format(self.batch_size)
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.batch_queue = []

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
        k_batch_job.service_rate = [RANDOM_STATE.gamma(average_processing_time ** 2/ self.worker_variability,
                               self.worker_variability / average_processing_time) for
            _ in range(self.number_of_users)]

        self.save_status()

        self.batch_queue.append(k_batch_job)

        self.save_status()

        if len(self.batch_queue) == self.batch_size:
            self.evaluate()

        self.save_status()

        return k_batch_job

    def release(self, k_batch_job):
        """
Release method for KBatch policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the released user's queue is not empty, it assigns the next policyjob to be worked.
        :param k_batch_job: a policyjob object.
        """
        super().release(k_batch_job)

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

    def evaluate(self):
        """
Evaluate method for KBatch policies. Sets the required variables by the solver then calls the appropriate solver assigned and implements its returned solution.
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

        assignment, _ = self.solver(a, p, w, len(self.batch_queue), self.number_of_users)

        for job_index, user_index in enumerate(assignment):
            job = self.batch_queue[job_index]
            self.users_queues[user_index].append(job)
            job.assigned_user = user_index
        self.batch_queue.clear()
        for user_index in range(self.number_of_users):
            if len(self.users_queues[user_index]) > 0:
                leftmost_llpq_element = self.users_queues[user_index][0]
                if not leftmost_llpq_element.is_busy(self.env.now):
                    leftmost_llpq_element.started = self.env.now
                    leftmost_llpq_element.request_event.succeed(leftmost_llpq_element.service_rate[user_index])

    def policy_status(self):
        """
Evaluates the current state of the policy. Overrides parent method with KBatch specific logic.
        :return: returns a list where the first item is the global queue length and all subsequent elements are the respective user queues length.
        """
        current_status = [len(self.batch_queue)]
        for i in range(self.number_of_users):
            current_status.append(len(self.users_queues[i]))
        return current_status
