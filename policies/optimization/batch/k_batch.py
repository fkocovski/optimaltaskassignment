from policies import *
from collections import deque


class K_BATCH(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, batch_size, solver):
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
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.batch_size = batch_size
        self.solver = solver
        self.name = "{}_BATCH".format(self.batch_size)
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.batch_queue = []

    def request(self, user_task, token):
        """
Request method for KBatch policies. Creates a PolicyJob object and calls for the appropriate evaluation method with the corresponding solver.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        k_batch_job = super().request(user_task, token)

        self.batch_queue.append(k_batch_job)

        if len(self.batch_queue) == self.batch_size:
            self.evaluate()

        return k_batch_job

    def release(self, k_batch_job):
        """
Release method for KBatch policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the released user's queue is not empty, it assigns the next policyjob to be worked.
        :param k_batch_job: a policyjob object.
        """
        super().release(k_batch_job)

        user_to_release_index = k_batch_job.assigned_user
        queue_to_pop = self.users_queues[user_to_release_index]

        queue_to_pop.popleft()

        if len(queue_to_pop) > 0:
            next_k_batch_job = queue_to_pop[0]
            next_k_batch_job.started = self.env.now
            next_k_batch_job.request_event.succeed(next_k_batch_job.service_rate[user_to_release_index])

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
            job.assigned = self.env.now
        self.batch_queue.clear()
        for user_index in range(self.number_of_users):
            if len(self.users_queues[user_index]) > 0:
                leftmost_llpq_element = self.users_queues[user_index][0]
                if not leftmost_llpq_element.is_busy(self.env.now):
                    leftmost_llpq_element.started = self.env.now
                    leftmost_llpq_element.request_event.succeed(leftmost_llpq_element.service_rate[user_index])
