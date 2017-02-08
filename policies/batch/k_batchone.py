from policies import *


class KBatchOne(Policy):
    def __init__(self, env, number_of_users, worker_variability, batch_size, solver, file_policy, file_statistics):
        """
Initializes a KBatchOne policy.
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
        self.assigned_job_to_user = [None] * self.number_of_users
        self.batch_queue = []

    def request(self, user_task):
        """
Request method for KBatchOne policies. Creates a PolicyJob object and calls for the appropriate evaluation method with the corresponding solver.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        super().request(user_task)

        average_processing_time = RANDOM_STATE.gamma(
            user_task.service_interval ** 2 / user_task.task_variability,
            user_task.task_variability / user_task.service_interval)

        k_batch_one_job = PolicyJob(user_task)
        k_batch_one_job.request_event = self.env.event()
        k_batch_one_job.arrival = self.env.now
        k_batch_one_job.service_rate = [
            RANDOM_STATE.gamma(average_processing_time ** 2 / self.worker_variability,
                               self.worker_variability / average_processing_time) for
            _ in range(self.number_of_users)]

        self.save_status()

        self.batch_queue.append(k_batch_one_job)

        self.save_status()

        if len(self.batch_queue) >= self.batch_size:
            self.evaluate()

        self.save_status()

        return k_batch_one_job

    def release(self, k_batch_one_job):
        """
Release method for KBatchOne policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the global queue's size is greater than the batch size, it calls the appropriate evaluation method again.
        :param k_batch_one_job: a policyjob object.
        """
        super().release(k_batch_one_job)

        self.save_status()

        user_to_release_index = k_batch_one_job.assigned_user
        self.assigned_job_to_user[user_to_release_index] = None

        self.save_status()

        if len(self.batch_queue) >= self.batch_size:
            self.evaluate()

        self.save_status()

    def evaluate(self):
        """
Evaluate method for KBatchOne policies. Sets the required variables by the solver then calls the appropriate solver assigned and implements its returned solution.
        """
        # wj
        w = [self.env.now - self.batch_queue[j].arrival for j in range(len(self.batch_queue))]

        # pij
        p = [[self.batch_queue[j].service_rate[i] for j in range(len(self.batch_queue))] for i in
             range(self.number_of_users)]
        current_user_element = [self.assigned_job_to_user[i] for i in range(self.number_of_users)]

        # ai
        a = [0 if current_user_element[i] is None else current_user_element[i].will_finish() - self.env.now for i
             in range(self.number_of_users)]

        assignment, _ = self.solver(a, p, w, len(self.batch_queue), self.number_of_users)

        for job_index, user_index in enumerate(assignment):
            if self.assigned_job_to_user[user_index] is None:
                job = self.batch_queue[job_index]
                self.batch_queue[job_index] = None
                self.assigned_job_to_user[user_index] = job
                job.assigned_user = user_index
                job.started = self.env.now
                job.request_event.succeed(job.service_rate[user_index])
        self.batch_queue = [job for job in self.batch_queue if job is not None]

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
