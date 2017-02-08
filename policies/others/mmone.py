from policies import *
from collections import deque


class MMONE(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics):
        """
Initializes an MMONE policy.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param file_policy: file object to calculate policy related statistics.
        :param file_statistics: file object to draw the policy evolution.
        """
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.waiting_queue = deque()
        self.assigned_job_to_user = None
        self.name = "M/M/1"

    def request(self, user_task):
        """
Request method for MMONE policies. Creates a PolicyJob object and calls for the appropriate evaluation method.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        super().request(user_task)

        self.save_status()

        average_processing_time = RANDOM_STATE.gamma(
            user_task.service_interval ** 2 / user_task.task_variability,
            user_task.task_variability / user_task.service_interval)

        mmone_job = PolicyJob(user_task)
        mmone_job.request_event = self.env.event()
        mmone_job.arrival = self.env.now

        mmone_job.service_rate = [RANDOM_STATE.gamma(average_processing_time ** 2 / self.worker_variability,
                                                     self.worker_variability / average_processing_time) for
                                  _ in range(self.number_of_users)]

        self.evaluate(mmone_job)

        self.save_status()

        return mmone_job

    def release(self, mmone_job):
        """
Release method for MMONE policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the waiting queue is not empty, it assigns the next policyjob to be worked to the currently freed user.
        :param mmone_job: a policyjob object to be assigned.
        """
        super().release(mmone_job)

        self.save_status()

        self.assigned_job_to_user = None

        if len(self.waiting_queue) > 0:
            next_mmone_job = self.waiting_queue.popleft()
            self.assigned_job_to_user = next_mmone_job
            next_mmone_job.started = self.env.now
            next_mmone_job.assigned_user = 0
            next_mmone_job.request_event.succeed(next_mmone_job.service_rate[0])

        self.save_status()

    def evaluate(self, mmone_job):
        """
Evaluate method for MMONE policies. Looks if the user is free. If yes it assigned the policyjob to him otherwise appends the policyjob in the global queue.
        :param mmone_job: a policyjob object to be assigned.
        """
        if self.assigned_job_to_user is not None:
            self.waiting_queue.append(mmone_job)
        else:
            self.assigned_job_to_user = mmone_job
            mmone_job.assigned_user = 0
            mmone_job.started = self.env.now
            mmone_job.request_event.succeed(mmone_job.service_rate[0])

    def policy_status(self):
        """
Evaluates the current state of the policy. Overrides parent method with SQ specific logic.
        :return: returns a list where the first item is the global queue length and the current user queue status.
        """
        current_status = [len(self.waiting_queue)]
        if self.assigned_job_to_user is None:
            current_status.append(0)
        else:
            current_status.append(1)
        return current_status
