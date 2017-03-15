from policies import *
from collections import deque


class LLQP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy):
        """
Initializes an LLQP policy.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param file_policy: file object to calculate policy related statistics.
        """
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.name = "LLQP"
        self.users_queues = [deque() for _ in range(self.number_of_users)]

    def request(self, user_task,token):
        """
Request method for LLQP policies. Creates a PolicyJob object and calls for the appropriate evaluation method.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        llqp_job = super().request(user_task,token)


        self.evaluate(llqp_job)


        return llqp_job

    def release(self, llqp_job):
        """
Release method for LLQP policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the released user's queue is not empty, it assigns the next policyjob to be worked.
        :param llqp_job: a policyjob object.
        """
        super().release(llqp_job)
        user_to_release_index = llqp_job.assigned_user

        user_queue_to_free = self.users_queues[user_to_release_index]


        user_queue_to_free.popleft()


        if len(user_queue_to_free) > 0:
            next_llqp_job = user_queue_to_free[0]
            next_llqp_job.started = self.env.now
            next_llqp_job.request_event.succeed(next_llqp_job.service_rate[user_to_release_index])

    def evaluate(self, llqp_job):
        """
Evaluate method for LLQP policies. Looks for the currently least loaded person to assign the policyjob.
        :param llqp_job: a policyjob object to be assigned.
        """
        llqp_index = None
        lowest_time = None
        for user_index, user_deq in enumerate(self.users_queues):
            current_total_time = 0
            if len(user_deq) > 0:
                leftmost_queue_element = user_deq[0]
                current_total_time = sum(job.service_rate[user_index] for job in user_deq)
                if leftmost_queue_element.is_busy(self.env.now):
                    current_total_time -= self.env.now - leftmost_queue_element.started
            if lowest_time is None or lowest_time > current_total_time:
                llqp_index = user_index
                lowest_time = current_total_time
        llqp_queue = self.users_queues[llqp_index]
        llqp_job.assigned_user = llqp_index
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[llqp_index])