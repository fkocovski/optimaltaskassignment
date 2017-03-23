from policies import *
from collections import deque


class SQ(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.name = "SQ"
        self.waiting_queue = deque()
        self.assigned_job_to_user = [None] * self.number_of_users

    def request(self, user_task,token):
        sq_job = super().request(user_task,token)

        self.evaluate(sq_job)

        return sq_job

    def release(self, sq_job):
        super().release(sq_job)
        user_to_release_index = sq_job.assigned_user

        self.assigned_job_to_user[user_to_release_index] = None

        if len(self.waiting_queue) > 0:
            next_sq_job = self.waiting_queue.popleft()
            next_sq_job.started = self.env.now
            next_sq_job.assigned = self.env.now
            next_sq_job.assigned_user = user_to_release_index
            next_sq_job.request_event.succeed(next_sq_job.service_rate[user_to_release_index])

    def evaluate(self, sq_job):
        try:
            idle_user_job_index, idle_user_job = next(
                (index, job) for index, job in enumerate(self.assigned_job_to_user) if job is None)
            self.assigned_job_to_user[idle_user_job_index] = sq_job
            sq_job.assigned_user = idle_user_job_index
            sq_job.assigned = self.env.now
            sq_job.started = self.env.now
            sq_job.request_event.succeed(sq_job.service_rate[idle_user_job_index])
        except StopIteration:
            self.waiting_queue.append(sq_job)

