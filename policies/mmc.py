from policies import *
from collections import deque


class MMC(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics):
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.name = "M/M/{}".format(self.number_of_users)
        self.waiting_queue = deque()
        self.assigned_job_to_user = [None] * self.number_of_users

    def request(self, user_task):
        super().request(user_task)

        self.save_status()

        average_processing_time = RANDOM_STATE.gamma(
            user_task.service_interval ** 2 / user_task.task_variability,
            user_task.task_variability / user_task.service_interval)

        mmc_job = PolicyJob(user_task)
        mmc_job.request_event = self.env.event()
        mmc_job.arrival = self.env.now

        mmc_job.service_rate = [RANDOM_STATE.gamma(average_processing_time ** 2 / self.worker_variability,
                                                   self.worker_variability / average_processing_time) for
                                _ in range(self.number_of_users)]

        self.evaluate(mmc_job)

        self.save_status()

        return mmc_job

    def release(self, mmc_job):
        super().release(mmc_job)

        self.save_status()

        user_to_release_index = mmc_job.assigned_user

        self.assigned_job_to_user[user_to_release_index] = None

        if len(self.waiting_queue) > 0:
            next_mmc_job = self.waiting_queue.popleft()
            next_mmc_job.started = self.env.now
            next_mmc_job.assigned_user = user_to_release_index
            next_mmc_job.request_event.succeed(next_mmc_job.service_rate[user_to_release_index])

    def evaluate(self, mmc_job):
        try:
            idle_user_index = self.assigned_job_to_user.index(None)
            self.assigned_job_to_user[idle_user_index] = mmc_job
            mmc_job.assigned_user = idle_user_index
            mmc_job.started = self.env.now
            mmc_job.request_event.succeed(mmc_job.service_rate[idle_user_index])
        except ValueError:
            self.waiting_queue.append(mmc_job)

    def policy_status(self):
        current_status = [len(self.waiting_queue)]
        for i in range(self.number_of_users):
            if self.assigned_job_to_user[i] is None:
                current_status.append(0)
            else:
                current_status.append(1)
        return current_status
