from policies import *


class K_BATCHONE(Policy):
    def __init__(self, env, number_of_users, worker_variability,file_policy, batch_size, solver):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.batch_size = batch_size
        self.solver = solver
        self.name = "{}BATCH_ONE".format(self.batch_size)
        self.assigned_job_to_user = [None] * self.number_of_users
        self.batch_queue = []

    def request(self, user_task,token):
        k_batch_one_job = super().request(user_task,token)


        self.batch_queue.append(k_batch_one_job)


        if len(self.batch_queue) >= self.batch_size:
            self.evaluate()


        return k_batch_one_job

    def release(self, k_batch_one_job):
        super().release(k_batch_one_job)


        user_to_release_index = k_batch_one_job.assigned_user
        self.assigned_job_to_user[user_to_release_index] = None


        if len(self.batch_queue) >= self.batch_size:
            self.evaluate()


    def evaluate(self):
        # wj
        w = [self.env.now - self.batch_queue[j].arrival for j in range(len(self.batch_queue))]

        # pij
        p = [[self.batch_queue[j].service_rate[i] for j in range(len(self.batch_queue))] for i in
             range(self.number_of_users)]

        # ai
        current_user_element = [self.assigned_job_to_user[i] for i in range(self.number_of_users)]
        a = [0 if self.assigned_job_to_user[i] is None else self.assigned_job_to_user[i].will_finish() - self.env.now for i
             in range(self.number_of_users)]

        assignment, _ = self.solver(a, p, w, len(self.batch_queue), self.number_of_users)

        for job_index, user_index in enumerate(assignment):
            if self.assigned_job_to_user[user_index] is None:
                job = self.batch_queue[job_index]
                self.batch_queue[job_index] = None
                self.assigned_job_to_user[user_index] = job
                job.assigned_user = user_index
                job.assigned = self.env.now
                job.started = self.env.now
                job.request_event.succeed(job.service_rate[user_index])
        self.batch_queue = [job for job in self.batch_queue if job is not None]