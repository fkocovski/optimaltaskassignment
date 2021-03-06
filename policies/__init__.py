import itertools


class Policy(object):
    def __init__(self, env, number_of_users, worker_variability, file_policy):
        """
Parent class initialization for all policy objects.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param file_policy: file object to calculate policy related statistics.
        """
        self.env = env
        self.number_of_users = number_of_users
        self.worker_variability = worker_variability
        self.file_policy = file_policy

    def request(self, user_task,token):
        """
Parent class request method for user task objects to request an optimal solution. Initializes a policy job.
        :param token: token going through the process.
        :param user_task: user task object that requests optimal solution.
        :return: initialized policy job object.
        """
        average_processing_time = token.random_state.gamma(
            user_task.service_interval ** 2 / user_task.task_variability,
            user_task.task_variability / user_task.service_interval)

        policy_job = PolicyJob(user_task,token)
        policy_job.request_event = self.env.event()
        policy_job.arrival = self.env.now

        policy_job.service_rate = [token.random_state.gamma(average_processing_time ** 2 / self.worker_variability,
                                                           self.worker_variability / average_processing_time) for
                                   _ in range(self.number_of_users)]

        return policy_job

    def release(self, policy_job):
        """
Parent class release method to manage and release finished policy job objects.
        :param policy_job: policy job object holding all relevant information to be released.
        """
        policy_job.finished = self.env.now
        policy_job.save_info(self.file_policy)


class PolicyJob(object):
    job_id = itertools.count()

    def __init__(self, user_task,token):
        """
Policy job object initialization method.
        :param user_task: user task passed is used to uniquely identify it inside a workflow process.
        """
        self.user_task = user_task
        self.token = token
        self.arrival = None
        self.assigned = None
        self.started = None
        self.finished = None
        self.assigned_user = None
        self.service_rate = None
        self.request_event = None
        self.job_id = next(PolicyJob.job_id)

    def is_busy(self, now):
        """
Method used to determine whether a policy job object is currently being worked at the time passed.
        :param now: current time passed as parameter.
        :return: returns a boolean indicating whether the policy job object is busy or not.
        """
        if self.started is None:
            return False
        elif self.will_finish() < now:
            return False
        else:
            return True

    def will_finish(self):
        """
Method used to determine future finish time of a policy job object by adding its allocated service time of the user working it to its started time.
        :return: returns a simpy time
        """
        return self.started + self.service_rate[self.assigned_user]

    def save_info(self, file):
        """
Method used to save information required to calculate key metrics.
        :param file: passed file object to write key metrics into.
        """
        if file is None:
            return

        file.write(
            "{},{},{},{},{},{},{},{},{}".format(self.job_id, self.arrival, self.assigned, self.started, self.finished,
                                             self.assigned_user + 1, self.user_task.node_id, self.user_task.name,self.token.id))
        for st in self.service_rate:
            file.write(",{}".format(st))
        file.write("\n")
