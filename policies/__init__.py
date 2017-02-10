import numpy as np

RANDOM_STATE = np.random.RandomState(1)
RANDOM_STATE_ACTIONS = np.random.RandomState(1)


class Policy(object):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics):
        """
Parent class initialization for all policy objects.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param file_policy: file object to calculate policy related statistics.
        :param file_statistics: file object to draw the policy evolution.
        """
        self.env = env
        self.number_of_users = number_of_users
        self.worker_variability = worker_variability
        self.file_policy = file_policy
        self.file_statistics = file_statistics

    def request(self, user_task):
        """
Parent class request method for user task objects to request an optimal solution. Initializes a policy job.
        :param user_task: user task object that requests optimal solution.
        :return: initialized policy job object.
        """
        average_processing_time = RANDOM_STATE.gamma(
            user_task.service_interval ** 2 / user_task.task_variability,
            user_task.task_variability / user_task.service_interval)

        policy_job = PolicyJob(user_task)
        policy_job.request_event = self.env.event()
        policy_job.arrival = self.env.now

        policy_job.service_rate = [RANDOM_STATE.gamma(average_processing_time ** 2 / self.worker_variability,
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

    def save_status(self):
        """
Parent class method that saves information required to plot the policy evolution over time.
        """
        if self.file_statistics is None:
            return

        current_status = self.policy_status()
        self.file_statistics.write("{}".format(self.env.now))
        for val in current_status:
            self.file_statistics.write(",{}".format(val))
        self.file_statistics.write("\n")

    def policy_status(self):
        """
Parent class method that is overriden by its children to save policy specific status information.
        """
        pass


class PolicyJob(object):
    def __init__(self, user_task):
        """
Policy job object initialization method.
        :param user_task: user task passed is used to uniquely identify it inside a workflow process.
        """
        self.user_task = user_task
        self.arrival = None
        self.started = None
        self.finished = None
        self.assigned_user = None
        self.service_rate = None
        self.request_event = None

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
            "{},{},{},{},{}".format(id(self), self.arrival, self.started, self.finished, self.assigned_user + 1))
        for st in self.service_rate:
            file.write(",{}".format(st))
        file.write("\n")
