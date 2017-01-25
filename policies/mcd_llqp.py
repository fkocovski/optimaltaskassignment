import numpy as np
from collections import deque

RANDOM_STATE = np.random.RandomState(1)


class LLQP(object):
    def __init__(self, env, number_of_users, worker_variability, task_variability, service_interval, states_actions,
                 file_statistics, file_policy):
        self.env = env
        self.number_of_users = number_of_users
        self.worker_variability = worker_variability
        self.task_variability = task_variability
        self.service_interval = service_interval
        self.name = "LLQP"
        self.file_statistics = file_statistics
        self.file_policy = file_policy
        self.users_queues = [deque() for _ in range(self.number_of_users)]

        self.states_actions = states_actions

    def request(self):

        average_processing_time = RANDOM_STATE.gamma(
            self.service_interval ** 2 / self.task_variability,
            self.task_variability / self.service_interval)

        llqp_job = PolicyJob()
        llqp_job.request_event = self.env.event()
        llqp_job.arrival = self.env.now

        llqp_job.service_rate = [RANDOM_STATE.gamma(average_processing_time ** 2 / self.worker_variability,
                                                    self.worker_variability / average_processing_time) for
                                 _ in range(self.number_of_users)]

        rwd, current_state = self.evaluate(llqp_job)

        self.save_status()

        return llqp_job, rwd, current_state

    def release(self, llqp_job):

        llqp_job.finished = self.env.now
        llqp_job.save_info(self.file_policy)
        user_to_release_index = llqp_job.assigned_user

        user_queue_to_free = self.users_queues[user_to_release_index]

        user_queue_to_free.popleft()

        self.save_status()

        if len(user_queue_to_free) > 0:
            next_llqp_job = user_queue_to_free[0]
            next_llqp_job.started = self.env.now
            next_llqp_job.request_event.succeed(next_llqp_job.service_rate[user_to_release_index])

    def evaluate(self, llqp_job):
        """
Evaluate method for LLQP policies. Looks for the currently least loaded person to assign the policyjob.
        :param llqp_job: a policyjob object to be assigned.
        """

        current_total_time = [None] * self.number_of_users
        for user_index, user_deq in enumerate(self.users_queues):
            if len(user_deq) > 0:
                leftmost_queue_element = user_deq[0]
                current_total_time[user_index] = sum(job.service_rate[user_index] for job in user_deq)
                if leftmost_queue_element.is_busy(self.env.now):
                    current_total_time[user_index] -= self.env.now - leftmost_queue_element.started
            else:
                current_total_time[user_index] = 0

        a_one, a_two = self.get_current_state(current_total_time)
        if np.random.random() < 0.1:
            action = np.random.randint(0, 2)
        else:
            a_one_max = np.amax(self.states_actions[a_one, a_two, 0])
            print(np.argmax(self.states_actions[a_one,a_two]))
            a_two_max = np.amax(self.states_actions[a_one, a_two, 1])
            if a_one_max > a_two_max:
                action = 0
            else:
                action = 1

        current_state = (a_one, a_two, action)

        rwd = self.set_reward(action, a_one, a_two)

        llqp_queue = self.users_queues[action]
        llqp_job.assigned_user = action
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[action])

        return rwd, current_state

    def set_reward(self, action, a_one, a_two):
        if a_one < a_two and action == 0:
            return 1
        elif a_one > a_two and action == 1:
            return 1
        elif a_one == a_two:
            return 1
        else:
            return -1

    def get_current_state(self, current_waiting_times):

        a_one = int(current_waiting_times[0])

        a_two = int(current_waiting_times[1])

        return a_one, a_two

    def save_status(self):
        """
Parent class method that saves information required to plot the policy evolution over time.
        """
        current_status = self.policy_status()
        self.file_statistics.write("{}".format(self.env.now))
        for val in current_status:
            self.file_statistics.write(",{}".format(val))
        self.file_statistics.write("\n")

    def policy_status(self):
        """
Parent class method that is overriden by its children to save policy specific status information.
        """
        current_status = [0]
        for i in range(self.number_of_users):
            current_status.append(len(self.users_queues[i]))
        return current_status


class PolicyJob(object):
    def __init__(self):
        """
Policy job object initialization method.
        :param user_task: user task passed is used to uniquely identify it inside a workflow process.
        """
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
        file.write(
            "{},{},{},{},{}".format(id(self), self.arrival, self.started, self.finished, self.assigned_user + 1))
        for st in self.service_rate:
            file.write(",{}".format(st))
        file.write("\n")
