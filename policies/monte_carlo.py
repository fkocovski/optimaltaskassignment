import numpy as np
from policies import *
from collections import deque


class MC(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics,q_table,epsilon,gamma):
        """
Initializes an LLQP policy.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param file_policy: file object to calculate policy related statistics.
        :param file_statistics: file object to draw the policy evolution.
        """
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.name = "LLQP"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.q_table = q_table
        self.epsilon = epsilon
        self.history = []
        self.returns = []
        self.gamma = gamma

    def request(self, user_task):
        """
Request method for LLQP policies. Creates a PolicyJob object and calls for the appropriate evaluation method.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        super().request(user_task)

        average_processing_time = RANDOM_STATE.gamma(
            user_task.service_interval ** 2 / user_task.task_variability,
            user_task.task_variability / user_task.service_interval)

        llqp_job = PolicyJob(user_task)
        llqp_job.request_event = self.env.event()
        llqp_job.arrival = self.env.now

        llqp_job.service_rate = [RANDOM_STATE.gamma(average_processing_time ** 2 / self.worker_variability,
                                                    self.worker_variability / average_processing_time) for
                                 _ in range(self.number_of_users)]

        self.evaluate(llqp_job)

        self.save_status()

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
        # llqp_index = None
        # lowest_time = None
        busy_times = [None]*self.number_of_users
        for user_index, user_deq in enumerate(self.users_queues):
            # current_total_time = 0
            if len(user_deq) > 0:
                leftmost_queue_element = user_deq[0]
                busy_times[user_index] = sum(job.service_rate[user_index] for job in user_deq)
                if leftmost_queue_element.is_busy(self.env.now):
                    busy_times[user_index] -= self.env.now - leftmost_queue_element.started
            # if lowest_time is None or lowest_time > current_total_time:
            #     llqp_index = user_index
            #     lowest_time = current_total_time
            else:
                busy_times[user_index] = 0


        # FIXME: do not use fixed index for current_state
        current_state = [None]*self.number_of_users
        for i,a in enumerate(busy_times):
            current_state[i] = int(a)

        if RANDOM_STATE.rand() < self.epsilon:
            action = RANDOM_STATE.randint(0,2)
        else:
            action = np.argmax(self.q_table[current_state[0],current_state[1]])


        self.update_policy_status(current_state[0],current_state[1],action)

        llqp_queue = self.users_queues[action]
        llqp_job.assigned_user = action
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[action])

    def policy_status(self):
        """
Evaluates the current state of the policy. Overrides parent method with LLQP specific logic.
        :return: returns a list where the first item is the global queue length (in LLQP always zero) and all subsequent elements are the respective user queues length.
        """
        current_status = [0]
        for i in range(self.number_of_users):
            current_status.append(len(self.users_queues[i]))
        return current_status

    def update_policy_status(self, user_one, user_two, action):
        if user_one < user_two and action == 0:
            self.returns.append(1)
        elif user_one > user_two and action == 1:
            self.returns.append(1)
        elif user_one == user_two:
            self.returns.append(1)
        else:
            self.returns.append(-1)

        self.history.append((user_one,user_two,action))

    def update_q_table(self):
        rewards = [None] * len(self.returns)
        for index, ret in enumerate(reversed(self.returns)):
            if index == 0:
                rewards[index] = ret
            else:
                rewards[-index] = ret + self.gamma * rewards[-index + 1]
        for index, sa in enumerate(self.history):
            self.q_table[sa] += (1 / self.history.count(sa)) * (rewards[index] - self.q_table[sa])

        return self.q_table
