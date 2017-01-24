from policies import *
from collections import deque


class LLQP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics):
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

        self.states_actions, self.states_actions_counts, self.returns = self.init_state_space_action_count()
        self.rewards = []
        self.history = []
        self.mc_chunk = 0
        self.current_state = None

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

        if self.mc_chunk % 10 == 0 and self.mc_chunk != 0:
            for action in range(2):
                for a_one in range(100):
                    for a_two in range(100):
                        self.states_actions[action, a_one, a_two] += (1 /self.states_actions_counts[action, a_one, a_two]) * (self.returns[action, a_one, a_two] -self.states_actions[action, a_one, a_two])

        current_total_time = [None] * self.number_of_users
        for user_index, user_deq in enumerate(self.users_queues):
            if len(user_deq) > 0:
                leftmost_queue_element = user_deq[0]
                current_total_time[user_index] = sum(job.service_rate[user_index] for job in user_deq)
                if leftmost_queue_element.is_busy(self.env.now):
                    current_total_time[user_index] -=  self.env.now - leftmost_queue_element.started
            else:
                current_total_time[user_index] = 0

        if numpy.random.random() < 0.1:
            action = numpy.random.randint(0, self.number_of_users)
        else:
            a_one_max = numpy.amax(self.states_actions[0])
            a_two_max = numpy.amax(self.states_actions[1])
            if a_one_max > a_two_max:
                action = 0
            else:
                action = 1
        a_one, a_two = self.get_current_state(current_total_time)

        self.states_actions_counts[action, a_one, a_two] += 1
        self.set_reward(action, a_one, a_two)

        llqp_queue = self.users_queues[action]
        llqp_job.assigned_user = action
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[action])

        self.mc_chunk += 1

    def policy_status(self):
        """
Evaluates the current state of the policy. Overrides parent method with LLQP specific logic.
        :return: returns a list where the first item is the global queue length (in LLQP always zero) and all subsequent elements are the respective user queues length.
        """
        current_status = [0]
        for i in range(self.number_of_users):
            current_status.append(len(self.users_queues[i]))
        return current_status

    def init_state_space_action_count(self):
        states_actions = numpy.zeros((2, 100, 100))
        states_actions_counts = numpy.ones((2, 100, 100), dtype=int)
        returns = numpy.zeros((2, 100, 100))
        return states_actions, states_actions_counts, returns

    def set_reward(self, action, a_one, a_two):
        if a_one < a_two and action == 0:
            self.returns[action, a_one, a_two] = 1
        elif a_one > a_two and action == 1:
            self.returns[action, a_one, a_two] = 1
        elif a_one == a_two:
            self.returns[action, a_one, a_two] = 1
        else:
            self.returns[action, a_one, a_two] = -10

    def get_current_state(self, current_waiting_times):
        if current_waiting_times[0] is None:
            a_one = 0
        else:
            a_one = int(current_waiting_times[0])

        if current_waiting_times[1] is None:
            a_two = 0
        else:
            a_two = int(current_waiting_times[1])

        return a_one, a_two
