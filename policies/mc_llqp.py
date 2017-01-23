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

        self.state_space = self.init_state_space()
        self.state_action = self.init_state_action()
        self.state_action_counts = self.init_state_action_count()
        self.mc_chunk = 0
        self.current_state = None
        self.returns = {}

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
        if self.mc_chunk == 0:
            self.current_state = self.state_space[0]

        if self.mc_chunk % 10 == 0 and self.mc_chunk != 0:
            for key in self.returns:
                self.returns[key] = self.get_reward(llqp_job)
                self.state_action[key] = self.state_action[key] + (1/self.state_action_counts[key])*(self.returns[key]-self.state_action[key])

        current_total_time = [None]*self.number_of_users
        for user_index, user_deq in enumerate(self.users_queues):
            if len(user_deq) > 0:
                leftmost_queue_element = user_deq[0]
                current_total_time[user_index] = sum(job.service_rate[user_index] for job in user_deq)
                if leftmost_queue_element.is_busy(self.env.now):
                    current_total_time[user_index] += leftmost_queue_element.will_finish() - self.env.now

        act_probs = self.qsv(self.current_state)
        if numpy.random.random() < 0.1:
            action = numpy.random.randint(0,self.number_of_users)
        else:
            action = numpy.argmax(act_probs)
        sa = (self.current_state,action)
        self.returns[sa] = 0
        self.state_action_counts[sa] += 1
        new_state = (action,llqp_job.service_rate[action])
        self.state_space.append(new_state)
        for state in self.state_space:
            for a in range(self.number_of_users):
                self.state_action[(state,a)] = 0.0
        self.current_state = new_state

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

    def init_state_space(self):
        # [(user,user_wait)]
        states = []
        for i in range(self.number_of_users):
            states.append((i,0.0))
        return states

    def init_state_action(self):
        # {((user,user_wait),action):0.0}
        av = {}
        for state in self.state_space:
            for i in range(self.number_of_users):
                av[(state, i)] = 0.0
        return av

    def init_state_action_count(self):
        counts = {}
        for sa in self.state_action:
            counts[sa] = 0
        return counts

    def qsv(self,state):
        qsv_values = [None]*self.number_of_users
        for i in range(self.number_of_users):
            qsv_values[i] = self.state_action[(state,i)]
        return numpy.array(qsv_values)

    def get_reward(self,llqp_job):
        current_st = self.current_state[1]
        if current_st == min(llqp_job.service_rate):
            reward = 1
        else:
            reward = -1

        return reward


