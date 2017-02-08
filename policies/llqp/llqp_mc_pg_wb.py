from policies import *
from collections import deque
import math

RANDOM_STATE_PROBABILITIES = np.random.RandomState(1)


class LLQP_MC_PG_WB(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, file_statistics, w, theta, gamma, alpha,
                 beta):
        """
Initializes an LLQP policy using a MC approach with VFA, PG and baseline.
        :param w: weight vector for state value approximation.
        :param beta: discount factor for state value approximation.
        :param env: simpy environment.
        :param number_of_users: the number of users present in the system.
        :param worker_variability: worker variability in absolute value.
        :param file_policy: file object to calculate policy related statistics.
        :param file_statistics: file object to draw the policy evolution.
        :param theta: weight vector for VFA.
        :param gamma: discounting factor for rewards.
        :param alpha: step size parameter for the gradient descent method.
        """
        super().__init__(env, number_of_users, worker_variability, file_policy, file_statistics)
        self.name = "LLQP"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.w = w
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.history = []
        self.jobs_lateness = []

    def request(self, user_task):
        """
Request method for MC policies. Creates a PolicyJob object and calls for the appropriate evaluation method.
        :param user_task: a user task object.
        :return: a policyjob object to be yielded in the simpy environment.
        """
        llqp_job = super().request(user_task)

        self.save_status()

        self.evaluate(llqp_job)

        self.save_status()

        return llqp_job

    def release(self, llqp_job):
        """
Release method for MC policies. Uses the passed parameter, which is a policyjob previously yielded by the request method and releases it. Furthermore it frees the user that worked the passed policyjob object. If the released user's queue is not empty, it assigns the next policyjob to be worked.
        :param llqp_job: a policyjob object.
        """
        super().release(llqp_job)

        user_to_release_index = llqp_job.assigned_user

        user_queue_to_free = self.users_queues[user_to_release_index]

        self.save_status()

        user_queue_to_free.popleft()

        self.save_status()

        if len(user_queue_to_free) > 0:
            next_llqp_job = user_queue_to_free[0]
            next_llqp_job.started = self.env.now
            next_llqp_job.request_event.succeed(next_llqp_job.service_rate[user_to_release_index])

        self.save_status()

    def evaluate(self, llqp_job):
        """
Evaluate method for MC policies. Creates a continuous state space which corresponds to the users busy times and follows and epsilon greedy policy approach to optimally choose the best user.
        :param llqp_job: a policyjob object to be assigned.
        """
        busy_times = self.get_busy_times()

        probabilities = self.policy_probabilities(busy_times)

        chosen_action = RANDOM_STATE_PROBABILITIES.choice(self.number_of_users, p=probabilities)

        self.history.append((busy_times, chosen_action))
        self.jobs_lateness.append(busy_times[chosen_action] + llqp_job.service_rate[chosen_action])

        llqp_queue = self.users_queues[chosen_action]
        llqp_job.assigned_user = chosen_action
        llqp_queue.append(llqp_job)
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[chosen_action])

    def get_busy_times(self):
        """
Calculates current busy times for users which represent the current state space.
        :return: list which indexes correspond to each user's busy time.
        """
        busy_times = [None] * self.number_of_users
        for user_index, user_deq in enumerate(self.users_queues):
            if len(user_deq) > 0:
                busy_times[user_index] = sum(job.service_rate[user_index] for job in user_deq)
                if user_deq[0].is_busy(self.env.now):
                    busy_times[user_index] -= self.env.now - user_deq[0].started
            else:
                busy_times[user_index] = 0
        return busy_times

    def policy_probabilities(self, busy_times):
        """
Calculates probabilities vector.
        :param busy_times: current state space.
        :return: list containing probabilities for each state.
        """
        probabilities = [None] * self.number_of_users
        for action in range(self.number_of_users):
            probabilities[action] = math.exp(self.action_value_approximator(busy_times, action)) / sum(
                math.exp(self.action_value_approximator(busy_times, a)) for a in range(self.number_of_users))
        return probabilities

    def policy_status(self):
        """
Evaluates the current state of the policy. Overrides parent method with MC specific logic.
        :return: returns a list where the first item is the global queue length (in MC always zero) and all subsequent elements are the respective user queues length.
        """
        current_status = [0]
        for i in range(self.number_of_users):
            current_status.append(len(self.users_queues[i]))
        return current_status

    def action_value_approximator(self, states, action):
        """
Action value function approximator. Uses the policy theta weight vector and returns for action and states vector an approximated value.
        :param states: list of users busy time.
        :param action: chosen action corresponding to the states.
        :return: a single approximated value.
        """
        value = 0.0
        for i, busy_time in enumerate(states):
            value += busy_time * self.theta[i + action * self.number_of_users]
        return value

    def state_value_approximator(self, states):
        """
State value function approximator. Uses the policy w weight vector and returns for all states the approximated value.
        :param states: list of users busy time.
        :return: a single approximated value.
        """
        value = 0.0
        for i, state in enumerate(states):
            value += self.w[i] * state
        return value

    def state_gradient(self, states):
        """
For all states calculates the gradient descent to be used in the theta update function.
        :param states: list of users busy time.
        :return: gradient to be used for updating w vector towards optimum.
        """
        state_gradient = np.zeros(self.number_of_users)
        for i, state in enumerate(states):
            state_gradient[i] = state
        return state_gradient

    def learn(self):
        """
PG with baseline method to learn based on its followed trajectory. Evaluates the history list and for each states-action pair updates its internal weight vectors.
        """
        for i, (states, action) in enumerate(self.history):
            disc_reward = -self.discount_rewards(i)
            delta = disc_reward - self.state_value_approximator(states)
            self.w += self.beta * delta * self.state_gradient(states)
            self.theta += self.alpha * self.gamma ** i * delta * (self.features(states, action) - sum(
                self.policy_probabilities(states)[a] * self.features(states, a) for a in range(self.number_of_users)))

    def features(self, states, action):
        """
Creates features vector for theta update function. For each action it creates a feature vector with busy times and the rest zeroes.
        :param states: busy times.
        :param action: chosen action.
        :return: vector full of zeroes except for action where the busy times are reported.
        """
        features = np.zeros(self.number_of_users ** 2)
        for act in range(self.number_of_users):
            features[act + action * self.number_of_users] = states[act]
        return features

    def discount_rewards(self, time):
        """
Discount rewards for one MC episode.
        """
        g = 0.0
        for t in range(time + 1):
            g += (self.gamma ** t) * self.jobs_lateness[t]
        return g
