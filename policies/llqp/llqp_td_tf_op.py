import randomstate.prng.pcg64 as pcg
import numpy as np
import tensorflow as tf
from policies import *
from collections import deque


class LLQP_TD_TF_OP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, theta, gamma, alpha, greedy):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.gamma = gamma
        self.greedy = greedy
        self.RANDOM_STATE_ACTIONS = pcg.RandomState(1)
        self.name = "LLQP_TD_TF_OP"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.history = None

        self.w = [tf.Variable(tf.zeros([self.number_of_users])) for _ in range(self.number_of_users)]
        self.x = tf.placeholder(tf.float32,shape=[self.number_of_users])
        self.q_val = [tf.reduce_sum(tf.multiply(self.x, self.w[a])) for a in range(self.number_of_users)]
        self.y = [tf.placeholder(tf.float32) for _ in range(self.number_of_users)]
        squared_deltas = [tf.square(self.y[a] - self.q_val[a]) for a in range(self.number_of_users)]
        # loss = [tf.reduce_sum(squared_deltas[a]) for a in range(self.number_of_users)]
        loss = [squared_deltas[a] for a in range(self.number_of_users)]
        optimizer = [tf.train.GradientDescentOptimizer(0.0001) for _ in range(self.number_of_users)]
        self.train = [optimizer[a].minimize(loss[a]) for a in range(self.number_of_users)]
        self.tf_init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.tf_init)

    def request(self, user_task, token):
        llqp_job = super().request(user_task, token)

        self.evaluate(llqp_job)

        return llqp_job

    def release(self, llqp_job):
        super().release(llqp_job)

        user_to_release_index = llqp_job.assigned_user

        user_queue_to_free = self.users_queues[user_to_release_index]

        user_queue_to_free.popleft()

        if len(user_queue_to_free) > 0:
            next_llqp_job = user_queue_to_free[0]
            next_llqp_job.started = self.env.now
            next_llqp_job.request_event.succeed(next_llqp_job.service_rate[user_to_release_index])

    def evaluate(self, llqp_job):
        busy_times = self.get_busy_times()

        if self.greedy:
            action = min(range(self.number_of_users),
                         key=lambda action: self.q(busy_times, action))
        else:
            action = self.RANDOM_STATE_ACTIONS.randint(0, self.number_of_users)

        llqp_queue = self.users_queues[action]
        llqp_job.assigned_user = action
        llqp_queue.append(llqp_job)
        llqp_job.assigned = self.env.now
        leftmost_llqp_queue_element = llqp_queue[0]
        if not leftmost_llqp_queue_element.is_busy(self.env.now):
            llqp_job.started = self.env.now
            llqp_job.request_event.succeed(llqp_job.service_rate[action])

        if not self.greedy:
            if self.history is not None:
                self.update_theta(busy_times)

            reward = busy_times[action] + llqp_job.service_rate[action]

            self.history = (busy_times, action, reward)

    def get_busy_times(self):
        busy_times = np.zeros(self.number_of_users)
        for user_index, user_deq in enumerate(self.users_queues):
            if len(user_deq) > 0:
                busy_times[user_index] = sum(job.service_rate[user_index] for job in user_deq)
                if user_deq[0].is_busy(self.env.now):
                    busy_times[user_index] -= self.env.now - user_deq[0].started
            else:
                busy_times[user_index] = 0
        return busy_times

    def q(self, states, action):
        q = self.sess.run(self.q_val[action], {self.x: states})
        return q

    def update_theta(self, new_busy_times):
        old_busy_times, old_action, reward = self.history
        # y = reward + self.gamma * (min(self.q(new_busy_times, a) for a in range(self.number_of_users))) - self.q(old_busy_times, old_action)
        y = reward + self.gamma * (min(self.q(new_busy_times, a) for a in range(self.number_of_users)))
        w_old = self.sess.run(self.w)
        self.sess.run(self.train[old_action],{self.x:old_busy_times,self.y[old_action]:y})
        w_new = self.sess.run(self.w)
        diff_w = [w_new[a]-w_old[a] for a in range(self.number_of_users)]
        for w_val in diff_w:
            print(w_val,y)
        print("------")
