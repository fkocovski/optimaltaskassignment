import numpy as np
import randomstate.prng.pcg64 as pcg
import tensorflow as tf
from policies import *
from collections import deque
from datetime import datetime

class LLQP_TD_TF_OP(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, w, gamma, greedy, sess):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.w = w
        self.gamma = gamma
        self.greedy = greedy
        self.sess = sess
        self.RANDOM_STATE_ACTIONS = pcg.RandomState(1)
        self.name = "LLQP_TD_TF_OP"
        self.users_queues = [deque() for _ in range(self.number_of_users)]
        self.history = None
        with tf.name_scope("inputs"):
            self.x = tf.placeholder(tf.float32,name="state_space")
            self.y = [tf.placeholder(tf.float32,name="user{}_delta".format(i)) for i in range(self.number_of_users)]
        with tf.name_scope("q_val"):
            self.q_val = [tf.reduce_sum(tf.multiply(self.x, self.w[i])) for i in range(self.number_of_users)]
        with tf.name_scope("optimizer"):
            squared_deltas = [tf.square(self.y[i] - self.q_val[i]) for i in range(self.number_of_users)]
            loss = [squared_deltas[a] for a in range(self.number_of_users)]
            optimizer = [tf.train.GradientDescentOptimizer(0.0001) for _ in range(self.number_of_users)]
            self.train = [optimizer[a].minimize(loss[a]) for a in range(self.number_of_users)]
        now = datetime.now()
        self.writer = tf.summary.FileWriter("../tensorboard/{}/{}".format(self.name, now.strftime("%H.%M.%S-%d.%m.%y")),
                                       tf.get_default_graph())
        self.update_count = 0
        scalars = []
        with tf.name_scope("summaries"):
            for i in range(self.number_of_users):
                # tf.summary.scalar("loss_user{}".format(i),tf.reduce_sum(loss[i]))
                scalars.append(tf.summary.scalar("qval_user{}".format(i),self.q_val[i]))
                tf.summary.histogram("User{}_w".format(i), w[i])
        self.merged_w = tf.summary.merge_all()

        self.merged_scalars = tf.summary.merge(scalars)


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
        merged_scalars = self.sess.run(self.merged_scalars,{self.x:states})
        self.writer.add_summary(merged_scalars,self.update_count)
        self.update_count += 1
        return q

    def update_theta(self, new_busy_times):
        old_busy_times, old_action, reward = self.history
        y = reward + self.gamma * (min(self.q(new_busy_times, a) for a in range(self.number_of_users)))
        self.sess.run(self.train[old_action], {self.x: old_busy_times, self.y[old_action]: y})
        # merged_w = self.sess.run(self.merged_w,{self.x: old_busy_times, self.y[old_action]: y})
        # self.writer.add_summary(merged_w,self.update_count)
        # self.update_count += 1