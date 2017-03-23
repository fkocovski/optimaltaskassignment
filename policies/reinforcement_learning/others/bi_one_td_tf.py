import numpy as np
import randomstate.prng.pcg64 as pcg
import tensorflow as tf
from policies import *
from datetime import datetime


class BI_ONE_TD_TF(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, gamma,
                 greedy, wait_size, sess, weights, biases, n_input):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.gamma = gamma
        self.greedy = greedy
        self.wait_size = wait_size
        self.sess = sess
        self.RANDOM_STATE_ACTIONS = pcg.RandomState(1)
        self.name = "BI_ONE_TD_TF"
        self.user_slot = [None] * self.number_of_users
        self.batch_queue = []
        self.history = None
        self.count = 0

        with tf.name_scope("input"):
            self.inp = tf.placeholder(tf.float32, name="state_space", shape=(1, n_input))

        with tf.name_scope("neural_network"):
            layer_1 = tf.add(tf.matmul(self.inp, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            self.pred = [tf.matmul(layer_2, weights['out'][b]) + biases['out'][b] for b in range(self.wait_size)]
            self.cost = [tf.nn.softmax(self.pred[b]) for b in range(self.wait_size)]
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.gradients = [optimizer.compute_gradients(self.cost[b]) for b in range(self.wait_size)]
            for g in self.gradients:
                print(g)
        tf_init = tf.global_variables_initializer()
        self.sess.run(tf_init)
        now = datetime.now()
        self.writer = tf.summary.FileWriter("../tensorboard/{}/{}".format(self.name, now.strftime("%d.%m.%y-%H.%M.%S")),
                                            tf.get_default_graph())

    def request(self, user_task, token):
        wz_one_job = super().request(user_task, token)

        self.batch_queue.append(wz_one_job)

        if len(self.batch_queue) >= self.wait_size:
            self.evaluate()

        return wz_one_job

    def release(self, wz_one_job):
        super().release(wz_one_job)

        user_to_release_index = wz_one_job.assigned_user

        self.user_slot[user_to_release_index] = None

        if len(self.batch_queue) >= self.wait_size:
            self.evaluate()

    def evaluate(self):

        state, w, p, a = self.state_space()
        output = self.sess.run(self.cost, {self.inp: state})

        choices = []

        for job_index, preferences in enumerate(output):
            user = self.RANDOM_STATE_ACTIONS.choice(np.arange(0, self.number_of_users), p=preferences.flatten())
            choices.append(user)
            if self.user_slot[user] is None:
                wz_one_job = self.batch_queue[job_index]
                self.batch_queue[job_index] = None
                self.user_slot[user] = wz_one_job
                wz_one_job.assigned_user = user
                wz_one_job.assigned = self.env.now
                wz_one_job.started = self.env.now
                wz_one_job.request_event.succeed(wz_one_job.service_rate[user])

        self.batch_queue = [job for job in self.batch_queue if job is not None]

        rewards = self.reward(w, p, a, choices)

        try:
            gradients_output = self.sess.run(self.gradients[0])

            for gradient in gradients_output:
                for grad, val in gradient:
                    print(grad)
                    print(val)
        except Exception as e:
            print(e)

        # for gradient,value in gradients[0]:
        #     print("GRADIENT")
        #     print(gradient)
        #     print("VALUE")
        #     print(value)
        #     print("TRUE VALUE")
        #     print(self.sess.run(self.weights["h1"]))
        #     print("==")
        print("------")

        if not self.greedy:
            if self.history is not None:
                # self.update_theta(state)
                pass
        self.history = (state, output, rewards, choices)

    def state_space(self):
        # wj
        w = [self.env.now - self.batch_queue[j].arrival for j in range(self.wait_size)]

        # pij
        p = [[self.batch_queue[j].service_rate[i] for j in range(self.wait_size)] for i in
             range(self.number_of_users)]
        flat_p = [item for sublist in p for item in sublist]

        # ai
        a = [0 if self.user_slot[i] is None else self.user_slot[i].will_finish() - self.env.now for i
             in range(self.number_of_users)]

        state = np.array(w + flat_p + a)
        state = state.reshape((1, len(state)))

        return state, w, p, a

    # def q(self, states, action):
    #     q = np.dot(states[action], self.theta[action])
    #     return q

    def update_theta(self):
        old_state_space, old_choices, reward = self.history
        # delta = -reward + self.gamma * (max(self.q(new_state_space, a) for a in range(self.number_of_users ** self.wait_size))) - self.q(
        #     old_state_space, old_action)
        # self.theta[old_action] += self.alpha * delta * old_state_space[old_action]

    def reward(self, w, p, a, choices):
        # reward = 0.0
        reward = []
        busy_times = [a[i] for i in range(self.number_of_users)]
        for job_index, user_index in enumerate(choices):
            # reward += w[job_index] + busy_times[user_index] + p[user_index][job_index]
            reward.append(w[job_index] + busy_times[user_index] + p[user_index][job_index])
            busy_times[user_index] += p[user_index][job_index]
        # print(reward)
        # print(w)
        # print(p)
        # print(a)
        # print(choices)
        # print("-----")
        return reward

    def score_function(self, gradient, policy_value):
        return gradient / policy_value
