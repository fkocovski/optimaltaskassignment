import numpy as np
import randomstate.prng.pcg64 as pcg
import tensorflow as tf
from policies import *
from datetime import datetime


class BI_ONE_MC_TF(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, gamma,
                 greedy, wait_size, sess,batch_input, weights, biases,n_input):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.gamma = gamma
        self.greedy = greedy
        self.wait_size = wait_size
        self.sess = sess
        self.batch_input = batch_input
        self.weights = weights
        self.biases = biases
        self.RANDOM_STATE_ACTIONS = pcg.RandomState(1)
        self.name = "BI_ONE_MC_TF"
        self.user_slot = [None] * self.number_of_users
        self.batch_queue = []
        self.history = []
        self.rewards = []

        with tf.name_scope("input"):
            self.inp = tf.placeholder(tf.float32, name="state_space", shape=(1, n_input))
            self.gradient_input = tf.placeholder(tf.float32, name="gradient_input", shape=(self.number_of_users, 1))
            self.factor_input = tf.placeholder(tf.float32, name="factor_input")

        with tf.name_scope("neural_network"):
            layer_1 = tf.add(tf.matmul(self.inp, self.weights['h1']), self.biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            self.pred = [tf.matmul(layer_2, self.weights['out'][b]) + self.biases['out'][b] for b in range(self.batch_input)]
            self.probabilities = [tf.nn.softmax(self.pred[b]) for b in range(self.batch_input)]

        with tf.name_scope("optimizer"):
            self.cost = [tf.matmul(self.probabilities[b], self.gradient_input) for b in range(self.batch_input)]
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            self.gradients = [self.optimizer.compute_gradients(self.cost[b],
                                                               [self.weights["h1"], self.weights["h2"], self.weights["out"][b],
                                                                self.biases["b1"], self.biases["b2"], self.biases["out"][b]]) for b in
                              range(self.batch_input)]
            gradients_values = [[(g * self.factor_input, v) for g, v in self.gradients[b]] for b in range(self.batch_input)]
            self.apply = [self.optimizer.apply_gradients(gradients_values[b]) for
                          b in range(self.batch_input)]

        if self.greedy:
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
        output = self.sess.run(self.probabilities, {self.inp: state})
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

        if not self.greedy:
            self.rewards.append(self.reward(w, p, a, choices))
            self.history.append((state, output, choices))

    def state_space(self):
        # wj
        # max_w = 1000*max([self.env.now - self.batch_queue[j].arrival for j in range(self.wait_size)])
        # w = np.full(self.batch_input,max_w)
        w = np.zeros(self.batch_input)

        # w = [self.env.now - self.batch_queue[j].arrival for j in range(self.wait_size)]

        for i in range(self.batch_input):
            try:
                w[i] = self.env.now - self.batch_queue[i].arrival
            except IndexError:
                break


        # pij
        # max_pij = 1000*max([max([self.batch_queue[j].service_rate[i] for j in range(self.wait_size)]) for i in range(self.number_of_users)])
        # p = np.full((self.number_of_users,self.batch_input),max_pij)
        p = np.zeros((self.number_of_users,self.batch_input))

        # p = [[self.batch_queue[j].service_rate[i] for j in range(self.wait_size)] for i in
        #      range(self.number_of_users)]

        for i in range(self.number_of_users):
            for j in range(self.batch_input):
                try:
                    p[i][j] = self.batch_queue[j].service_rate[i]
                except IndexError:
                    break
        # flat_p = [item for sublist in p for item in sublist]
        flat_p = p.flatten()

        # ai
        a = [0 if self.user_slot[i] is None else self.user_slot[i].will_finish() - self.env.now for i
             in range(self.number_of_users)]

        # state = np.array(w + flat_p + a)
        state = np.concatenate((w,flat_p,np.array(a)))
        state = state.reshape((1, len(state)))

        return state, w, p, a

    def train(self):
        print(len(self.history))
        for t, (state, output, choices) in enumerate(self.history):
            rewards = self.discount_rewards(t)
            for job_index, preferences in enumerate(output):
                chosen_user = choices[job_index]
                prob_value = preferences.flatten()[chosen_user]
                reward = rewards[job_index]
                factor = reward / prob_value
                grad_input = np.zeros((self.number_of_users,1))
                grad_input[chosen_user] = 1.0
                h1 = self.sess.run(self.weights["h1"])
                print("===")
                print(h1.shape)
                self.sess.run(self.apply[job_index],
                              {self.inp: state, self.gradient_input: grad_input, self.factor_input: factor})
                h1 = self.sess.run(self.weights["h1"])
                print("---")
                print(h1)
                print("===")

    def reward(self, w, p, a, choices):
        reward = []
        busy_times = [a[i] for i in range(self.number_of_users)]
        for job_index, user_index in enumerate(choices):
            reward.append(w[job_index] + busy_times[user_index] + p[user_index][job_index])
            busy_times[user_index] += p[user_index][job_index]
        return reward

    def discount_rewards(self, time):
        g = np.zeros(self.batch_input)
        for t,rewards in enumerate(self.rewards[time:]):
            for i,reward in enumerate(rewards):
                g[i] += (self.gamma**t)*reward
        return g
