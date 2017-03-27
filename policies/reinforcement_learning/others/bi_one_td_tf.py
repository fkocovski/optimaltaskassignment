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
        self.history = []
        self.rewards = []
        self.count = 0

        with tf.name_scope("input"):
            self.inp = tf.placeholder(tf.float32, name="state_space", shape=(1, n_input))
            self.gradient_input = tf.placeholder(tf.float32,name="gradient_input",shape=(self.number_of_users,1))
            self.factor_input = tf.placeholder(tf.float32,name="factor_input")

        with tf.name_scope("neural_network"):
            layer_1 = tf.add(tf.matmul(self.inp, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            self.pred = [tf.matmul(layer_2, weights['out'][b]) + biases['out'][b] for b in range(self.wait_size)]
            self.probabilities = [tf.nn.softmax(self.pred[b]) for b in range(self.wait_size)]
            self.cost = [tf.matmul(self.probabilities[b],self.gradient_input) for b in range(self.wait_size)]
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.gradients = [self.optimizer.compute_gradients(self.cost[b],[weights["h1"],weights["h2"],weights["out"][b],biases["b1"],biases["b2"],biases["out"][b]]) for b in range(self.wait_size)]
            self.apply = [[] for _ in range(self.wait_size)]
            for b in range(self.wait_size):
                for g,v in self.gradients[b]:
                    self.apply[b].append((g*self.factor_input,v))

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

        self.rewards.append(self.reward(w, p, a, choices))
        self.history.append((state, output, choices))

        # if self.count % 10 == 0:
        #     self.discount_rewards(self.count)
        # self.count += 1

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


    def update_theta(self):
        for i, (state,output,choices) in enumerate(self.history):
            rewards = self.discount_rewards(i)
            for index,preferences in enumerate(output):
                reshape_output = np.reshape(preferences,(self.number_of_users,1))
                chosen_user = choices[index]
                prob_value = output[index][0][chosen_user]
                reward = rewards[index]
                # for operation in range(len(self.gradients[index])):
                # TODO: maybe pro job input
                grads_vals = self.sess.run(self.gradients[index],{self.inp:state,self.gradient_input:reshape_output})
                factor = reward/prob_value
                # np.multiply(grads_vals[0], factor, grads_vals[0])
                # grads = grads_vals[0]
                # vals = grads_vals[1]

                # for g,v in grads_vals:
                #     np.multiply(g,factor,g)

                # print(grads_vals)

                # new_grads_vals = []
                # for var_grad,var_val in zip(grads,vals):
                #     new_grads_vals.append((var_grad*factor,var_val))


                self.sess.run(self.apply[index],{self.inp:state,self.gradient_input:reshape_output,self.factor_input:factor})


    def reward(self, w, p, a, choices):
        reward = []
        busy_times = [a[i] for i in range(self.number_of_users)]
        for job_index, user_index in enumerate(choices):
            reward.append(w[job_index] + busy_times[user_index] + p[user_index][job_index])
            busy_times[user_index] += p[user_index][job_index]
        return reward

    def discount_rewards(self, time):
        g = [0.0]*self.wait_size
        for t in range(time + 1):
            for reward_index,reward in enumerate(self.rewards[t]):
                g[reward_index] += (self.gamma ** t) * reward
        return g