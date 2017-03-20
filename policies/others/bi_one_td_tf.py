import randomstate.prng.pcg64 as pcg
import numpy as np
import tensorflow as tf
from policies import *

class BI_ONE_TD_TF(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, theta, gamma, alpha,
                 greedy, wait_size,sess,w):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.theta = theta
        self.gamma = gamma
        self.alpha = alpha
        self.greedy = greedy
        self.wait_size = wait_size
        self.RANDOM_STATE_ACTIONS = pcg.RandomState(1)
        self.name = "BI_ONE_TD_TF"
        self.user_slot = [None] * self.number_of_users
        self.batch_queue = []
        self.history = None

        self.sess = sess


        # Network Parameters
        n_input = self.wait_size+self.wait_size*self.number_of_users+self.number_of_users  # wj+pij+ai
        n_out = self.wait_size
        # http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        mean_size = n_input+n_out/2
        n_hidden_1 = mean_size
        n_hidden_2 = mean_size

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_out]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_out]))
        }


        self.inp = tf.placeholder(tf.float32,shape=n_input)
        self.out = [tf.Variable(tf.zeros([n_out])) for _ in range(self.wait_size)]

        # Construct model
        pred = self.multilayer_perceptron(self.inp, weights, biases)

        cost = tf.reduce_mean(tf.nn.softmax(pred))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    def request(self, user_task,token):
        wz_one_job = super().request(user_task,token)

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

        state = self.state_space()
        
        output = self.sess.run(self.out, {self.inp:state})

        for user,preferences in enumerate(output):
            job = self.RANDOM_STATE_ACTIONS.choice(preferences,p=preferences)
            if self.user_slot[user] is None:
                wz_one_job = self.batch_queue[job_index]
                self.batch_queue[job_index] = None
                self.user_slot[user_index] = wz_one_job
                wz_one_job.assigned_user = user_index
                wz_one_job.assigned = self.env.now
                wz_one_job.started = self.env.now
                wz_one_job.request_event.succeed(wz_one_job.service_rate[user_index])

        self.batch_queue = [job for job in self.batch_queue if job is not None]

        if not self.greedy:
            if self.history is not None:
                self.update_theta(state_space)

        self.history = (state_space, action, combinations)

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

        state = w+flat_p+a

        return state

    def q(self, states, action):
        q = np.dot(states[action], self.theta[action])
        return q

    def update_theta(self, new_state_space):
        old_state_space, old_action, old_combinations = self.history
        reward = self.reward(old_state_space, old_action, old_combinations)
        delta = -reward + self.gamma * (
        max(self.q(new_state_space, a) for a in range(self.number_of_users ** self.wait_size))) - self.q(
            old_state_space, old_action)
        self.theta[old_action] += self.alpha * delta * old_state_space[old_action]

    def reward(self, state_space, action, combinations):
        reward = 0.0
        busy_times = [state_space[action][self.wait_size + i] for i in range(self.number_of_users)]
        for job_index, user_index in enumerate(combinations[action]):
            reward += state_space[action][job_index] + busy_times[user_index] + state_space[action][2 * self.wait_size + user_index]
            busy_times[user_index] += state_space[action][2 * self.wait_size + user_index]
        return reward

    # Create model
    def multilayer_perceptron(self,x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer
