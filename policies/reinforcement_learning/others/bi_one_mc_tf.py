import numpy as np
import randomstate.prng.pcg64 as pcg
import tensorflow as tf
from policies import *


class BI_ONE_MC_TF(Policy):
    def __init__(self, env, number_of_users, worker_variability, file_policy, gamma,
                 greedy, wait_size, sess, batch_input, predictions, probabilities, apply, state_space_input,
                 gradient_input,
                 factor_input, writer,seed):
        super().__init__(env, number_of_users, worker_variability, file_policy)
        self.gamma = gamma
        self.greedy = greedy
        self.wait_size = wait_size
        self.sess = sess
        self.batch_input = batch_input

        self.predictions = predictions
        self.probabilities = probabilities
        self.apply = apply
        self.state_space_input = state_space_input
        self.gradient_input = gradient_input
        self.factor_input = factor_input
        self.writer = writer

        self.RANDOM_STATE_ACTIONS = pcg.RandomState(seed)
        self.name = "BI_ONE_MC_TF"
        self.user_slot = [None] * self.number_of_users
        self.batch_queue = []
        self.history = []
        self.rewards = []
        # if self.greedy:
        #     self.global_step = 0
        #     softmaxes = []
        #     preds = []
        #     with tf.name_scope("greedy"):
        #         for b in range(self.batch_input):
        #             softmaxes.append(tf.summary.histogram("softmax_{}".format(b), self.probabilities[b]))
        #             preds.append(tf.summary.histogram("predictions_{}".format(b), self.predictions[b]))
        #         self.summary_softmaxes = tf.summary.merge(softmaxes)
        #         self.summary_preds = tf.summary.merge(preds)
        #         self.reward_sum = tf.placeholder(tf.float32)
        #         self.summary_lateness = tf.summary.scalar("mean_lateness", tf.reduce_mean(self.reward_sum))


    def request(self, user_task, token):
        bi_one_job = super().request(user_task, token)

        self.batch_queue.append(bi_one_job)

        if len(self.batch_queue) >= self.wait_size:
            self.evaluate()

        return bi_one_job

    def release(self, bi_one_job):
        super().release(bi_one_job)

        user_to_release_index = bi_one_job.assigned_user

        self.user_slot[user_to_release_index] = None

        if len(self.batch_queue) >= self.wait_size:
            self.evaluate()

    def evaluate(self):

        state, w, p, a = self.state_space()
        output = self.sess.run(self.probabilities, {self.state_space_input: state})

        choices = [None] * self.batch_input

        for job_index, preferences in enumerate(output):
            try:
                bi_one_job = self.batch_queue[job_index]
                user = self.RANDOM_STATE_ACTIONS.choice(np.arange(0, self.number_of_users), p=preferences.flatten())
                choices[job_index] = user
                if self.user_slot[user] is None:
                    self.batch_queue[job_index] = None
                    self.user_slot[user] = bi_one_job
                    bi_one_job.assigned_user = user
                    bi_one_job.assigned = self.env.now
                    bi_one_job.started = self.env.now
                    bi_one_job.request_event.succeed(bi_one_job.service_rate[user])
            except IndexError:
                break

        self.batch_queue = [job for job in self.batch_queue if job is not None]

        # if self.greedy:
        #     if self.global_step % 10 == 0:
        #         rewards = self.reward(w, p, a, choices)
        #         sum_preds, sum_soft, sum_lat = self.sess.run(
        #             [self.summary_preds, self.summary_softmaxes, self.summary_lateness],
        #             {self.state_space_input: state, self.reward_sum: rewards})
        #         self.writer.add_summary(sum_preds, self.global_step)
        #         self.writer.add_summary(sum_soft, self.global_step)
        #         self.writer.add_summary(sum_lat, self.global_step)
        #     self.global_step += 1

        if not self.greedy:
            self.rewards.append(self.reward(w, p, a, choices))
            self.history.append((state, output, choices))

    def state_space(self):
        w = np.zeros(self.batch_input)
        r = np.zeros(self.batch_input)

        for i in range(self.batch_input):
            try:
                w[i] = self.env.now - self.batch_queue[i].arrival
                r[i] = 1
            except IndexError:
                break

        p = np.zeros((self.number_of_users, self.batch_input))

        for i in range(self.number_of_users):
            for j in range(self.batch_input):
                try:
                    p[i][j] = self.batch_queue[j].service_rate[i]
                except IndexError:
                    break

        flat_p = p.flatten()

        a = [0 if self.user_slot[i] is None else self.user_slot[i].will_finish() - self.env.now for i
             in range(self.number_of_users)]

        state = np.concatenate((w, flat_p, np.array(a), r))
        state = state.reshape((1, len(state)))

        return state, w, p, a

    def train(self):

        for t, (state, output, choices) in enumerate(self.history):
            disc_rewards = self.discount_rewards(t)
            tmp_choices = [choice for choice in choices if choice is not None]
            for job_index, chosen_user in enumerate(tmp_choices):
                prob_value = output[job_index].flatten()[chosen_user]
                reward = disc_rewards[job_index]
                factor = reward / prob_value
                grad_input = np.zeros((self.number_of_users, 1))
                grad_input[chosen_user] = 1.0
                self.sess.run(self.apply[job_index],
                              {self.state_space_input: state, self.gradient_input: grad_input,
                               self.factor_input: factor})

    def reward(self, w, p, a, choices):
        reward = np.zeros(self.batch_input)
        busy_times = [a[i] for i in range(self.number_of_users)]
        tmp_choices = [choice for choice in choices if choice is not None]
        for job_index, user_index in enumerate(tmp_choices):
            reward[job_index] = w[job_index] + busy_times[user_index] + p[user_index][job_index]
            busy_times[user_index] += p[user_index][job_index]
        return reward

    def discount_rewards(self, time):
        g = np.zeros(self.batch_input)
        for t, rewards in enumerate(self.rewards[time:]):
            for i, reward in enumerate(rewards):
                g[i] += (self.gamma ** t) * reward
        return g

    def save_summary(self, step, summary):
        summary = self.sess.run(summary)
        self.writer.add_summary(summary, step)
