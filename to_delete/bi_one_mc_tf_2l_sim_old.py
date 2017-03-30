import simpy
import tensorflow as tf
import time
from datetime import datetime
from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.others.bi_one_mc_tf import BI_ONE_MC_TF
from simulations import *

batch_input = 3
n_input = batch_input + NUMBER_OF_USERS * batch_input + NUMBER_OF_USERS + batch_input  # wj+pij+ai+rj
n_out = NUMBER_OF_USERS
hidden_layer_size = int((n_input+n_out)/2)
n_hidden_1 = batch_input
n_hidden_2 = batch_input
epochs = 100
gamma = 0.75
learn_rate = 0.001
var_multiplicator = 0.001
remaining_time_intervals = 5
policy_name = "{}_BI_ONE_MC_TF_2L_NU{}_GI{}_SIM{}".format(BATCH_SIZE, NUMBER_OF_USERS, GENERATION_INTERVAL,
                                                              SIM_TIME)

with tf.name_scope("weights"):
    weights = {
        'h1': tf.Variable(var_multiplicator*tf.random_normal([n_input, n_hidden_1]), name="h1"),
        'h2': tf.Variable(var_multiplicator*tf.random_normal([n_hidden_1, n_hidden_2]), name="h2"),
        'out': [tf.Variable(var_multiplicator*tf.random_normal([n_hidden_2, n_out]), name="out") for _ in
                range(batch_input)]
    }
with tf.name_scope("biases"):
    biases = {
        'b1': tf.Variable(var_multiplicator*tf.random_normal([n_hidden_1]), name="b1"),
        'b2': tf.Variable(var_multiplicator*tf.random_normal([n_hidden_2]), name="b2"),
        'out': [tf.Variable(var_multiplicator*tf.random_normal([n_out]), name="out") for _ in range(batch_input)]
    }

with tf.name_scope("input"):
    state_space_input = tf.placeholder(tf.float32, name="state_space_input", shape=(1, n_input))
    gradient_input = tf.placeholder(tf.float32, name="gradient_input", shape=(NUMBER_OF_USERS, 1))
    factor_input = tf.placeholder(tf.float32, name="factor_input")

with tf.name_scope("neural_network"):
    layer_1 = tf.add(tf.matmul(state_space_input, weights['h1']), biases['b1'])
    layer_1 = tf.nn.elu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.elu(layer_2)
    pred = [tf.add(tf.matmul(layer_2, weights['out'][b]), biases['out'][b]) for b in
                 range(batch_input)]
    probabilities = [tf.nn.softmax(pred[b]) for b in range(batch_input)]

with tf.name_scope("optimizer"):
    cost = [tf.matmul(probabilities[b], gradient_input) for b in range(batch_input)]
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    gradients = [optimizer.compute_gradients(cost[b],
                                                [weights["h1"], weights["h2"],
                                                 weights["out"][b],
                                                 biases["b1"], biases["b2"], biases["out"][b]])
                      for b in
                      range(batch_input)]
    gradients_values = [[(g * factor_input, v) for g, v in gradients[b]] for b in range(batch_input)]
    apply = [optimizer.apply_gradients(gradients_values[b]) for
                  b in range(batch_input)]


with tf.Session() as sess:
    tf_init = tf.global_variables_initializer()
    sess.run(tf_init)
    summary_h1 = tf.summary.histogram("h1", weights["h1"])
    summary_h2 = tf.summary.histogram("h2", weights["h2"])
    summary_wout = [tf.summary.histogram("wout_{}".format(b), weights["out"][b]) for b in range(batch_input)]
    summary_b1 = tf.summary.histogram("b1", biases["b1"])
    summary_b2 = tf.summary.histogram("b2", biases["b2"])
    summary_bout = [tf.summary.histogram("bout_{}".format(b), biases["out"][b]) for b in range(batch_input)]
    now = datetime.now()
    writer = tf.summary.FileWriter(
        "../tensorboard/{}/{}-{}EP".format(policy_name, now.strftime("%d.%m.%y-%H.%M.%S"), epochs),
        tf.get_default_graph())
    to_go = epochs
    for i in range(epochs):
        if i % remaining_time_intervals == 0 :
            start = time.time()
        env = simpy.Environment()

        policy_train = BI_ONE_MC_TF(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, gamma, False, BATCH_SIZE, sess,
                                    batch_input,pred,probabilities,apply,state_space_input,gradient_input,factor_input,writer,i)

        start_event = simple_process(env, policy_train, i, GENERATION_INTERVAL, False, None, None, None)

        env.process(start_event.generate_tokens())

        env.run(until=SIM_TIME)
        policy_train.train()

        if i % remaining_time_intervals == 0:
            policy_train.save_summarry(i, summary_h1)
            policy_train.save_summarry(i, summary_h2)
            policy_train.save_summarry(i, summary_b1)
            policy_train.save_summarry(i, summary_b2)
            for b in range(batch_input):
                policy_train.save_summarry(i, summary_wout[b])
                policy_train.save_summarry(i, summary_bout[b])
            end = time.time()
            elapsed = end-start
            remaining_time = time.gmtime(to_go*elapsed)
            print("finished {}, training will finish in {}".format(i,time.strftime("%H:%M:%S",remaining_time)))
            print("======")
        to_go -= 1

    env = simpy.Environment()

    file_policy = create_files("{}.csv".format(policy_name))

    policy = BI_ONE_MC_TF(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, gamma, True, BATCH_SIZE, sess,
                          batch_input,pred,probabilities,apply,state_space_input,gradient_input,factor_input,writer,1)

    start_event = simple_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=SIM_TIME)

    print("finished test")
    print("======")

    file_policy.close()

    calculate_statistics(file_policy.name, outfile=True)
    evolution(file_policy.name, outfile=True)
