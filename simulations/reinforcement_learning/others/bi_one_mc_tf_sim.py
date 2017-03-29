import simpy
import tensorflow as tf
import time
from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.others.bi_one_mc_tf import BI_ONE_MC_TF
from simulations import *

batch_input = 2
n_input = batch_input + NUMBER_OF_USERS * batch_input + NUMBER_OF_USERS + batch_input  # wj+pij+ai+rj
n_out = NUMBER_OF_USERS
hidden_layers_size = int((n_input+n_out)/2)
n_hidden_1 = hidden_layers_size
n_hidden_2 = hidden_layers_size
epochs = 1000
gamma = 0.5
learn_rate = 0.001
var_multiplicator = 0.0001
policy_name = "{}_BI_ONE_MC_TF_NU{}_GI{}_TRSD{}_SIM{}".format(batch_input, NUMBER_OF_USERS, GENERATION_INTERVAL, SEED,
                                                              SIM_TIME)

with tf.name_scope("weights"):
    weights = {
        'h1': tf.Variable(var_multiplicator*tf.random_normal([n_input, n_hidden_1]), name="h1"),
        'h2': tf.Variable(var_multiplicator*tf.random_normal([n_hidden_1, n_hidden_2]), name="h2"),
        'out': [tf.Variable(var_multiplicator*tf.random_normal([n_hidden_2, n_out]), name="out") for b in
                range(batch_input)]
    }
with tf.name_scope("biases"):
    biases = {
        'b1': tf.Variable(var_multiplicator*tf.random_normal([n_hidden_1]), name="b1"),
        'b2': tf.Variable(var_multiplicator*tf.random_normal([n_hidden_2]), name="b2"),
        'out': [tf.Variable(var_multiplicator*tf.random_normal([n_out]), name="out") for b in range(batch_input)]
    }

with tf.name_scope("input"):
    state_space_input = tf.placeholder(tf.float32, name="state_space_input", shape=(1, n_input))
    gradient_input = tf.placeholder(tf.float32, name="gradient_input", shape=(NUMBER_OF_USERS, 1))
    factor_input = tf.placeholder(tf.float32, name="factor_input")

with tf.name_scope("neural_network"):
    layer_1 = tf.add(tf.matmul(state_space_input, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    pred = [tf.add(tf.matmul(layer_2, weights['out'][b]), biases['out'][b]) for b in
                 range(batch_input)]
    probabilities = [tf.nn.softmax(pred[b]) for b in range(batch_input)]

with tf.name_scope("optimizer"):
    cost = [tf.matmul(probabilities[b], gradient_input) for b in range(batch_input)]
    optimizer = [tf.train.AdamOptimizer(learning_rate=learn_rate) for _ in range(batch_input)]
    gradients = [optimizer[b].compute_gradients(cost[b],
                                                          [weights["h1"], weights["h2"],
                                                           weights["out"][b],
                                                           biases["b1"], biases["b2"], biases["out"][b]])
                      for b in
                      range(batch_input)]
    gradients_values = [[(g * factor_input, v) for g, v in gradients[b]] for b in range(batch_input)]
    apply = [optimizer[b].apply_gradients(gradients_values[b]) for
                  b in range(batch_input)]


with tf.Session() as sess:
    tf_init = tf.global_variables_initializer()
    sess.run(tf_init)
    to_go = epochs
    for i in range(epochs):
        if i % 100 == 0 :
            start = time.time()
        env = simpy.Environment()

        policy_train = BI_ONE_MC_TF(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, gamma, False, BATCH_SIZE, sess,
                                    batch_input,probabilities,apply,state_space_input,gradient_input,factor_input,None)

        start_event = acquisition_process(env, policy_train, SEED, GENERATION_INTERVAL, False, None, None, None)

        env.process(start_event.generate_tokens())

        env.run(until=SIM_TIME)
        policy_train.train()
        if i % 100 == 0:
            end = time.time()
            elapsed = end-start
            remaining_time = time.gmtime(to_go*elapsed)
            print("finished {}, training will finish in {}".format(i,time.strftime("%H:%M:%S",remaining_time)))
            print("======")
        to_go -= 1

    env = simpy.Environment()

    file_policy = create_files("{}.csv".format(policy_name))

    policy = BI_ONE_MC_TF(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, gamma, True, BATCH_SIZE, sess,
                          batch_input,probabilities,apply,state_space_input,gradient_input,factor_input,epochs)

    start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=SIM_TIME)

    print("finished test")
    # wh1 = sess.run(weights["h1"])
    # print(wh1)
    print("======")

    file_policy.close()

    calculate_statistics(file_policy.name, outfile=True)
    evolution(file_policy.name, outfile=True)
