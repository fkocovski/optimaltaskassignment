import simpy
import tensorflow as tf
from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.reinforcement_learning.others.bi_one_mc_tf import BI_ONE_MC_TF
from simulations import *

policy_name = "{}_BI_ONE_MC_TF_NU{}_GI{}_TRSD{}_SIM{}".format(BATCH_SIZE, NUMBER_OF_USERS, GENERATION_INTERVAL, SEED,
                                                              SIM_TIME)
n_input = BATCH_SIZE + NUMBER_OF_USERS * BATCH_SIZE + NUMBER_OF_USERS  # wj+pij+ai
n_out = NUMBER_OF_USERS
n_hidden_1 = n_input * 10
n_hidden_2 = n_input * 10
epochs = 5

with tf.name_scope("weights"):
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1],seed=SEED), name="h1"),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2],seed=SEED+1), name="h2"),
        'out': [tf.Variable(tf.random_normal([n_hidden_2, n_out],seed=SEED+2), name="out") for _ in range(BATCH_SIZE)]
    }
with tf.name_scope("biases"):
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1],seed=SEED+3), name="b1"),
        'b2': tf.Variable(tf.random_normal([n_hidden_2],seed=SEED+4), name="b2"),
        'out': [tf.Variable(tf.random_normal([n_out],seed=SEED+5), name="out") for _ in range(BATCH_SIZE)]
    }

with tf.Session() as sess:
    gamma = 0.5
    sim_time_training = SIM_TIME * 2
    for i in range(epochs):
        env = simpy.Environment()

        policy_train = BI_ONE_MC_TF(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, gamma, False, BATCH_SIZE, sess, weights,
                                    biases, n_input)

        start_event = acquisition_process(env, policy_train, SEED, GENERATION_INTERVAL, False, None, None, None)

        env.process(start_event.generate_tokens())

        env.run(until=sim_time_training)
        policy_train.update_theta()
        print("finished {}".format(i))

    env = simpy.Environment()

    file_policy = create_files("{}.csv".format(policy_name))

    policy = BI_ONE_MC_TF(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, gamma, True, BATCH_SIZE, sess, weights,
                          biases, n_input)

    start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=SIM_TIME)

    file_policy.close()

    calculate_statistics(file_policy.name, outfile=True)
    evolution(file_policy.name, outfile=True)
