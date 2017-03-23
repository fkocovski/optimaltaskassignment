from datetime import datetime

import simpy
import tensorflow as tf

from evaluation.statistics import calculate_statistics
from evaluation.subplot_evolution import evolution
from policies.reinforcement_learning.llqp import LLQP_TD_TF_OP
from simulations import *

with tf.name_scope("user_w"):
    w = [tf.Variable(tf.random_normal([NUMBER_OF_USERS],seed=SEED+i),name="user{}_w".format(i)) for i in range(NUMBER_OF_USERS)]
policy_name = "LLQP_TD_TF_OP_NU{}_GI{}_TRSD{}_SIM{}".format(NUMBER_OF_USERS, GENERATION_INTERVAL, SEED, SIM_TIME)

with tf.Session() as sess:
    tf_init = tf.global_variables_initializer()
    tf_init_local = tf.local_variables_initializer()
    sess.run([tf_init, tf_init_local])
    now = datetime.now()
    gamma = 0.5
    sim_time_training = SIM_TIME * 50

    env = simpy.Environment()

    policy_train = LLQP_TD_TF_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, w, gamma, False, sess)

    start_event = acquisition_process(env, policy_train, SEED, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=sim_time_training)

    env = simpy.Environment()

    file_policy = create_files("{}.csv".format(policy_name))

    policy = LLQP_TD_TF_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, w, gamma, True, sess)

    start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=SIM_TIME)

    file_policy.close()

    calculate_statistics(file_policy.name, outfile=True)
    evolution(file_policy.name, outfile=True)



