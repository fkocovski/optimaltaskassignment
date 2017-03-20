import simpy
import tensorflow as tf

from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.llqp.llqp_td_tf_op import LLQP_TD_TF_OP
from simulations import *

# w = [tf.Variable(tf.zeros([NUMBER_OF_USERS])) for _ in range(NUMBER_OF_USERS)]
w = [tf.Variable(tf.random_normal([NUMBER_OF_USERS],seed=SEED)) for _ in range(NUMBER_OF_USERS)]
with tf.Session() as sess:
    tf_init = tf.global_variables_initializer()
    sess.run(tf_init)
    gamma = 0.5
    sim_time_training = SIM_TIME * 100

    env = simpy.Environment()

    policy_train = LLQP_TD_TF_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, w, gamma, False, sess)

    start_event = acquisition_process(env, policy_train, SEED, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=sim_time_training)

    env = simpy.Environment()

    file_policy = create_files(
        "{}_NU{}_GI{}_TRSD{}_SIM{}.csv".format(policy_train.name, NUMBER_OF_USERS, GENERATION_INTERVAL, SEED, SIM_TIME))

    policy = LLQP_TD_TF_OP(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, w, gamma, True, sess)

    start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=SIM_TIME)

    file_policy.close()

    calculate_statistics(file_policy.name, outfile=True)
    evolution(file_policy.name, outfile=True)
