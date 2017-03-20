import simpy
import tensorflow as tf

from evaluation.subplot_evolution import evolution
from evaluation.statistics import calculate_statistics
from policies.others.bi_one_td_tf import BI_ONE_TD_TF
from simulations import *
from datetime import datetime

policy_name = "BI_ONE_TD_TF_NU{}_GI{}_TRSD{}_SIM{}".format(NUMBER_OF_USERS, GENERATION_INTERVAL, SEED, SIM_TIME)

# Network Parameters
n_input = BATCH_SIZE+BATCH_SIZE*NUMBER_OF_USERS+NUMBER_OF_USERS  # wj+pij+ai
n_out = BATCH_SIZE
# http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
mean_size = n_input+n_out/2
n_hidden_1 = int(mean_size)
n_hidden_2 = int(mean_size)


# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1]),name="h1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_out]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_out]))
}


out = [tf.Variable(tf.zeros([n_out])) for _ in range(BATCH_SIZE)]

with tf.Session() as sess:
    now = datetime.now()
    writer = tf.summary.FileWriter("../tensorboard/{}/{}".format(policy_name,now.strftime("%d.%m.%y-%H.%M.%S")), tf.get_default_graph())
    gamma = 0.5
    sim_time_training = SIM_TIME * 2

    env = simpy.Environment()

    policy_train = BI_ONE_TD_TF(env, NUMBER_OF_USERS, WORKER_VARIABILITY, None, gamma, False,BATCH_SIZE, sess,weights,biases,out)

    start_event = acquisition_process(env, policy_train, SEED, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=sim_time_training)

    env = simpy.Environment()

    file_policy = create_files("{}.csv".format(policy_name))

    policy = BI_ONE_TD_TF(env, NUMBER_OF_USERS, WORKER_VARIABILITY, file_policy, gamma, True,BATCH_SIZE, sess,weights,biases,out)

    start_event = acquisition_process(env, policy, 1, GENERATION_INTERVAL, False, None, None, None)

    env.process(start_event.generate_tokens())

    env.run(until=SIM_TIME)

    file_policy.close()

    calculate_statistics(file_policy.name, outfile=True)
    evolution(file_policy.name, outfile=True)



