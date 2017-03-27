import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def half_der(g):
    return g / 10


RS = np.random.RandomState(42)

learning_rate = 0.01
training = 1000
display_steps = 50

x_train = RS.uniform(size=50)
Wt = 5
bt = 10
y_train = Wt * x_train + bt * RS.randn(50)

plt.scatter(x_train, y_train)
# plt.show()


W = tf.Variable([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], tf.float32)
W1 = tf.Variable([[2.0], [3.0], [4.0]], tf.float32)
b = tf.Variable([[20.0], [30.0], [40.0]], tf.float32)
b1 = tf.Variable(20.0, tf.float32)
x = tf.placeholder(tf.float32, shape=(2, 1))
x1 = tf.placeholder(tf.float32, shape=(1, 1))
y = tf.placeholder(tf.float32, shape=(1, 3),name="y")
factor = tf.placeholder(tf.float32)
linear_model = tf.matmul(y, tf.add(tf.matmul(W, x), b))
linear_model1 = tf.matmul(W1, x1)


# loss = tf.reduce_mean(tf.square(linear_model - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
variables = [W, b]

grads_vals = optimizer.compute_gradients(linear_model,[W, b])
new_grads_vals = [(g*factor,v) for g,v in grads_vals]
apply = optimizer.apply_gradients(new_grads_vals)
# derivatives = optimizer.compute_gradients(loss,variables)
# hg = [(half_der(g),v) for (g,v) in derivatives]
# apply_grads = optimizer.apply_gradients(hg)
# train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# ders = sess.run(linear_model, {x: 5})
# ders_initial = sess.run(tf.gradients(linear_model, [W, b]), {x: [[1.0], [2.0]], y: [[1.0, 0.0, 0.0]]})
# ders_initial1 = sess.run(tf.gradients(tf.square(linear_model1),[W1]), {x1: [[1.0]]})
# ders = sess.run(optimizer.compute_gradients(tf.square(linear_model),[W,b]), {x: [[1.1],[2.1]]})
ders2 = sess.run(grads_vals, {x: [[1.0], [2.0]], y: [[1.0, 0.0, 0.0]]})
# for grad, val in ders2:
#     np.multiply(grad, 0.5, grad)

for g,v in ders2:
    print(g,v)
    print("===")

sess.run(apply,{x: [[1.0], [2.0]], y: [[1.0, 0.0, 0.0]],factor:0.5})
# print(ders_initial)
# print(ders2)
# print(ders_initial1)
# print("====")
# print(ders)
# print(ders,"ders")
# print(sess.run([g for g,v in hg], {x: x_tr, y: y_tr}))
# sess.run(apply_grads,{x: x_tr,y:y_tr})
# for i in range(training):
# for (x_tr,y_tr) in zip(x_train,y_train):
#     ders = sess.run([g for (g,v) in derivatives], {x: x_tr, y: y_tr})
#     print(ders,"ders")
#     print(sess.run([g for g,v in hg], {x: x_tr, y: y_tr}))
# half_grads = []
# for der,var in ders:
#     half_grads.append((half_der(der),var))
# print("Original: {}\nHalf: {}".format(ders,half_grads))

# sess.run(apply_grads,{x: x_tr,y:y_tr})
# sess.run(train, {x: x_tr, y: y_tr})

# if i % display_steps == 0:
#     loss_ep,W_ep,b_ep = sess.run([loss,W,b],{x:x_train,y:y_train})
#     print("Epoch: {}, W: {}, b: {}, loss: {}".format(i,W_ep,b_ep,loss_ep))

# training_cost = sess.run(loss, feed_dict={x: x_train, y: y_train})
#
# #Graphic display
# plt.plot(x_train,y_train, 'ro', label='Original data')
# plt.plot(x_train, sess.run(W) * x_train+ sess.run(b), label='Fitted line')
# plt.legend()
# plt.show()
