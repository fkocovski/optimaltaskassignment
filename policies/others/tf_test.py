import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_train = np.random.uniform(size=100)
wt = 2
bt = 1
y_train = wt*x_train + bt + 0.01*np.random.randn(100)



res = np.linalg.lstsq(np.vstack((x_train,np.ones(100))).T,y_train)

print(res)

plt.scatter(x_train,y_train)

# plt.show()

W = tf.Variable(0.0, tf.float32)
b = tf.Variable(0.0, tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b
loss = tf.reduce_sum(tf.square(linear_model - y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# x_train = [1,2]
# y_train = [0,4]
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(10000):
  sess.run(train, {x:x_train[i%100], y:y_train[i%100]})

curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: {} b: {} loss: {}".format(curr_W, curr_b, curr_loss))