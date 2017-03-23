import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def half_der(g):
    return g/10

RS = np.random.RandomState(42)

learning_rate = 0.01
training = 1000
display_steps = 50

x_train = RS.uniform(size=50)
Wt = 5
bt = 10
y_train = Wt * x_train + bt*RS.randn(50)

plt.scatter(x_train, y_train)
# plt.show()


W = tf.Variable(0.0, tf.float32)
b = tf.Variable(0.0, tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = tf.add(tf.multiply(W,x),b)
loss = tf.reduce_mean(tf.square(linear_model - y))
optimizer = tf.train.AdamOptimizer(learning_rate)
variables = [W,b]
derivatives = optimizer.compute_gradients(loss,variables)
hg = [(half_der(g),v) for (g,v) in derivatives]
apply_grads = optimizer.apply_gradients(hg)
# train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(training):
    for (x_tr,y_tr) in zip(x_train,y_train):
        ders = sess.run([g for (g,v) in derivatives], {x: x_tr, y: y_tr})
        print(ders,"ders")
        print(sess.run([g for g,v in hg], {x: x_tr, y: y_tr}))
        # half_grads = []
        # for der,var in ders:
        #     half_grads.append((half_der(der),var))
        # print("Original: {}\nHalf: {}".format(ders,half_grads))

        sess.run(apply_grads,{x: x_tr,y:y_tr})
        # sess.run(train, {x: x_tr, y: y_tr})

    if i % display_steps == 0:
        loss_ep,W_ep,b_ep = sess.run([loss,W,b],{x:x_train,y:y_train})
        print("Epoch: {}, W: {}, b: {}, loss: {}".format(i,W_ep,b_ep,loss_ep))

training_cost = sess.run(loss, feed_dict={x: x_train, y: y_train})

#Graphic display
plt.plot(x_train,y_train, 'ro', label='Original data')
plt.plot(x_train, sess.run(W) * x_train+ sess.run(b), label='Fitted line')
plt.legend()
plt.show()
