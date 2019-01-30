# cross_entropy, one_hot, reshape
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7  # 0~6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0~6, shape=(?,1)
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot  shape =(?,1,7) we don't want it
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])  # shape=(?,7)
# if input Rank N, output rank N+1
W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# softmax = exp(logits) / reduce_sum(exp(logits),dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot)

cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                X: x_data, Y: y_data
            })
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc
            ))

    pred = sess.run(prediction, feed_dict={X: x_data})
    for p, y in zip(pred, y_data.flatten()):  # fl
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

'''
one-hot-encoding 을 logit 이 아니라, hypothesis 로 해야 할 이유가 없지 않나요? 
어차피 logit 중에 젤 큰 값이 hypothesis 에서도 제일 크게 나올 텐데요.  
prediction 을 tf.argmax(logits, 1) 로 바꾸면, 프로그램에서 굳이 nn.softmax 를 한번 더 호출할 이유가 없는 것 같습니다. 
(어차피 softmax_cross_entropy_with_logits 가 softmax ==>  cross_entropy 로 이어지는 과정이기에)﻿
'''