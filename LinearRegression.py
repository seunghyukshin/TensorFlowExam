import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

#Build Graph
x_train = [1,2,3]
y_train = [1,2,3]
W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = x_train*W + b #linear regression
cost = tf.reduce_mean(tf.square(hypothesis-y_train)) #reduce_mean returns average
                                                    #square means double multiply
#GradientDescent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

#Run/update Graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
