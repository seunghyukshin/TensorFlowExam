import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# Build Graph
#x_train = [1,2,3]
#y_train = [1,2,3]
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])
W = tf.Variable(tf.random_normal([1]), name='weight') # becaues we don't know W,b values
b = tf.Variable(tf.random_normal([1]), name='bias')



hypothesis = X*W + b #linear regression
cost = tf.reduce_mean(tf.square(hypothesis-Y)) # reduce_mean returns average
                                                    # square means double multiply
# GradientDescent
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Run/update Graph
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # "Variable" requires initializing

# Training
for step in range(2001):
    #sess.run(train)
    cost_val, W_val, b_val , _ = sess.run([cost, W, b, train],  # "_" means that train value we don't want
        feed_dict={X:[1, 2, 3, 4, 5],
                   Y:[2.1, 3.1, 4.1, 5.1, 6.1]}) # We expects W:1 b:1.1

    if step % 20 == 0:
        #print(step, sess.run(cost), sess.run(W), sess.run(b))
        print(step, cost_val, W_val, b_val )

# Test model
print(sess.run(hypothesis, feed_dict={X:[5]})) # expected 6.1
print(sess.run(hypothesis, feed_dict={X:[2.5]}))         #3.6
print(sess.run(hypothesis, feed_dict={X:[1.5 ,3.5]}))    #2.6 4.6