import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]

y_data = [[152.], [185.], [180.], [196.], [142.]]

X = tf.placeholder(tf.float32, shape=[None,3])  # Now 5 element, but it will have N numbers.
Y = tf.placeholder(tf.float32, shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')


hypothesis = tf.matmul(X,W)+ b # matmul means matrix multiply

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis-Y))
# Minimize.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5) #very small learning rate
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                         feed_dict = {X:x_data, Y:y_data})

    if step % 10 == 0 :
        print(step, "\nCost:",  cost_val, "\nPrediction: \n", hy_val)




