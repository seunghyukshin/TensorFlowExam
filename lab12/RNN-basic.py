import tensorflow as tf
import numpy as np

'''
    Step 1. cell을 만든다
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
                                     출력의 크기를 정해줌
...
    Step 2. cell을 넘겨주고 입력데이터도 같이 넣어준다
outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)
    y       h
'''

hidden_size = 5  # output from the LSTM
input_dim = 5  # one-hot size
batch_size = 1  # one sentence
sequence_length = 6  # |ihello| == 6

idx2char = ['h', 'i', 'e', 'l', 'o']
x_data = [[0, 1, 0, 2, 3, 3]]
x_one_hot = [[[1, 0, 0, 0, 0],
              [0, 1, 0, 0, 0],
              [1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0],
              [0, 0, 0, 1, 0],
              [0, 0, 0, 1, 0]]]  # hihell
y_data = [[1, 0, 2, 3, 3, 4]]

X = tf.placeholder(tf.float32,
                   [None, sequence_length, input_dim])  # batchsize = None
Y = tf.placeholder(tf.int32, [None, sequence_length])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,
                                    state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X, initial_state=initial_state, dtype=tf.float32)

weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(  # seq2seq.sequnce_loss : 0.3 0.7 예측시 loss값 0.5
    #                        0.1 0.9 예측시 loss값 0.3
    logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_one_hot, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_one_hot})
        print(i, "loss:", l, "prediction: ", result, "true Y: ", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", ''.join(result_str))
