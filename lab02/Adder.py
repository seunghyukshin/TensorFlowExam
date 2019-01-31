import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0)
# add_node = tf.add(node1,node2)
#
# sess = tf.Session()
# print("sess.run(node1,node2):",sess.run([node1,node2]))
# print("sess.run(add_node):",sess.run(add_node))

# 1.BuildGraph
graph = tf.Graph()
with graph.as_default():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    add_node = a + b

# 2. RunGraph & 3. Update variables
sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter('./first_graph', graph)
print(sess.run(add_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(add_node, feed_dict={a: [1, 3], b: [2, 4]}))
writer.flush()
writer.close()
sess.close()