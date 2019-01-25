import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
add_node = tf.add(node1,node2)

sess = tf.Session()
print("sess.run(node1,node2):",sess.run([node1,node2]))
print("sess.run(add_node):",sess.run(add_node))


