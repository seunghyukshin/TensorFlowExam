import  os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()
print(sess.run(hello))
# 출력된 b는 Byte literals이라는 뜻