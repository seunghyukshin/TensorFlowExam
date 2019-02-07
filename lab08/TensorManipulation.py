import numpy as np

t = np.array([0., 1., 2., 3., 4., 5., 6.])

print(t.ndim)  # rank
print(t.shape)  # shape

# ![a](https://user-images.githubusercontent.com/31649100/52111445-1518ef80-2647-11e9-8d8e-f1952380c14b.png)
# ![b](https://user-images.githubusercontent.com/31649100/52111469-24983880-2647-11e9-9918-df0f20a1dcaa.png)

import tensorflow as tf

sess = tf.InteractiveSession()

# Broadcasting : Shape이 달라도 연산이 가능
matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant(3.)
print(sess.run((matrix1 + matrix2)))

matrix1 = tf.constant([[1., 2.]])
matrix2 = tf.constant([[3.], [4.]])
print(sess.run((matrix1 + matrix2)))

# Argmax : 크기가 큰 인덱스의 값을 출력함
x = [[0, 1, 2],
     [2, 1, 0]]
print(sess.run(tf.argmax(x, axis=0)))  # axis=0:행을 비교 ##[1,0,0]
print(sess.run(tf.argmax(x, axis=1)))  ##[2,0]

# ★ Reshape ★ : 배열 값은 그대로 가져가고 모양만 변경
t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
print(t.shape)
print(sess.run(tf.reshape(t, shape=[-1, 3])))  # 4행3열 2차원

## squeeze : 배열을 풀어줌
print(sess.run(tf.squeeze([[0], [1], [2]])))  # [0,1,2]

## expand : squeeze 반대
print(sess.run(tf.expand_dims([0, 1, 2], 1)))

# Casting
print(sess.run(tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32)))  # 내림[1,2,3,4]
print(sess.run(tf.cast([True, False, 1 == 1], tf.int32)))  # true:1/false:0으로

# stack : 정한 axis방향으로 배열에 넣어버림
# ones_like : 배열의 모양은 그대로, 값은 모두 1로
# zeros_like : 배열의 모양은 그대로, 값은 모두 0로

# zip
arr1 = [1, 2, 3]
arr2 = [4, 5, 6]
for x, y in zip(arr1, arr2):
    print(x, y)
         #1  4
         #2  5
         #3  6

