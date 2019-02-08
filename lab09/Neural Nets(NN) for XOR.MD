# Neural Nets (NN) for XOR

하나의 logistic regression unit으로는 XOR problem을 해결할 수 없었다.\
![1](https://user-images.githubusercontent.com/31649100/52452252-a88d7b80-2b85-11e9-8c59-52ffb0ff08e5.png)

![2](https://user-images.githubusercontent.com/31649100/52452253-a9261200-2b85-11e9-9ac7-0fc95ded2c3b.png)

세개의 unit을 사용하고, w와 b값을 정해준 후 대입하여 XOR값이 나왔다.

이 w와 b값은 어떻게 구한 것이냐?

![3](https://user-images.githubusercontent.com/31649100/52452464-66b10500-2b86-11e9-8816-c0a715bd7832.png)

왼쪽 그림과 같이 두개의 unit을 2차원배열로 표현하여 하나로 합칠 수 있다.(Multinomial Classfication에서 했었음)\
오른쪽 그림과 같은 식을 얻을 수 있다.

~~~
# NN
K = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(K, W2) + b2)
~~~


# 딥넷트웍 학습 시키기 (Backpropagation)

    예측한 값과 실제 값의 차이(error)를 뒤에서 부터 다시 보냄

