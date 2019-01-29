# Lecture 6-1 Softmax classification: Multinomial classification


## Logistic regression
기존의 regression 함수로는 0과 1을 출력시키기에 적합하지 않았다.
$$z=H_L(x)=Wx$$
$$g(z)={1\over 1+e^{-z}} $$
sigmoid 또는 logistic regression

![logistic regression](https://user-images.githubusercontent.com/31649100/51876179-0ca48880-23ab-11e9-9ee2-657e2087d5e1.png)
Y hat : prediction

## Multinomial classification

![lr graph](https://user-images.githubusercontent.com/31649100/51876661-afa9d200-23ac-11e9-925a-f9587712b38a.png)

A에 대하여
$\begin{bmatrix}
w_1 & w_2 & w_3 
\end{bmatrix}$
$\begin{bmatrix}
x_1 \\
x_2 \\ 
x_3 
\end{bmatrix}
=\begin{bmatrix}
w_1x_1+w_2x_2+w_3x_3
\end{bmatrix}$

B에 대하여
$\begin{bmatrix}
w_1 & w_2 & w_3 
\end{bmatrix}$
$\begin{bmatrix}
x_1 \\
x_2 \\ 
x_3 
\end{bmatrix}
=\begin{bmatrix}
w_1x_1+w_2x_2+w_3x_3
\end{bmatrix}$

C에 대하여
$\begin{bmatrix}
w_1 & w_2 & w_3 
\end{bmatrix}$
$\begin{bmatrix}
x_1 \\
x_2 \\ 
x_3 
\end{bmatrix}
=\begin{bmatrix}
w_1x_1+w_2x_2+w_3x_3
\end{bmatrix}$

합치면
$\begin{bmatrix}
w_{A1} & w_{A2} & w_{A3} \\
w_{B1} & w_{B2} & w_{B3} \\
w_{C1} & w_{C2} & w_{C3} 
\end{bmatrix}$
$\begin{bmatrix}
x_1 \\
x_2 \\ 
x_3 
\end{bmatrix}
=\begin{bmatrix}
w_{A1}x_1+w_{A2}x_2+w_{A3}x_3\\
w_{B1}x_1+w_{B2}x_2+w_{B3}x_3\\
w_{C1}x_1+w_{C2}x_2+w_{C3}x_3
\end{bmatrix}=\begin{bmatrix}
\bar{y_A}\\\bar{y_B}\\\bar{y_C}
\end{bmatrix}$

## SOFTMAX

$$S(y_i) = {e^{y_i} \over \sum_{j} e^{y_j}} $$
1. 0에서 1사이
2. sum이 1이다
   

0.7  → 1.0\
0.2  → 0.0\
0.1  → 0.0\
'one-hot' encoding으로

## Cost function : Cross-entropy

$$D(S,L) = -\sum_{i} L_i\log{({S_i})}\\
S(y)=\bar y , L=y$$ 
맞은 예측의 cost 값은 0 \
틀린 예측의 cost 값은 무한대
