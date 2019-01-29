1. Learning rate
2. Data preprocessing
3. Overfitting


# Learning rate
Gradient descent에서 알파값 = learning_rate\


## overshooting일 경우
cost가 줄어들지 않음. 밥그릇 바깥으로 튕겨나감. 학습 또한 되지않음

## small일 경우
too long ... ... 

## Tip
0.01 넣어보고 시작해본다.
발산이 되면 작게\
늦게 움직이면 크게


# Data preprocessing
data값의 차이가 큰경우\
learning rate 맞게 했는데 이상한 결과...\
**nomarlize로 선처리하자!**

## Standardization

$$x'_j = {x_j-μ_j \over σ_j}$$

```
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
```

# Overfitting
:학습 데이터에만 맞게끔 모델링된 것.

## Solution
- 많은 학습 데이터!
- features의 갯수를 줄이기!
- **Regularization**(일반화)

## Regularization
구부리지말고 펴자!\
Term을 추가하면됨.
$$ + λ \sum{W^2} \\
 λ: regularization strength $$



# Learning & Test data set

# evaluation: is this good ??
## 데이터의 전부를 train 하지마라.
## 30%는 test set으로!!


α(learnig rate)와 λ(regularziation strength)값을 조절하기위해 training도 나눌 필요있다.\
Training set과 Validation set으로

- Training set (교과서)
- Validation set (모의고사)
- Training set (수능)
  
# Online learning
이전에 학습한 데이터를 다시 학습 할 필요 없음

