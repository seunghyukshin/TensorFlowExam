# Better non-linearity
    Neural Network 에서는 sigmoid 사용하지 않는다.
    deep해질수록(hidden layer 多) 처음에서 결과에 주는 영향력이 사라짐. 
        →  Vanish Gradient현상
        
    'RELU'를 사용한다.
    h = x (x>0)
      = 0 (x<=0)

# Weight 초기화를 잘해보자
## weight값을 조절을 잘해서 Vanish gradient현상을 줄여보는 방향
    not all 0
    
### DEEP BELIEF NETWORK (DBN)
두개의 Layer끼리씩 값을 주고받아 weight형성
~~~
W = np.random.randn(fan_in, fan_out)/np.sqrt(fan_in)
~~~
fan_inㄱ