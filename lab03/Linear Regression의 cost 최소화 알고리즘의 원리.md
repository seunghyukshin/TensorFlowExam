# Minmalize cost
<table>
<tr><th>x</th><th>y</th></tr>
<tr><th>1</th><th>1</th></tr>
<tr><th>2</th><th>2</th></tr>
<tr><th>3</th><th>3</th></tr>
</table> 
→ cost/W 그래프 모양 : 아래로 볼록

→ Gradient descent algorithm : 시작점이 달라도 도착 지점은 한 곳\
: minimize cost function

$$W := W- α{\delta \over  \delta W} cost(W)$$

$$W := W- α{1 \over m}\sum_{i=1}^m ( H(x_i )-y_i )^2 $$

밥그릇 모양의 그래프 모양이라면 gradient decent algorithm 가능하다.

