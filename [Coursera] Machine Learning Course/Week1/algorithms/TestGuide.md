
Fake data to use for testing the algorithm functions work correctly.

```
% Random Test Cases

X1 = [ones(20,1) (exp(1) + exp(2) * (0.1:0.1:2))'];
Y1 = X1(:,2) + sin(X1(:,1)) + cos(X1(:,2));
X2 = [X1 X1(:,2).^0.5 X1(:,2).^0.25];
Y2 = Y1.^0.5 + Y1;

% Functions calls

warmUpExercise());
computeCost(X1, Y1, [0.5 -0.5]'));
gradientDescent(X1, Y1, [0.5 -0.5]', 0.01, 10));
featureNormalize(X2(:,2:4)));
computeCostMulti(X2, Y2, [0.1 0.2 0.3 0.4]'));
gradientDescentMulti(X2, Y2, [-0.1 -0.2 -0.3 -0.4]', 0.01, 10));
normalEqn(X2, Y2));
```