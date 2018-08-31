function J = computeCost(X, y, theta)

    % COMPUTECOST Compute cost for linear regression
    %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    %   parameter for linear regression to fit the data points in X and y
    
    m = length(y);                  % The number of training examples
    predictions = X * theta;        % Prediction (ÿ)  
    sqrErr = (predictions-y) .^ 2;  % Calculate the mean squared error 
    
    % Compute the cost of the dataset based on prediction accuracy

    J = 1/(2*m) * sum(sqrErr);
    
end
