function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

    %GRADIENTDESCENT Performs gradient descent to learn theta
    %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    %   taking num_iters gradient steps with learning rate alpha

    m         = length(y);           % number of training examples
    J_history = zeros(num_iters, 1); % Column vector of the costs after each iter
   
    for iter = 1:num_iters

        % Get the loss for the training set each iteration

        h = (X * theta - y)';        
    
        % Update the parameters

        theta(1) = theta(1) - alpha * (1/m) * h * X(:, 1);

        theta(2) = theta(2) - alpha * (1/m) * h * X(:, 2);

        % Save the cost J in every iteration for visualising loss in plot
        J_history(iter) = computeCost(X, y, theta);

    end

    % Plot the loss to see it's decline visually

    plot(J_history);

end