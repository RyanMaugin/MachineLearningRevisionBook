function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

    % GRADIENTDESCENTMULTI Performs gradient descent to learn theta
    %   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
    %   taking num_iters gradient steps with learning rate alpha

    m = length(y);                   % number of training examples
    J_history = zeros(num_iters, 1); % Keeps track of cost function result as each step of GD step happens

    % Start performing gradient descent for the number of iterations passed in

    for iter = 1:num_iters

        % Get the loss transposed cost vector

        h = (X * theta - y)';        
        
        % Update each parameter theta

        for k=1:length(theta)
            theta(k) = theta(k) - (alpha * ((1/m) * h * X(:, k)));        
        end

        % Save the cost J in every iteration    
        J_history(iter) = computeCostMulti(X, y, theta);
        
    end

end
