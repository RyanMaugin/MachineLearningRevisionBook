function [X_norm, mu, sigma] = featureNormalize(X)

    % FEATURENORMALIZE Normalizes the features in X 
    %   FEATURENORMALIZE(X) returns a normalized version of X where
    %   the mean value of each feature is 0 and the standard deviation
    %   is 1. This is often a good preprocessing step to do when
    %   working with learning algorithms.

    X_norm = X;                     % Holds normalised m x n dimension feature matrix
    mu     = zeros(1, size(X, 2));  % Holds 1 x n mean vector for each feature column in X
    sigma  = zeros(1, size(X, 2));  % Holds 1 x n standard deviation vector for each feature column in X

    % Normalise X using the mean normalisation formula
    mu     = X - mean(X);
    sigma  = std(X);
    X_norm = mu ./ sigma;

end
