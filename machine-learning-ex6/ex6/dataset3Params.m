function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
combs = nchoosek(values, 2);
revs = zeros(size(combs));
revs(:,1) = combs(:,2);
revs(:,2) = combs(:,1);
reps = zeros(size(values, 2), 2);
reps(:, 1) = values;
reps(:, 2) = values;
combinations = [combs; revs; reps];
[rows, cols] = size(combinations);
combinations(:, cols + 1) = zeros(rows, 1);

for i = 1:rows
	C = combinations(i, 1);
	sigma = combinations(i, 2);
	model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
	predictions = svmPredict(model, Xval);
	error = mean(double(predictions ~= yval));
	combinations(i, 3) = error;
end

[~, x] = min(combinations(:, 3));
optimal_values = combinations(x, :);
C = optimal_values(:, 1);
sigma = optimal_values(:, 2);



% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
