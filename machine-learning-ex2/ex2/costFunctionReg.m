function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


[J_unregularized, grad_unregularized] = costFunction(theta, X, y);

theta_squared = theta.^2;
sum_theta_matrix = ones(length(theta),1);
sum_theta_matrix(1,1) = 0;
theta_sum_squared = theta_squared.'*sum_theta_matrix;
regularized = (lambda*theta_sum_squared)/(2*m);
J = J_unregularized + regularized;

regularized_derivative = (lambda/m)*theta;
regularized_derivative(1,1) = 0;
grad = grad_unregularized.+regularized_derivative;


% =============================================================

end
