function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
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
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% Cost Function
z = X*theta;
h = sigmoid(z);
logh = log(h);
log_one_minus_h = log(1-h);
minusy_logh= -1*(y.*logh);
one_minus_y_log_one_minus_h = (1-y).*log_one_minus_h;
J_unregularized = ((minusy_logh - one_minus_y_log_one_minus_h).'*ones(m,1))/m;

theta_mod = theta;
theta_mod(1,1) = 0;

theta_mod_squared = theta_mod.^2;
sum_theta_squared = sum(theta_mod_squared);
regularization_of_J = lambda/(2*m)*sum_theta_squared;
J = J_unregularized + regularization_of_J;

% gradient
h_minus_y = h.-y;
Summation = h_minus_y.'*X;
grad_unregularized = Summation.'/m;

regularization_of_grad = (lambda/m) * theta_mod;
grad = grad_unregularized.+regularization_of_grad;

% =============================================================

grad = grad(:);

end
