function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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

% Cost Function
z = X*theta;
h = sigmoid(z);
logh = log(h);
log_one_minus_h = log(1-h);
minusy_logh= -1*(y.*logh);
one_minus_y_log_one_minus_h = (1-y).*log_one_minus_h;
J = ((minusy_logh - one_minus_y_log_one_minus_h).'*ones(m,1))/m;


% Note: grad should have the same dimensions as theta
% gradient
h_minus_y = h.-y;
Summation = h_minus_y.'*X;
grad = Summation.'/m;

% =============================================================

end
