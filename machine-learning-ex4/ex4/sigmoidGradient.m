function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).
for i = 1: size(g,1)
  for j = 1: size(g,2)
    a1 = -1*z(i,j);
    a2 = e^a1;
    a3 = 1+a2;
    sigmoid  = 1/a3;
    g(i,j) = sigmoid*(1-sigmoid);
  endfor
endfor
% =============================================================
end
