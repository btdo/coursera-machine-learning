function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i = 1: size(g,1)
  for j = 1: size(g,2)
    a1 = -1*z(i,j);
    a2 = e^a1;
    a3 = 1+a2;
    g(i,j) = 1/a3;
  endfor
endfor

% =============================================================

end
