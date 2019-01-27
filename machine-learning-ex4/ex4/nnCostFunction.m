function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

a_1 = [ones(m, 1) X];
z_2 = a_1*Theta1.';
a_2 = sigmoid(z_2);
a_2 = [ones(m, 1) a_2];
z_3 = a_2*Theta2.';
a_3 = sigmoid(z_3);
hTheta =  a_3;
log_hTheta  = log(hTheta);
log_1_minus_hTheta = log(1-hTheta);
J_matrix = zeros(m,1);
for i=1:m
  y_vector = zeros(num_labels,1);  
  y_vector(y(i,1),1)=1;  
  log_hTheta_row = log_hTheta(i,:);
  log_1_minus_hTheta_row = log_1_minus_hTheta(i,:);
  J_matrix(i,1) = -1*(log_hTheta_row*y_vector)-(log_1_minus_hTheta_row*(1-y_vector));  
endfor

J_unreg = sum(J_matrix,1)/m;

% Theta without bias
Theta1_without_bias = Theta1(:,2:size(Theta1,2));
Theta1_square = Theta1_without_bias.^2;
Theta1_square_sum_rows = sum(Theta1_square,2);
Theta1_square_sum_columns = sum(Theta1_square_sum_rows,1);

Theta2_without_bias = Theta2(:,2:size(Theta2,2));
Theta2_square = Theta2_without_bias.^2;
Theta2_square_sum_rows = sum(Theta2_square,2);
Theta2_square_sum_columns = sum(Theta2_square_sum_rows,1);

regularization = (lambda*(Theta1_square_sum_columns + Theta2_square_sum_columns))/(2*m);

J = J_unreg + regularization;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

accum_delta_1 = zeros(size(Theta1));
accum_delta_2 = zeros(size(Theta2));

for t=1:m
  %1
  a1 = [X(t,:)]';
  a1 = [1 ; a1];
  z2 = Theta1*a1;
  a2 = sigmoid(z2);
  a2 = [1 ; a2];
  z3 = Theta2*a2;
  a3 = sigmoid(z3); 
  
  %2
  yk_vector = zeros(num_labels,1);  
  yk_vector(y(t,1),1)=1;  
  delta3 = a3 - yk_vector;
  
  %3
  Theta2Mod = Theta2(:,2:end); % remove bias
  delta2 = Theta2Mod'*delta3.*sigmoidGradient(z2);
  
  %delta2 = Theta2'*delta3.*(a2.*(1-a2));
  %delta2 = delta2(2:end);
  
  %4  
  accum_delta_1 = accum_delta_1 + delta2*a1';
  accum_delta_2 = accum_delta_2 + delta3*a2';
  
endfor

Theta1_grad_unreg = (1/m)*accum_delta_1;
Theta2_grad_unreg = (1/m)*accum_delta_2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

regularized_1 = zeros(size(Theta1_grad_unreg));
regularized_1_size = size(regularized_1);

for j=1:regularized_1_size(1,1)
  for i=1:regularized_1_size(1,2)
    if j == 1
      regularized_1(j,i) = 0;
    elseif i == 1
      regularized_1(j,i) = 0;
    else
      regularized_1(j,i) = (lambda/m)*Theta1(j,i);
    endif    
  endfor  
endfor

Theta1_grad = Theta1_grad_unreg + regularized_1;

regularized_2 = zeros(size(Theta2_grad_unreg));
regularized_2_size = size(regularized_2);
for j=1:regularized_2_size(1,1)
  for i=1:regularized_2_size(1,2)
    if j == 1
      regularized_1(j,i) = 0;
    elseif i == 1
      regularized_1(j,i) = 0;
    else
      regularized_1(j,i) = (lambda/m)*Theta2(j,i);
    endif    
  endfor  
endfor

Theta2_grad = Theta2_grad_unreg + regularized_2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
