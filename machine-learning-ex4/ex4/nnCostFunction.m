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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Theta1 has size 25 x 401
% Theta2 has size 10 x 26
% 400+1 input layer units, 25+1 hidden layer units and 10 output units
    
    size(Theta1)
    size(Theta2)
    
    X = [ones(size(X,1),1) X];

    %a_1 = X;
    
    z_2 = Theta1*X';
    
    a_2 = sigmoid(z_2);
    
    a_2 = [ones(1,size(a_2,2)); a_2];
        
    %size(a_2)
    
    %size(Theta2)
    
    z_3 = Theta2*a_2;
    
    a_3 = sigmoid( z_3 )';      %a_3 is h_theta(x)

    size(a_3)
    
    f = recode(y,num_labels);
    
    %J = ( sum( sum( ( -1*f'*log( a_3 ) - (1-f')*log( 1 - a_3) ) ) ) / m );
    % this vectorized impl is not working, so using loop based impl.
    
    for i = 1:m
    
        for k = 1:num_labels
           
            J = J +  -1*f(i,k)*log(a_3(i,k)) - (1 - f(i,k))*log(1-a_3(i,k));
            
        end
        
    end
    
    J = J/m;
    
    %compute regularizaion term for Theta1
    
    reg = 0;
    
    for j=1:size(Theta1,1)
        
        for k=2:size(Theta1,2)
                
            reg = reg + Theta1(j,k)^2;
        end
 
    end
    
    %compute regularizaion term for Theta2
    
    for j=1:size(Theta2,1)
        
        for k=2:size(Theta2,2)
                
            reg = reg + Theta2(j,k)^2;
        end
 
    end
    
    reg = (lambda * reg) / (2*m);
    
    J = J + reg;
    
    % Backpropagation algorithm
    
    grad1_acc = 0;
    grad2_acc = 0;
    
    for i = 1:m
       
        a_1 = X(i,:);           %get i-th row i.e. ith training example
        
        z_2 = Theta1 * a_1';
        
        a_2 = [1; sigmoid(z_2)];  % (25+1 x 401)(1 x 401)' ==> (25+1 x 1)
        
        a_3 = sigmoid( Theta2 * a_2);   % (10x26)(26x1) ==> (10 x 1)
        
        
        delta_3 = a_3 - f(i,:)';
        
        delta_2 = Theta2' * delta_3 .* sigmoidGradient( [1; z_2] );
        
        grad1_acc = grad1_acc + delta_2(2:end) * a_1;
        
        grad2_acc = grad2_acc + delta_3 * a_2';
        
    end
    
    Theta1_grad = (grad1_acc / m) + (lambda/m) * [zeros(size(Theta1,1),1) Theta1(:,2:end)];
    
    Theta2_grad = (grad2_acc / m) + (lambda/m) * [zeros(size(Theta2,1),1) Theta2(:,2:end)];
    
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
