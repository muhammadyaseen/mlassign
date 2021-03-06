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

    J = ((-1*y'*log( sigmoid(X*theta) ) - (1-y')*log( 1 - sigmoid(X*theta ) )) / m ) + (lambda/(2*m))*(sum( theta(2:size(theta,1),:).^2 ) );
    
    theta_1_n = theta(2:size(theta,1),:);  % theta vector, excluding theta(1)
    
    theta_u = [0; theta_1_n];
    
    grad =  (( ( sigmoid(X*theta) - y )' * X ) / m) + (lambda/m)*theta_u';
    
    %grad(1,1) = (( ( sigmoid(X*theta) - y )' * X ) / m);



% =============================================================

end
