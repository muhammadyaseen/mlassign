function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

%  J =     ( Sum( h(x_i) - y_i )^2 ) * (0.5/m)

    %theta1_times_x0 = theta(1) * X(:,1:1);
    %theta2_times_x1 = theta(2) * X(:,2:2);
    
    %h = theta1_times_x0 + theta2_times_x1;
    
   J = (sum( ( X*theta - y ).^2) ) / (2*m);

% =========================================================================

end