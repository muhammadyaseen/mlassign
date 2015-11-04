function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

C_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sig_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

opt_history = zeros( 64, 3 );

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

%     for i = 1:length(C_vec)
%         
%         for j = 1:length(sig_vec)
%             
%             model = svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sig_vec(j)));
%             predictions = svmPredict( model, Xval );
%             val_error = mean( double( predictions ~= yval ) );
%             
%             opt_history((i-1)*length(C_vec) + j,:) = [ sig_vec(j), C_vec(i), val_error];
%             
%         end
%         
%         
%     end
%     
%     opt_history
%     
%     [val idx] = min(opt_history);
%     
%     min_cost = opt_history( idx(3), 3)
%     
%     C = opt_history( idx(3), 2)
%     sigma = opt_history( idx(3), 1)
%     
%     
%     fprintf('Program paused. Press enter to continue.\n');
%    pause;




% =========================================================================

end