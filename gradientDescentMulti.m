function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters


si = size(X,2);
t = zeros(si ,1);
J = computeCost(X, y, theta);
t = theta - ((alpha*((theta'*X') - y'))*X/m)';
theta = t;
J1 = computeCost(X, y, theta);

if(J1>J),
    break,fprintf('Wrong alpha');
else if(J1==J)
    break;
end;


% ========================== END ==============================

% Save the cost J in every iteration    
J_history(iter) = sum(computeCost(X, y, theta));

 
end

end







