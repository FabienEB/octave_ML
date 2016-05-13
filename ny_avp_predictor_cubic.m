
clear ; close all; clc

% Average Position Predicts CPC 

% Data Sources
data = load('upload_data3.csv');
X = data(:, 1:3);
y = data(:, 4);
m = length(y);


%% Feature Normalization 

[X_norm, mu] = featureNormalize(X);

X = [ones(m, 1) X_norm];
X_ori =  [ones(m, 1) data(:, 1:2)];

% Choose alpha value / Iterations
alpha = 0.001;
num_iters = 15000;

% Init Theta and Run Gradient Descent 
sii = size(X,2);
theta = zeros(sii, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
%figure; 
%plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
%xlabel('Number of iterations');
%ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%Add testing data

for i=1:10;
  avp = i/2;
predict_cpc(i) = [1, ((avp)/mu(1)),(avp^2/mu(2)), ((avp^3)/mu(3))]* theta;
end ;

fprintf(' %f \n', predict_cpc);

csvwrite('avp_cpc_predictor.csv',predict_cpc');
