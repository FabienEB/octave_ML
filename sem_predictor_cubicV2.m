

clear ; close all; %clc

% Average Position Predicts CPC 

% Data Sources
data = load('upload_data3.csv');
X = data(:, 1:4);
y = data(:, 5);
m = length(y);


%% Feature Normalization 

[X_norm, mu] = featureNormalize_day(X);

X = [ones(m, 1) X_norm];
X_ori =  [ones(m, 1) data(:, 1:2)];

% Choose alpha value / Iterations
alpha = 0.01;
num_iters = 5000;

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

fprintf(' 1- friday  \n');
fprintf(' 2- Saturday \n');
fprintf(' 3- Sunday \n');
fprintf(' 4- Monday\n');
fprintf(' 5- Tuesday\n');
fprintf(' 6- Wednesday\n');
fprintf(' 7- Thursday  \n');
fprintf('\n');



%Add testing data

for i=1:10;
  avp = i/2;
predict_cpc(i) = [1, (7/mu(1)),((avp)/mu(2)),(avp^2/mu(3)), ((avp^3)/mu(4))]* theta;
end ;
 
final_predict= predict_cpc';

csvwrite('avp_predictor.csv',final_predict);

fprintf('Average position : \n');
fprintf(' %f \n', final_predict);

