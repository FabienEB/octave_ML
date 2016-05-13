

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
alpha = 0.01;
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
 
%--------------------------------------------------------------------
% Average Position Predicts CTR  

% Data Sources
data = load('upload_data1.csv');
X = data(:, 5:7);
y = data(:, 8);
m = length(y);


%% Feature Normalization 

[X_norm, mu] = featureNormalize(X);

X = [ones(m, 1) X_norm];
X_ori =  [ones(m, 1) data(:, 1:2)];

% Choose alpha value / Iterations
alpha = 0.01;
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
predict_ctr(i) = [1, ((avp)/mu(1)),(avp^2/mu(2)), ((avp^3)/mu(3))]* theta;
end ;

%--------------------------------------------------------------------
% Average Position Predicts clicks 

% Data Sources
data = load('upload_data1.csv');
X = data(:, 9:11);
y = data(:, 12);
m = length(y);


%% Feature Normalization 

[X_norm, mu] = featureNormalize(X);

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

%Add testing data

for i=1:10;
  avp = i/2;
predict_click(i) = [1, ((avp)/mu(1)),(avp^2/mu(2)), ((avp^3)/mu(3))]* theta;
end ;

%--------------------------------------------------------------------
% Average Position Predicts clicks 

% Data Sources '
data = load('upload_data1.csv');
X = data(:, 13:15);
y = data(:, 16);
m = length(y);


%% Feature Normalization 

[X_norm, mu] = featureNormalize(X);

Xc = [ones(m, 1) X_norm];
X_ori =  [ones(m, 1) data(:, 1:2)];

% Choose alpha value / Iterations
alpha = 0.01;
num_iters = 5000;

% Init Theta and Run Gradient Descent 
sii = size(Xc,2);
theta = zeros(sii, 1);
[theta, J_history] = gradientDescentMulti(Xc, y, theta, alpha, num_iters);


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
pause;


%Add testing data

for i=1:10;
  avp = i/2;
predict_conv(i) = [1, ((avp)/mu(1)),(avp^2/mu(2)), ((avp^3)/mu(3))]* theta;
end ;

final_predict= [predict_cpc',predict_ctr',predict_click',predict_conv'];

csvwrite('avp_predictor.csv',final_predict);


fprintf('Visualizing J(theta_0, theta_1) ...\n')


% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
