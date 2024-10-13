% XOR inputs (x1, x2)
x = [0 0;...
     0 1;...
     1 0;...
     1 1];

% desired output (target)
d = [0;   
     1;    
     1;    
     0];

% learning rate
myu = 0.1;

% initial weights for inputs and bias
w0 = 0.05; % bias
w1 = 0.05; % weight for x1
w2 = 0.05; % weight for x2
w3 = 0.05; % weight for x1*x2 (interaction term)
w4 = 0.05; % weight for x1^2
w5 = 0.05; % weight for x2^2

% params
alpha = 11;
convergence_threshold = 0.00001;
mse = 0;
ii  = 1;

% initialize weight storage matrix
max_epochs = 1000;
w_mat = zeros(6, max_epochs);
index = 1; 
epoch = 0;

% loop until convergence
while true
    if index > 4
        index = 1;
        epoch = epoch + 1;
    end
    
    % extract current inputs x1 and x2
    x1 = x(index, 1);
    x2 = x(index, 2);
    
    % nonlinear input transformation
    z1 = x1;           % raw x1
    z2 = x2;           % raw x2
    z3 = x1 * x2;      % Interaction term
    z4 = x1^2;         % Nonlinear term for x1
    z5 = x2^2;         % Nonlinear term for x2
    
    % linear combination of transformed inputs with weights
    v = w0 + w1*z1 + w2*z2 + w3*z3 + w4*z4 + w5*z5;
    
    % sigmoid activation function
    y = 1 / (1 + exp(-alpha * v));
    
    % error calculation (target - sigmoid output)
    error = d(index) - y;
    
    % mse
    mse = 0.5 * (error^2);

    % check for convergence
    if mse < convergence_threshold
        break;
    end

    fprintf('Epoch: %d, Iteration: %d, Input Index: %d, Weights: [w0: %.4f, w1: %.4f, w2: %.4f, w3: %.4f, w4: %.4f, w5: %.4f], Output: %.4f, Error: %.4f, MSE: %.4f\n', ...
        epoch, ii, index, w0, w1, w2, w3, w4, w5, y, error, mse);

    % weight updates (gradient descent)
    w0 = w0 + myu * error * 1;   % bias
    w1 = w1 + myu * error * z1;  % z1 (x1)
    w2 = w2 + myu * error * z2;  % z2 (x2)
    w3 = w3 + myu * error * z3;  % z3 (x1*x2)
    w4 = w4 + myu * error * z4;  % z4 (x1^2)
    w5 = w5 + myu * error * z5;  % z5 (x2^2)
    
    if epoch > 0
        w_mat(1, epoch) = w0;
        w_mat(2, epoch) = w1;
        w_mat(3, epoch) = w2;
        w_mat(4, epoch) = w3;
        w_mat(5, epoch) = w4;
        w_mat(6, epoch) = w5;
    end
    
    index = index + 1;
    ii = ii + 1;
end

% check outputs after training
fprintf('\nChecking outputs after training:\n');
for i = 1:size(x, 1)
    x1 = x(i, 1);
    x2 = x(i, 2);
    
    z1 = x1;            
    z2 = x2;            
    z3 = x1 * x2;   
    z4 = x1^2;        
    z5 = x2^2;        
    
    v = w0 + w1*z1 + w2*z2 + w3*z3 + w4*z4 + w5*z5;
    y = 1 / (1 + exp(-alpha * v));
    
    fprintf('Input: [%d, %d], Output: %.4f, Target: %.0f\n', x1, x2, y, d(i));
end

% plot the weight convergence over epochs
figure(1);
subplot(2,3,1);
plot(w_mat(1, 1:epoch));
title('w0 (bias)');
xlabel('Epoch');
ylabel('Value');
subplot(2,3,2);
plot(w_mat(2, 1:epoch));
title('w1 (x1)');
xlabel('Epoch');
ylabel('Value');
subplot(2,3,3);
plot(w_mat(3, 1:epoch));
title('w2 (x2)');
xlabel('Epoch');
ylabel('Value');
subplot(2,3,4);
plot(w_mat(4, 1:epoch));
title('w3 (x1*x2)');
xlabel('Epoch');
ylabel('Value');
subplot(2,3,5);
plot(w_mat(5, 1:epoch));
title('w4 (x1^2)');
xlabel('Epoch');
ylabel('Value');
subplot(2,3,6);
plot(w_mat(6, 1:epoch));
title('w5 (x2^2)');
xlabel('Epoch');
ylabel('Value');
