% XOR inputs (x1, x2)
x = [0 0; 0 1; 1 0; 1 1];

% Desired output (target)
d = [0; 1; 1; 0]; % XOR truth table

% Learning rate
myu = 0.1;

% Number of neurons in the first layer
num_neurons = 2;

% Initialize weights and biases for the first layer
w1 = rand(num_neurons, size(x, 2)); % Random weights for each neuron
b1 = rand(num_neurons, 1);          % Random biases for each neuron

% Initialize weights and bias for the second layer (OR gate)
w2 = [1; 1]; % Fixed weights for OR gate
b2 = -0.5;   % Bias for OR gate (threshold)

% Sigmoid activation function
sigmoid = @(z) 1 ./ (1 + exp(-z));

% Training loop
max_epochs = 10000;
w1_history = zeros(num_neurons, size(x, 2), max_epochs); % Storage for weight history
b1_history = zeros(num_neurons, max_epochs);             % Storage for bias history
total_error_history = zeros(max_epochs, 1);              % Storage for total error history

for epoch = 1:max_epochs
    total_error = 0;
    for i = 1:size(x, 1)
        % Forward pass through the first stage (Adaline units)
        z1 = sigmoid(w1 * x(i, :)' + b1); % Outputs of the first layer

        % Second stage (OR gate)
        y_in = w2' * z1 + b2;             % Input to the OR gate
        y = y_in >= 0;                    % Output of the OR gate

        % Calculate error
        error = d(i) - y;
        total_error = total_error + error^2;

        % Update weights and biases using LMS
        if error ~= 0
            for j = 1:num_neurons
                % Update weights for first layer
                w1(j, :) = w1(j, :) + myu * error * z1(j) * x(i, :);
                b1(j) = b1(j) + myu * error * z1(j);
            end
        end
    end

    % Store weights and biases for plotting
    w1_history(:, :, epoch) = w1;
    b1_history(:, epoch) = b1;
    total_error_history(epoch) = total_error; % Store total error

    % Print the total error for each epoch
    fprintf('Epoch: %d, Total Error: %.4f\n', epoch, total_error);
    
    % Check for convergence
    if total_error < 0.01 % Use a threshold instead of exact zero
        break;
    end
end

% Check outputs after training
fprintf('\nChecking outputs after training:\n');
for i = 1:size(x, 1)
    z1 = sigmoid(w1 * x(i, :)' + b1); % Outputs of the first layer
    y_in = w2' * z1 + b2;             % Input to the OR gate
    y = y_in >= 0;                    % Output of the OR gate
    fprintf('Input: [%d, %d], Output: %d, Target: %d\n', x(i, 1), x(i, 2), y, d(i));
end


% Plot the weight, bias evolution, and total error in one figure
figure;

% Create subplots for weights and biases
for j = 1:num_neurons
    subplot(num_neurons + 1, 1, j);
    plot(squeeze(w1_history(j, 1, 1:epoch)), 'DisplayName', sprintf('w1(%d, x1)', j));
    hold on;
    plot(squeeze(w1_history(j, 2, 1:epoch)), 'DisplayName', sprintf('w1(%d, x2)', j));
    plot(squeeze(b1_history(j, 1:epoch)), 'DisplayName', sprintf('b1(%d)', j));
    title(sprintf('Neuron %d Weights and Bias', j));
    xlabel('Epoch');
    ylabel('Value');
    legend;
    hold off;
end

% Create a subplot for total error
subplot(num_neurons + 1, 1, num_neurons + 1);
plot(1:epoch, total_error_history(1:epoch), 'k', 'LineWidth', 1.5);
title('Total Error Over Epochs');
xlabel('Epoch');
ylabel('Total Error');
grid on;

