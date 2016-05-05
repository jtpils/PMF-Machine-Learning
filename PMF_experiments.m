clc; clear; close all;
start_all = tic;
data_path = '../data/u.data';
train_percent = 0.2;    % set 0.2 for sparse data
repeat = 5;


%% Read data
fprintf('Reading data...\n');
start = tic;
R = readData(data_path, 943, 1682);
I = R > 0;
rate_num = sum(sum(I));
duration = toc(start);
fprintf('Reading data finished, %.2f secs consumed.\n', duration);

%% Choosing lambda
num_train = floor(train_percent * rate_num);
randperm_100 = randperm(rate_num);
sample_train_list = randperm_100(1, 1:num_train);
I_train = sampleData(data_path, 943, 1682, sample_train_list);
I_val = I - I_train;

fprintf('\nChoosing regularization hyperparameters...\n');
start = tic;
lambda_u_set = [0.1, 1, 10, 100];
lambda_v_set = [0.1, 1, 10, 100];
cost_lambda_train = zeros(4, 4, repeat);
cost_lambda_val = zeros(4, 4, repeat);
for t = 1 : repeat
    fprintf('Cross validation turn %d\n', t);
    num_80 = floor(0.8 * num_train);
    randperm_train = randperm(num_train);
    randperm_top_80 = randperm_train(1, 1:num_80);
    sample_80_list = sample_train_list(randperm_top_80);
    I_train_80 = sampleData(data_path, 943, 1682, sample_80_list);
    I_val_20 = I_train - I_train_80;
    [cost_train, cost_val] = findLambda(R, I_train_80, I_val_20);
    cost_lambda_train(:, :, t) = cost_train;
    cost_lambda_val(:, :, t) = cost_val;
end
sum_cost_lambda_train = sum(cost_lambda_train, 3);
average_cost_lambda_train = sum_cost_lambda_train / repeat;
sum_cost_lambda_val = sum(cost_lambda_val, 3);
average_cost_lambda_val = sum_cost_lambda_val / repeat;
[r,c] = find(average_cost_lambda_val == min(average_cost_lambda_val(:)));
lambda_u = lambda_u_set(r);
lambda_v = lambda_v_set(c);
duration = toc(start);
fprintf('Choosing regularization hyperparameters finished, %.2f secs consumed.\n', duration);

%% Choosing K
fprintf('\nChoosing number of factors...\n');
start = tic;
K_set = [1 2 3 4 5];
cost_K_train = zeros(repeat, 5);
cost_K_val = zeros(repeat, 5);
for t = 1 : repeat
    fprintf('Cross validation turn %d\n', t);
    num_80 = floor(0.8 * num_train);
    randperm_train = randperm(num_train);
    randperm_top_80 = randperm_train(1, 1:num_80);
    sample_80_list = sample_train_list(randperm_top_80);
    I_train_80 = sampleData(data_path, 943, 1682, sample_80_list);
    I_val_29 = I_train - I_train_80;
    [cost_train, cost_val] = findK(R, I_train_80, I_val_20, lambda_u, lambda_v);
    cost_K_train(t, :) = cost_train;
    cost_K_val(t, :) = cost_val;
end
sum_cost_K_train = sum(cost_K_train, 1);
average_cost_K_train = sum_cost_K_train / repeat;
sum_cost_K_val = sum(cost_K_val, 1);
average_cost_K_val = sum_cost_K_val / repeat;
[min_cost, location] = min(average_cost_K_val);
K = K_set(location);
duration = toc(start);
fprintf('Choosing number of factors finished, %.2f secs consumed.\n', duration);

%% Train on total data
fprintf('\nTrain using all training data...\n');
start = tic;
usr_num = size(R, 1);
mv_num = size(R, 2);
U = random('norm', 0, 3, K, usr_num);
V = random('norm', 0, 3, K, mv_num);

cost = 10000;
ite = 0;
dcost = cost;
while dcost > 0.001
    [new_cost, U_new, V_new] = PMFCostFunction(R, I_train, U, V, lambda_u, lambda_v);
    U = U_new;
    V = V_new;
    dcost = abs(cost - new_cost);
    cost = new_cost;            
    fprintf('After %d iterations, cost becomes %f.\n', ite, cost);
    ite = ite + 1;
end
train_cost = cost;        
val_cost = evalCost(R, I_val, U, V);   
fprintf('[Validation] cost is %f.\n', val_cost);
duration = toc(start);
fprintf('Train using all training data finished, %.2f secs consumed.\n', duration);

duration_all = toc(start_all);
fprintf('Total %.2f secs consumed.\n', duration_all);

