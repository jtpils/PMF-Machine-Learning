clc; clear;
data_path = '../data/u.data';

%% Read data
fprintf('Reading data...\n');
R = readData(data_path, 943, 1682);
I = R > 0;
rate_num = sum(sum(I));

%% Choosing lambda
%select 80 percent
num_80 = floor(0.8 * rate_num);
randperm_100 = randperm(rate_num);
sample_80_list = randperm_100(1, 1:num_80);
I_80 = sampleData(data_path, 943, 1682, sample_80_list);
I_20 = I - I_80;

fprintf('Choosing regularization hyperparameters...\n');
lambda_u_set = [0.1, 1, 10, 100];
lambda_v_set = [0.1, 1, 10, 100];
repeat = 1;
cost = zeros(4, 4, repeat);
for t = 1 : repeat
    fprintf('Cross validation turn %d\n', t);
    num_64 = floor(0.64 * rate_num);
    randperm_80 = randperm(num_80);
    randperm_top_64 = randperm_80(1, 1:num_64);
    sample_64_list = sample_80_list(randperm_top_64);
    I_train_64 = sampleData(data_path, 943, 1682, sample_64_list);
    I_val_16 = I_80 - I_train_64;
    cost_mat = findLambda(R, I_train_64, I_val_16);
    cost(:, :, t) = cost_mat;
end
sum_cost = sum(cost, 3);
[r,c] = find(sum_cost == min(sum_cost(:)))
lambda_u = lambda_u_set(r);
lambda_v = lambda_v_set(c);

%% Choosing K
fprintf('Choosing K hyperparameters...\n');
K_set = [1 2 3 4 5];
cost = zeros(repeat, 5) + 100;
for t = 1 : repeat
    fprintf('Cross validation turn %d\n', t);
    num_64 = floor(0.64 * rate_num);
    randperm_80 = randperm(num_80);
    randperm_top_64 = randperm_80(1, 1:num_64);
    sample_64_list = sample_80_list(randperm_top_64);
    I_train_64 = sampleData(data_path, 943, 1682, sample_64_list);
    I_val_16 = I_80 - I_train_64;
    cost_mat = findK(R, I_train_64, I_val_16, lambda_u, lambda_v);
    cost(t, :) = cost_mat;
end
sum_cost = sum(cost, 1);
[min_cost location] = min(sum_cost);
K = K_set(location);

%% Train use all data
fprintf('Train using all data...\n');
usr_num = size(R, 1);
mv_num = size(R, 2);
U = random('norm', 0, 3, K, usr_num);
V = random('norm', 0, 3, K, mv_num);

cost = 10000;
ite = 0;
dcost = cost;
while dcost > 0.001
    [new_cost U_new V_new] = PMFCostFunction(R, I_80, U, V, lambda_u, lambda_v);
    U = U_new;
    V = V_new;
    dcost = abs(cost - new_cost);
    cost = new_cost;            
    fprintf('After %d iterations, cost becomes %f.\n', ite, cost);
    ite = ite + 1;
end
        
val_cost = evalCost(R, I_20, U, V);   
fprintf('[Validation] cost is %f.\n', val_cost);

