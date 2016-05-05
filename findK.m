function [cost_train, cost_val] = findK(R, I_train, I_val, lambda_u, lambda_v)

usr_num = size(R, 1);
mv_num = size(R, 2);

K_set = [1 2 3 4 5];
cost_train = zeros(1, 5) + 100;
cost_val = zeros(1, 5) + 100;

for k = 1:5
%     fprintf('[Train] K = %d...\n', k);
    U = random('norm', 0, 3, k, usr_num);
    V = random('norm', 0, 3, k, mv_num);

    cost = 10000;
    ite = 0;
    dcost = cost;
    while dcost > 0.001
        [new_cost U_new V_new] = PMFCostFunction(R, I_train, U, V, lambda_u, lambda_v);
        U = U_new;
        V = V_new;
        dcost = abs(cost - new_cost);
        cost = new_cost;
%             U = U - step .* dU;
%             V = V - step .* dV;
            
%         fprintf('After %d iterations, cost becomes %f.\n', ite, cost);
        ite = ite + 1;
    end
    fprintf('[Train] K = %d, cost is %f.\n', k, cost);
    cost_train(k) = cost;
        
    val_cost = evalCost(R, I_val, U, V);   
    cost_val(k) = val_cost;
    fprintf('[Validation] K = %d, cost is %f.\n', k, val_cost);
end

end