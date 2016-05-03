function [cost_train, cost_val] = findLambda(R, I_train, I_val)
K = 2;
lambda_u_set = [0.1, 1, 10, 100];
lambda_v_set = [0.1, 1, 10, 100];

usr_num = size(R, 1);
mv_num = size(R, 2);

cost_train = zeros(4, 4) + 100;
cost_val = zeros(4, 4) + 100;

for iu = 1:4
    for iv = 1:4
        lambda_u = lambda_u_set(iu);
        lambda_v = lambda_v_set(iv);
        fprintf('[Train] lambda_u = %f, lambda_v = %f...\n', lambda_u, lambda_v);
        U = random('norm', 0, 3, K, usr_num);
        V = random('norm', 0, 3, K, mv_num);

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
            
            fprintf('After %d iterations, cost becomes %f.\n', ite, cost);
            ite = ite + 1;
        end
        cost_train(iu, iv) = cost;
        
        val_cost = evalCost(R, I_val, U, V);   
        cost_val(iu, iv) = val_cost;
        fprintf('[Validation] lambda_u = %f, lambda_v = %f, cost is %f.\n', lambda_u, lambda_v, val_cost);
    end
end

end






