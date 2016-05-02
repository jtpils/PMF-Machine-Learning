function [cost U_new V_new] = PMFCostFunction(R, I, U, V, lambda_u, lambda_v)

K = size(U, 1);

cost = sqrt( sum(sum( I .* ((R - U' * V) .^ 2) )) / (sum(sum(I))) );

U_new = U;
V_new = V;
dU = zeros(size(U));
dV = zeros(size(V));

for i = 1 : size(U, 2)
    left = zeros(K, K);
    for j = 1 : size(V, 2)
        if I(i, j) ~= 0
            left = left + (V(:, j) * V(:, j)'); 
        end
    end
    left = eye(K) *  lambda_u + left;
    right = V * (I(i, :) .* R(i, :))';
    Ui_new = pinv(left) * right;
    U_new(:, i) = Ui_new;
    
%     s = zeros(K, 1);
%     s = I(i, :) * (V' .* repmat(V' * U(:, i), 1, K));
%     s = s';
%     dU(:, i) = -right + s + lambda_u .* U(:, i);
end

%Use new U to update V
U = U_new;

for j = 1 : size(V, 2)
    left = zeros(K, K);
    for i = 1 : size(U, 2)
        if I(i, j) ~= 0
            left = left + (U(:, i) * U(:, i)'); 
        end
    end
    left =  eye(K) * lambda_v + left;
    right = U * (I(:, j) .* R(:, j));
    Vj_new = pinv(left) * right;
    V_new(:, j) = Vj_new;
    
%     s = zeros(K, 1);
%     s = I(:, j)' * (U' .* repmat(U' * V(:, j), 1, K));
%     s = s';
%     dV(:, j) = -right + s + lambda_v .* V(:, j);
end

end