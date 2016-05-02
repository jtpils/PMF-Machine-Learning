function cost = evalCost(R, I, U, V)

cost = sqrt( sum(sum( I .* ((R - U' * V) .^ 2) )) / (sum(sum(I))) );

end