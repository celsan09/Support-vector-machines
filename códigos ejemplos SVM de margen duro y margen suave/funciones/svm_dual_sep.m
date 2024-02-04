function [w, b, alphas] = svm_dual_sep(data, labels)
% INPUT
% data: num-by-dim matrix. num is the number of data points,
% dim is the dimension of a point
% labels: num-by-1 vector, specifying the class that each point
% belongs to.
% either be +1 or be -1
% OUTPUT
% w: dim-by-1 vector, the normal direction of hyperplane
% b: a scalar, the bias
% alphas: num-by-1 vector, dual variables

[num, ~] = size(data);
H = (data * data') .* (labels * labels');
cvx_begin
    variable alphas(num);
    maximize(sum(alphas) - alphas' * H * alphas / 2);
    subject to
        alphas >= 0;
        labels' * alphas == 0
cvx_end
sv_ind = alphas > 1e-4;
w = data' * (alphas .* labels);
b = mean(labels(sv_ind) - data(sv_ind, :) * w);
end