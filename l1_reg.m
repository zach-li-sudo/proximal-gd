%% Example 10.38 in the book, First-Order Methods in Opt, by Beck
% % l1-regularized least square
% % Proximal Gradient Descent with fixed step-size
% % author: Zach Li
clear all; clc;

%% l1-l2 minimization
% % min 1/2 ||Ax-b||_2^2 + \lambda*||x||_1
% % prox_t(x) = \arg\min_{z} 1/(2t)*||x-z||^2_2 + lambda*||z||_1
% %           = SoftThreshold_{\lambda}(x)

m = 20;
n = 100;

A = randn(m,n);
Lf = max(eig(A'*A));
b = randn(m,1);
lambda = 10;


x0 = randn(n,1);
tol = 1e-12;
max_iter = 2e4;


f = @(x) 0.5*norm(A*x-b)^2;
g = @(x) lambda*norm(x, 1);
f_grad = @(x) A'*(A*x-b);

soft_thresh = @(x,t) sign(x) .* max(abs(x) - lambda*t, 0);

[x, iter, errs] = prox_gd(x0, f, g, f_grad, soft_thresh, Lf, tol, max_iter);

figure(1)
semilogy(errs);
title('Proximal Gradient Descent for $l_1$ regularization', 'Interpreter', 'latex');
grid on
xlabel('iterations');
ylabel('$\Vert(prox(x_k)-x_k)\Vert_2$', 'Interpreter', 'latex');
saveas(gcf, 'pgd.png');
saveas(gcf, 'pgd.pdf');
