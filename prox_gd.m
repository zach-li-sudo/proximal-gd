function [x, iter, errs] = prox_gd(x0, f, g, f_grad, soft_thresh, Lf, tol, max_iter)
    iter = 1;
    %xp_gd = x0 - (1/Lf)*f_grad(x0);
    %err = norm(xp_gd);
    err = 1e6;
    errs = zeros(max_iter, 1);
    
    while err >= tol && iter < max_iter
        xp_gd = x0 - (1/Lf)*f_grad(x0);
        x = soft_thresh(xp_gd,1/Lf);
        err = norm(x-x0);
        errs(iter, 1) = err;
        iter = iter + 1;
        x0 = x;
    end
    
    errs = errs(1:iter);
end