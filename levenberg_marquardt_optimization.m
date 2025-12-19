function [x_opt, optinfo] = levenberg_marquardt_optimization(residual_func, x0, params)

    x = x0(:);
    n_params = length(x);

    n_global = 54;
    if n_params < n_global
        error('parameter length < 54');
    end

    n_local = n_params - n_global;
    if mod(n_local,6) ~= 0
        error('n_local not multiple of 6');
    end
    m = n_local / 6;

    rot_groups = build_rotation_groups(m);

    % LM parameters
    lambda    = params.initial_lambda;
    max_iter  = params.max_iterations;
    max_inner = params.max_inner_iterations;

    % tolerances
    grad_tol = params.function_tolerance;
    step_tol = params.step_tolerance;

    % history
    optinfo.parameter_history = x.';
    optinfo.cost_history = [];
    optinfo.lambda_history = [];
    optinfo.rho_history = [];
    optinfo.residual_norm_history = [];
    optinfo.gradient_norm_history = [];
    optinfo.iterations = 0;

    % initial evaluation
    [E, J] = residual_func(x);
    cost  = 0.5*(E.'*E);
    g     = J.'*E;
    gnorm = norm(g);

    fprintf('iter %2d cost=%.6e ||E||=%.3e ||g||=%.3e lambda=%.2e\n', ...
        0, cost, norm(E), gnorm, lambda);

    optinfo.cost_history(1) = cost;
    optinfo.lambda_history(1) = lambda;
    optinfo.rho_history(1) = 0;
    optinfo.residual_norm_history(1) = norm(E);
    optinfo.gradient_norm_history(1) = gnorm;

    % ---------------- LM main loop ----------------
    for iter = 1:max_iter

        success = false;
        cost_old = cost;

        for inner = 1:max_inner

            % ---- full LM step ----
            dx = solve_step_full(E, J, lambda);

            % ---- right perturbation update ----
            x_trial = apply_right_update(x, dx, rot_groups);
            dx_eff  = effective_increment(x, x_trial, rot_groups);

            % evaluate new point
            [E_new, J_new] = residual_func(x_trial);
            cost_new = 0.5*(E_new.'*E_new);

            % actual and predicted reduction
            act_red = cost - cost_new;

            Jdx = J * dx_eff;
            pred_red = -E.'*Jdx - 0.5*(Jdx.'*Jdx);

            if abs(pred_red) < 1e-15
                rho = 0;
            else
                rho = act_red / pred_red;
            end

            accept = (rho > params.rho_threshold) && isfinite(cost_new) && (cost_new < cost);

            if accept
                % accept step
                x = x_trial;
                E = E_new;
                J = J_new;
                cost = cost_new;

                g = J.'*E;
                gnorm = norm(g);

                lambda = max(lambda / params.lambda_factor, 1e-12);
                success = true;

                optinfo.parameter_history(end+1,:) = x.';
                optinfo.cost_history(end+1) = cost;
                optinfo.lambda_history(end+1) = lambda;
                optinfo.rho_history(end+1) = rho;
                optinfo.residual_norm_history(end+1) = norm(E);
                optinfo.gradient_norm_history(end+1) = gnorm;

                fprintf('iter %2d cost=%.6e lambda=%.2e rho=%.3f ||dx||=%.2e\n', ...
                    iter, cost, lambda, rho, norm(dx_eff));

                break;
            else
                % reject step
                lambda = lambda * params.lambda_factor;
                fprintf('  inner %2d reject lambda=%.2e rho=%.3f cost_new=%.3e\n', ...
                    inner, lambda, rho, cost_new);

                if lambda > 1e16
                    fprintf('lambda too large stop\n');
                    break;
                end
            end
        end

        if ~success
            fprintf('iter %2d no acceptable step stop\n', iter);
            break;
        end

        % ---------- stopping criteria ----------

        % (A) gradient
        if gnorm < grad_tol
            fprintf('converged: ||grad|| < %.1e\n', grad_tol);
            break;
        end

        % (B) step length
        if norm(dx_eff) < step_tol * max(norm(x),1)
            fprintf('converged: ||dx|| < %.1e\n', step_tol);
            break;
        end

        % (C) cost absolute change
        if abs(cost_old - cost) < params.cost_tol_abs
            fprintf('converged: |Δcost| < %.1e\n', params.cost_tol_abs);
            break;
        end

        % (D) cost relative change
        if abs(cost_old - cost) / max(cost_old,1) < params.cost_tol_rel
            fprintf('converged: relative Δcost < %.1e\n', params.cost_tol_rel);
            break;
        end

        optinfo.iterations = iter;
    end

    x_opt = x;
    optinfo.final_cost = cost;
    optinfo.final_residual_norm = norm(E);
    optinfo.final_gradient_norm = gnorm;

    fprintf('LM done final cost=%.6e iters=%d ||E||=%.3e ||g||=%.3e\n', ...
        cost, optinfo.iterations, norm(E), gnorm);
end

%% -------- functions --------

function dx = solve_step_full(E, J, lambda)
% Standard LM step 

    JTJ = J.' * J;
    g   = J.' * E;

    % Levenberg damping
    D = diag(diag(JTJ));
    d = diag(D);
    d(d == 0) = 1e-12;
    D = diag(d);

    A = JTJ + lambda * D;
    dx = - A \ g;
end

function rot_groups = build_rotation_groups(m)
    n_global = 54;
    rot_groups = cell(m+1,1);
    rot_groups{1} = 46:48;   % delta_psi rotation
    for j = 1:m
        base = n_global + (j-1)*6;
        rot_groups{j+1} = (base+4):(base+6);
    end
end

function x_new = apply_right_update(x, dx, rot_groups)
    x_new = x + dx;
    for k = 1:length(rot_groups)
        idx = rot_groups{k};
        x_new(idx) = logSO3( expSO3(x(idx)) * expSO3(dx(idx)) );
    end
end

function dx_eff = effective_increment(x, x_new, rot_groups)
    dx_eff = x_new - x;
    for k = 1:length(rot_groups)
        idx = rot_groups{k};
        dx_eff(idx) = logSO3( expSO3(x(idx)).' * expSO3(x_new(idx)) );
    end
end

function R = expSO3(phi)
    phi = phi(:);
    th = norm(phi);
    if th < 1e-12
        R = eye(3) + skew(phi);
        return;
    end
    a = phi / th;
    A = skew(a);
    R = eye(3) + sin(th)*A + (1-cos(th))*(A*A);
end

function phi = logSO3(R)
    ct = (trace(R)-1)/2;
    ct = max(min(ct,1),-1);
    th = acos(ct);
    if th < 1e-12
        phi = 0.5*vee(R - R.');
        return;
    end
    phi = (th/(2*sin(th))) * vee(R - R.');
end

function S = skew(v)
    v = v(:);
    S = [0 -v(3) v(2);
         v(3) 0 -v(1);
        -v(2) v(1) 0];
end

function v = vee(S)
    v = [S(3,2); S(1,3); S(2,1)];
end
