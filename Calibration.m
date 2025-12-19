clear; clc
fprintf('=== 串并混联系统标定  LM 变量投影  连续性软约束  全局先验正则 ===\n');

%% 1 数据
fprintf('1 数据读取...\n');
load('measured_data_withP.mat');
measurement_data = filtered_measurement_data;
m = length(measurement_data);
fprintf('共 %d 个测量点\n', m);

%% 2 初始化参数
fprintf('2 初始化标定参数...\n');
[param_errors_initial, nominal_params]  = initialize_calibration_parameters(measurement_data);
x0 = params_to_vector(param_errors_initial, nominal_params, m);
fprintf('优化参数数量: %d\n', length(x0));

%% 3 约束与权重
constraints = define_calibration_constraints();

weights = struct();

% 系统残差权重
weights.w_l  = 1e10;
weights.w_Fp = 1e8;
weights.w_Fr = 1e8;

% 局部变量软约束权重
weights.w_anchor   = 1e4;
weights.w_cont_p   = 0;
weights.w_cont_r   = 0;
weights.w_pz_bound = 1e8;
weights.w_ang_bound= 1e6;

% 先验均值为 0 不启用
weights.prior = struct();
weights.prior.enable = true;
weights.prior.w = 0;

weights.prior.sigma_A = 5.0e-2;
weights.prior.sigma_B = 5.0e-2;
weights.prior.sigma_l = 1.0e-2;

weights.prior.sigma_psi_p = 1.0e-2;
weights.prior.sigma_psi_r = deg2rad(2.0);

weights.prior.sigma_phi = 5.0e-4;

%% 4 初始残差
fprintf('3 初始残差评估...\n');
[E0, J0] = compute_weighted_residuals_and_jacobian(x0, measurement_data, nominal_params, weights, constraints);
fprintf('初始 cost = %.6e, ||E||=%.6e, J size=%dx%d\n', 0.5*(E0.'*E0), norm(E0), size(J0,1), size(J0,2));

%% 5 LM
fprintf('4 开始 LM 优化...\n');
lm_params = setup_lm_parameters(constraints);

[x_opt, optinfo] = levenberg_marquardt_optimization( ...
    @(x) compute_weighted_residuals_and_jacobian(x, measurement_data, nominal_params, weights, constraints), ...
    x0, lm_params);

fprintf('=== 完成  final cost=%.6e  iters=%d ===\n', optinfo.final_cost, optinfo.iterations);


%% 6 收敛性分析
fprintf('6 参数收敛性分析...\n');
plot_parameter_convergence_analysis(optinfo, measurement_data, nominal_params)
fprintf('\n=== 收敛性分析完成 ===\n');


%% ===================== functions =====================

function constraints = define_calibration_constraints()

    constraints = struct();
    constraints.enable = true;

    constraints.pz_min = 0.5;
    constraints.pz_max = 1.0;

    constraints.angle_max = deg2rad(30);

    constraints.max_position_change = 0.005;
    constraints.max_angle_change    = deg2rad(2);

    constraints.function_tolerance = 1e-8;
    constraints.step_tolerance     = 1e-8;
    constraints.gradient_tolerance = 1e-8;

    constraints.sampling_time = 0.1;
end

function lm_params = setup_lm_parameters(constraints)
    lm_params = struct();

    lm_params.max_iterations = 20;
    lm_params.max_inner_iterations = 60;

    lm_params.initial_lambda = 1e-3;
    lm_params.lambda_factor  = 5;
    lm_params.rho_threshold  = 0.1;

    lm_params.function_tolerance = constraints.function_tolerance;
    lm_params.step_tolerance     = constraints.step_tolerance;
    lm_params.gradient_tolerance = constraints.gradient_tolerance;

    lm_params.cost_tol_abs = 1e-8;
    lm_params.cost_tol_rel = 1e-6;

end

function [E, J] = compute_weighted_residuals_and_jacobian(x, measurement_data, nominal_params, weights, constraints)
% 返回加权残差 E 与加权雅可比 J
% E = [系统残差; 局部软约束残差; 全局先验残差]
% J = [系统雅可比; 局部软约束雅可比; 全局先验雅可比]

    m = length(measurement_data);

    param_errors = vector_to_params(x, nominal_params, m);

    % ---- 系统部分 ----
    [E_sys, J_sys, ~] = caliJaco(measurement_data, nominal_params, param_errors);

    w_row = build_row_weights_system(m, weights);
    E_sys_w = w_row .* E_sys;
    J_sys_w = bsxfun(@times, J_sys, w_row);

    % ---- 局部变量软约束 ----
    [E_reg_w, J_reg_w] = build_local_pose_soft_constraints(param_errors, nominal_params, constraints, weights);

    % ---- 全局参数先验正则 ----
    [E_prior_w, J_prior_w] = build_global_prior_soft_constraints(param_errors, m, weights);

    % ---- 拼接 ----
    E = [E_sys_w; E_reg_w; E_prior_w];
    J = [sparse(J_sys_w); J_reg_w; J_prior_w];
end

function w_row = build_row_weights_system(m, weights)
% caliJaco 每个测量点 12 行
    w_block = [ sqrt(weights.w_l)  * ones(6,1);
                sqrt(weights.w_Fp) * ones(3,1);
                sqrt(weights.w_Fr) * ones(3,1) ];
    w_row = repmat(w_block, m, 1);
end

function [Ew, Jw] = build_global_prior_soft_constraints(param_errors, m, weights)
% 对前 54 个全局参数加高斯先验
% prior mean 0
% prior residual
%   e = p / sigma
% Jacobian
%   euclidean params use I/sigma
%   rotation-vector param delta_psi(4:6) uses Jr^{-1}/sigma to match right update

    n_global = 54;
    n_params = n_global + 6*m;

    if ~isfield(weights,'prior') || ~isfield(weights.prior,'enable') || ~weights.prior.enable
        Ew = zeros(0,1);
        Jw = spalloc(0, n_params, 0);
        return;
    end

    wp = sqrt(weights.prior.w);

    % sigmas
    sA = weights.prior.sigma_A;
    sB = weights.prior.sigma_B;
    sl = weights.prior.sigma_l;
    sp = weights.prior.sigma_psi_p;
    sr = weights.prior.sigma_psi_r;
    sf = weights.prior.sigma_phi;

    sigma = [ sA*ones(18,1);
              sB*ones(18,1);
              sl*ones(6,1);
              sp*ones(3,1);
              sr*ones(3,1);
              sf*ones(6,1) ];

    if numel(sigma) ~= 54
        error('sigma length mismatch');
    end

    % current global parameters
    xg = zeros(54,1);
    xg(1:18)   = reshape(param_errors.delta_A.', [], 1);
    xg(19:36)  = reshape(param_errors.delta_B.', [], 1);
    xg(37:42)  = param_errors.delta_l(:);
    xg(43:48)  = param_errors.delta_psi(:);
    xg(49:54)  = param_errors.delta_phi(:);

    Ew = wp * (xg ./ sigma);

    % Jacobian
    invsig = wp * (1 ./ sigma);

    % start with diagonal
    Jw = spalloc(54, n_params, 60);
    Jw = Jw + sparse(1:54, 1:54, invsig, 54, n_params);

    % fix rotation-vector prior Jacobian for delta_psi orientation
    % global indices 46:48 correspond to delta_psi(4:6)
    phi_psi = param_errors.delta_psi(4:6);
    Jr_inv = inverse_right_jacobian_so3(phi_psi);

    % remove diagonal entries at 46:48 then insert Jr_inv/sigma
    Jw(46:48,46:48) = sparse(3,3);
    Jw(46:48,46:48) = wp * (1/sr) * sparse(Jr_inv);
end

function [Ew, Jw] = build_local_pose_soft_constraints(param_errors, nominal_params, constraints, weights)
% 1) anchor：铆钉第一个测量点（现在是 6 个：x,y,z + 3个旋转向量分量）
% 2) continuity：相邻测量点连续性
% 3) pz bounds：绝对高度上下界
% 4) angle bound：姿态幅值上界

    m = size(param_errors.local_params, 1);
    n_global = 54;
    n_params = n_global + 6*m;

    dp = param_errors.local_params(:,1:3).';
    dr = param_errors.local_params(:,4:6).';

    [p0, phi0] = get_nominal_pose(nominal_params);
    expSO3(phi0); 

    sig_p = max(constraints.max_position_change, 1e-12);
    sig_r = max(constraints.max_angle_change,    1e-12);

    n_anchor = 6;                    
    n_cont_p = 3*max(m-1,0);
    n_cont_r = 3*max(m-1,0);
    n_pz     = 2*m;
    n_ang    = m;

    n_reg = n_anchor + n_cont_p + n_cont_r + n_pz + n_ang;

    Ew = zeros(n_reg,1);
    Jw = spalloc(n_reg, n_params, max(220, 90*m));

    row = 0;

    % ---------------- anchor（第一个测量点）----------------
    wa = sqrt(weights.w_anchor);
    idx_dp1 = n_global + (0)*6 + (1:3);
    idx_dr1 = n_global + (0)*6 + (4:6);

    % anchor x
    row = row + 1;
    Ew(row) = wa * (dp(1,1) / sig_p);
    Jw(row, idx_dp1(1)) = wa * (1 / sig_p);

    % anchor y
    row = row + 1;
    Ew(row) = wa * (dp(2,1) / sig_p);
    Jw(row, idx_dp1(2)) = wa * (1 / sig_p);

    % ========= anchor z =========
    row = row + 1;
    Ew(row) = wa * (dp(3,1) / sig_p);
    Jw(row, idx_dp1(3)) = wa * (1 / sig_p);

    % anchor rotation-vector
    Jr1_inv = inverse_right_jacobian_so3(dr(:,1));
    for k = 1:3
        row = row + 1;
        Ew(row) = wa * (dr(k,1) / sig_r);
        Jw(row, idx_dr1) = wa * (1 / sig_r) * Jr1_inv(k,:);
    end

    % ---------------- continuity position ----------------
    wc_p = sqrt(weights.w_cont_p);
    if m >= 2
        for j = 1:(m-1)
            idx_dpj   = n_global + (j-1)*6 + (1:3);
            idx_dpj1  = n_global + (j)*6   + (1:3);
            dstep = (dp(:,j+1) - dp(:,j)) / sig_p;
            for k = 1:3
                row = row + 1;
                Ew(row) = wc_p * dstep(k);
                Jw(row, idx_dpj(k))  = wc_p * (-1 / sig_p);
                Jw(row, idx_dpj1(k)) = wc_p * ( 1 / sig_p);
            end
        end
    end

    % ---------------- continuity rotation ----------------
    wc_r = sqrt(weights.w_cont_r);
    if m >= 2
        for j = 1:(m-1)
            idx_drj  = n_global + (j-1)*6 + (4:6);
            idx_drj1 = n_global + (j)*6   + (4:6);

            Rj  = expSO3(dr(:,j));
            Rj1 = expSO3(dr(:,j+1));

            e = logSO3(Rj.' * Rj1);
            Jr_inv = inverse_right_jacobian_so3(e);
            Jl_inv = inverse_right_jacobian_so3(-e);

            for k = 1:3
                row = row + 1;
                Ew(row) = wc_r * (e(k)/sig_r);
                Jw(row, idx_drj )  = wc_r * (1/sig_r) * (-Jl_inv(k,:));
                Jw(row, idx_drj1)  = wc_r * (1/sig_r) * ( Jr_inv(k,:));
            end
        end
    end

    % ---------------- pz bounds ----------------
    wz = sqrt(weights.w_pz_bound);
    for j = 1:m
        idx_dpj = n_global + (j-1)*6 + (1:3);
        pz = p0(3) + dp(3,j);

        row = row + 1;
        low = constraints.pz_min - pz;
        if low > 0
            Ew(row) = wz * low;
            Jw(row, idx_dpj(3)) = wz * (-1);
        end

        row = row + 1;
        high = pz - constraints.pz_max;
        if high > 0
            Ew(row) = wz * high;
            Jw(row, idx_dpj(3)) = wz * ( 1);
        end
    end

    % ---------------- angle bound ----------------
    wang = sqrt(weights.w_ang_bound);
    for j = 1:m
        idx_drj = n_global + (j-1)*6 + (4:6);
        ang = norm(dr(:,j));
        row = row + 1;
        if ang > constraints.angle_max && ang > 1e-12
            Ew(row) = wang * (ang - constraints.angle_max);
            Jr_inv = inverse_right_jacobian_so3(dr(:,j));
            grad = (dr(:,j).' / ang) * Jr_inv;
            Jw(row, idx_drj) = wang * grad;
        end
    end

    if row ~= n_reg
        error('internal residual size mismatch: row=%d, expected=%d', row, n_reg);
    end
end

function [p0, phi0] = get_nominal_pose(nominal_params)
    Pn = nominal_params.P_pose_nominal;
    if isstruct(Pn)
        p0 = Pn.p(:);
        phi0 = Pn.rotvec(:);
        return;
    end
    if isnumeric(Pn) && numel(Pn)==6
        Pn = Pn(:);
        p0 = Pn(1:3);
        phi0 = Pn(4:6);
        return;
    end
    error('nominal_params.P_pose_nominal format not recognized');
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
    ct = (trace(R) - 1)/2;
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
    S = [ 0    -v(3)  v(2);
          v(3)  0    -v(1);
         -v(2)  v(1)  0 ];
end

function v = vee(S)
    v = [S(3,2); S(1,3); S(2,1)];
end

function J_r_inv = inverse_right_jacobian_so3(phi)
    phi = phi(:);
    theta = norm(phi);
    Phi = skew(phi);
    if theta < 1e-8
        J_r_inv = eye(3) - 0.5*Phi + (1/12)*(Phi*Phi);
    else
        A = (1 + cos(theta)) / (2*theta*sin(theta));
        J_r_inv = eye(3) - 0.5*Phi + (1/theta^2 - A)*(Phi*Phi);
    end
end

function param_errors = vector_to_params(x, nominal_params, m) %#ok<INUSD>
% 参数顺序固定
% delta_A 18
% delta_B 18
% delta_l 6
% delta_psi 6
% delta_phi 6
% local_params 6*m

    x = x(:);

    n_deltaA = 18;
    n_deltaB = 18;
    n_deltal = 6;
    n_delta_psi = 6;
    n_delta_phi = 6;

    idx = 1;

    param_errors = struct();

    param_errors.delta_A = reshape(x(idx:idx+n_deltaA-1), 3, 6).';
    idx = idx + n_deltaA;

    param_errors.delta_B = reshape(x(idx:idx+n_deltaB-1), 3, 6).';
    idx = idx + n_deltaB;

    param_errors.delta_l = x(idx:idx+n_deltal-1);
    idx = idx + n_deltal;

    param_errors.delta_psi = x(idx:idx+n_delta_psi-1);
    idx = idx + n_delta_psi;

    param_errors.delta_phi = x(idx:idx+n_delta_phi-1);
    idx = idx + n_delta_phi;

    param_errors.local_params = reshape(x(idx:end), 6, m).';
end

function x = params_to_vector(param_errors, nominal_params, m) %#ok<INUSD>

    n_deltaA = 18;
    n_deltaB = 18;
    n_deltal = 6;
    n_delta_psi = 6;
    n_delta_phi = 6;

    n_global = n_deltaA + n_deltaB + n_deltal + n_delta_psi + n_delta_phi;
    n_local  = 6*m;

    x = zeros(n_global + n_local, 1);

    idx = 1;
    x(idx:idx+n_deltaA-1) = reshape(param_errors.delta_A.', [], 1);
    idx = idx + n_deltaA;

    x(idx:idx+n_deltaB-1) = reshape(param_errors.delta_B.', [], 1);
    idx = idx + n_deltaB;

    x(idx:idx+n_deltal-1) = param_errors.delta_l(:);
    idx = idx + n_deltal;

    x(idx:idx+n_delta_psi-1) = param_errors.delta_psi(:);
    idx = idx + n_delta_psi;

    x(idx:idx+n_delta_phi-1) = param_errors.delta_phi(:);
    idx = idx + n_delta_phi;

    x(idx:end) = reshape(param_errors.local_params.', [], 1);
end

