function [E, J, params] = caliJaco(measurement_data, nominal_params, param_errors)
% 串并混联机器人系统标定雅可比矩阵计算函数
%
% 修改点：
%   使用 P 点测量位姿 + 名义 Franka FK 构造 F 点“测量位姿”
%   不再使用 data_j.f_meas / data_j.theta_F_meas
%   构造得到的 R_F_meas 已是末端坐标系旋转，不需要 dRF

%% ============ 0. 开关：是否用 P 测量构造 F 测量 ============
use_P_meas_build_F_meas = true;
if isfield(nominal_params, 'use_P_meas_build_F_meas')
    use_P_meas_build_F_meas = logical(nominal_params.use_P_meas_build_F_meas);
end

%% 参数提取和初始化
num_measurements = length(measurement_data);

delta_A   = param_errors.delta_A;
delta_B   = param_errors.delta_B;
delta_l   = param_errors.delta_l;
delta_phi = param_errors.delta_phi;   % 6x1
delta_psi = param_errors.delta_psi;   % 6x1

n_deltaA     = numel(delta_A);      % 18
n_deltaB     = numel(delta_B);      % 18
n_deltal     = numel(delta_l);      % 6
n_deltaphi   = numel(delta_phi);    % 6
n_delta_psi  = numel(delta_psi);    % 6

n_local_per_measurement = 6;
n_local_total = num_measurements * n_local_per_measurement;

n_total_params = n_deltaA + n_deltaB + n_deltal + n_delta_psi + n_deltaphi + n_local_total;

%% 安装误差（用于“预测模型”）
p_PB_actual = nominal_params.delta_psi_nominal(1:3) + delta_psi(1:3);

rotvec_PB_nominal = nominal_params.delta_psi_nominal(4:6);
rotvec_PB_error   = delta_psi(4:6);

R_PB_nominal  = expSO3(rotvec_PB_nominal);
Delta_R_error = expSO3(rotvec_PB_error);
R_PB_actual   = R_PB_nominal * Delta_R_error;

T_P_B_actual = [R_PB_actual, p_PB_actual; 0, 0, 0, 1];

%% 名义安装（用于“构造测量的 F 位姿”）
% 注意：这里必须用名义，否则你构造的“测量”会依赖待估参数，产生循环依赖
p_PB_nom = nominal_params.delta_psi_nominal(1:3);
R_PB_nom = expSO3(nominal_params.delta_psi_nominal(4:6));

%% 定义参数索引范围（顺序：A, B, l, psi, phi, 局部参数）
idx_deltaA     = 1:n_deltaA;
idx_deltaB     = n_deltaA + (1:n_deltaB);
idx_deltal     = n_deltaA + n_deltaB + (1:n_deltal);
idx_delta_psi  = n_deltaA + n_deltaB + n_deltal + (1:n_delta_psi);
idx_deltaphi   = n_deltaA + n_deltaB + n_deltal + n_delta_psi + (1:n_deltaphi);

local_param_indices = cell(num_measurements, 1);
for j = 1:num_measurements
    start_idx = n_deltaA + n_deltaB + n_deltal + n_delta_psi + n_deltaphi + (j-1)*n_local_per_measurement + 1;
    local_param_indices{j} = start_idx:(start_idx + n_local_per_measurement - 1);
end

%% 初始化雅可比矩阵和残差向量
n_residuals_per_measurement = 12;  % 6 limb + 6 serial
n_total_residuals = num_measurements * n_residuals_per_measurement;

J = zeros(n_total_residuals, n_total_params);
E = zeros(n_total_residuals, 1);

%% ============ 主循环：每个测量点 ============
for j = 1:num_measurements


    data_j = measurement_data{j};

    % 1) 腿长测量
    l_meas_j = data_j.l_meas(:) + nominal_params.l0(:);

    % 2) 关节角
    q_serial_j = data_j.q_serial(:);

    % 3) 构造 F 的“测量位姿”
    if use_P_meas_build_F_meas

        % ---- (a) 读取 P 测量（通常在测量坐标系 M 下） ----
        pP_meas_M   = data_j.platform_position(:);
        eulP_meas_M = data_j.theta_P_meas(:).';   % 1x3, 按 ZYX 约定

        % ---- (b) 变换到全局 G：不需要 dRF ----
        [pP_meas_G, R_GP_meas] = transform_pose_to_global_no_dRF( ...
            pP_meas_M, eulP_meas_M, nominal_params.T_G_M);

        % ---- (c) Franka 名义 FK：得到 B->F ----
        % 你给的函数签名是 franka_forward_kinematics(q, dh_actual)
        [~, R_BF_nom, f_B_nom, ~] = franka_forward_kinematics(q_serial_j, nominal_params.mdh_nominal);

        % ---- (d) 名义安装：P->B ----
        f_P_nom  = p_PB_nom + R_PB_nom * f_B_nom;        % F 在 P 中的位置
        R_PF_nom = R_PB_nom * R_BF_nom;                  % F 相对 P 的旋转

        % ---- (e) 得到 F 的测量位姿（全局） ----
        f_meas_j = pP_meas_G + R_GP_meas * f_P_nom;
        R_F_meas = R_GP_meas * R_PF_nom;

    else
        % 回退：使用原来 F 的测量并做 transform（包含 dRF）
        f_meas_j = data_j.f_meas(:);
        theta_F_meas_j = data_j.theta_F_meas(:);
        [f_meas_j, R_F_meas] = transform_measurement_to_global( ...
            f_meas_j, theta_F_meas_j, nominal_params.T_G_M);
    end

    % 4) 取局部变量（P 的“待优化”位姿）
    local_params_j = param_errors.local_params(j, :);   % 1x6

    p_j = nominal_params.P_pose_nominal.p + local_params_j(1:3)';   % 3x1

    phi_Pj_nominal      = nominal_params.P_pose_nominal.rotvec(:);  % 3x1
    delta_phi_Pj_local  = local_params_j(4:6)';                      % 3x1
    R_Pj_nominal = expSO3(phi_Pj_nominal);
    R_Pj = R_Pj_nominal * expSO3(delta_phi_Pj_local);

    % 5) 单点残差与雅可比
    [residual_j, J_j, f_P, R_PF, f_pred, R_F_pred]  = single_measurement_jacobian( ...
        l_meas_j, f_meas_j, R_F_meas, q_serial_j, p_j, R_Pj, ...
        nominal_params, delta_A, delta_B, delta_l, delta_phi, T_P_B_actual, ...
        n_deltaA, n_deltaB, n_deltal, n_deltaphi, n_delta_psi, n_local_per_measurement, nominal_params.T_G_M);

    % 6) 填充总残差、总雅可比
    residual_start = (j-1)*n_residuals_per_measurement + 1;
    residual_end   = residual_start + n_residuals_per_measurement - 1;

    E(residual_start:residual_end) = residual_j;

    J_start_row = residual_start;
    J_end_row   = residual_end;

    J(J_start_row:J_end_row, idx_deltaA)    = J_j(:, 1:n_deltaA);
    J(J_start_row:J_end_row, idx_deltaB)    = J_j(:, n_deltaA+1:n_deltaA+n_deltaB);
    J(J_start_row:J_end_row, idx_deltal)    = J_j(:, n_deltaA+n_deltaB+1:n_deltaA+n_deltaB+n_deltal);
    J(J_start_row:J_end_row, idx_delta_psi) = J_j(:, n_deltaA+n_deltaB+n_deltal+1:n_deltaA+n_deltaB+n_deltal+n_delta_psi);
    J(J_start_row:J_end_row, idx_deltaphi)  = J_j(:, n_deltaA+n_deltaB+n_deltal+n_delta_psi+1:n_deltaA+n_deltaB+n_deltal+n_delta_psi+n_deltaphi);

    J(J_start_row:J_end_row, local_param_indices{j}) = J_j(:, n_deltaA+n_deltaB+n_deltal+n_delta_psi+n_deltaphi+1:end);

end

%% 返回参数结构
params.indices.deltaA     = idx_deltaA;
params.indices.deltaB     = idx_deltaB;
params.indices.deltal     = idx_deltal;
params.indices.deltaphi   = idx_deltaphi;
params.indices.delta_psi  = idx_delta_psi;
params.indices.local      = local_param_indices;

params.sizes.n_total      = n_total_params;
params.sizes.n_residuals  = n_total_residuals;
params.sizes.deltaphi     = n_deltaphi;

end

%% ================== helper：P 测量 -> G（不使用 dRF） ==================
function [p_G, R_GX] = transform_pose_to_global_no_dRF(p_M, eul_M_zyx, T_G_M)

    R_G_M = T_G_M(1:3,1:3);
    t_G_M = T_G_M(1:3,4);

    p_G = R_G_M * p_M(:) + t_G_M(:);

    % eul2rotm 对 'ZYX' 的输入是 [yaw pitch roll] = [z y x]
    if iscolumn(eul_M_zyx), eul_M_zyx = eul_M_zyx.'; end
    R_MX = eul2rotm(eul_M_zyx, 'ZYX');

    R_GX = R_G_M * R_MX;
end




function [residual_j, J_j,f_P,R_PF,f_pred,R_F_pred ] = single_measurement_jacobian(...
    l_meas, f_meas, R_F_meas, q_serial, ...
    p_j, R_Pj, ...  
    nominal_params, delta_A, delta_B, delta_l, delta_phi,T_P_B_actual, ...
    n_deltaA, n_deltaB, n_deltal, n_deltaphi, n_delta_psi, n_local_per_measurement,T_G_M)
% 单次测量的雅可比矩阵计算函数
% 考虑安装误差变换矩阵${}^{P}\boldsymbol{T}_{B}$的影响
% 输入参数:
%   delta_phi - 6×1几何参数误差向量 [Δa4, Δa5, Δa7, Δd3, Δd5, Δd7]
%   delta_psi - 6×1安装误差参数 [位置偏移(3), 方向偏移(3)]

% 总参数数量
n_total_params_j = n_deltaA + n_deltaB + n_deltal + n_delta_psi + n_deltaphi + n_local_per_measurement;

%% 1. 初始化输出
limb_residual = zeros(6, 1);
J_limb = zeros(6, n_total_params_j);

serial_residual = zeros(6, 1);
J_serial = zeros(6, n_total_params_j);

%% 2. 参数索引定义（顺序：A, B, l, psi, phi, 局部参数）
idx_deltaA = 1:n_deltaA;
idx_deltaB = n_deltaA + (1:n_deltaB);
idx_deltal = n_deltaA + n_deltaB + (1:n_deltal);
idx_delta_psi = n_deltaA + n_deltaB + n_deltal + (1:n_delta_psi); 
idx_deltaphi = n_deltaA + n_deltaB + n_deltal + n_delta_psi + (1:n_deltaphi);  % 现在只有6个
idx_local = n_deltaA + n_deltaB + n_deltal + n_delta_psi + n_deltaphi + (1:n_local_per_measurement);

idx_p = idx_local(1:3);
idx_theta = idx_local(4:6);

%% 3. 提取安装误差参数
R_PB = T_P_B_actual(1:3, 1:3);
p_PB = T_P_B_actual(1:3, 4);

%% 4. 并联部分计算（支链残差）
for i = 1:6
    r_A_actual = nominal_params.r_A(i,:)' + delta_A(i,:)';
    r_B_actual = nominal_params.r_B(i,:)' + delta_B(i,:)';


    l_pred_vec = R_Pj * r_A_actual + p_j - r_B_actual;
    l_pred = norm(l_pred_vec);
    
    if l_pred < 1e-10 
        u_ij = safe_unit_vector(l_pred_vec, 1e-8);
    else
        u_ij = l_pred_vec / l_pred;
    end

    limb_residual(i) = l_meas(i) - l_pred - delta_l(i);  
    
    % 并联部分雅可比计算
    idx_A_start = (i-1)*3 + 1;
    J_limb(i, idx_A_start:idx_A_start+2) = -u_ij' * R_Pj;
    
    idx_B_start = n_deltaA + (i-1)*3 + 1;
    J_limb(i, idx_B_start:idx_B_start+2) = u_ij';
    
    idx_l = n_deltaA + n_deltaB + i;
    J_limb(i, idx_l) = -1;
    
    J_limb(i, idx_p) = -u_ij';

    J_limb(i, idx_theta) =  u_ij' * R_Pj * skew(r_A_actual);


end

%% 5. 串联部分计算（末端位姿残差）
%使用6维的delta_phi调用正运动学函数
[T_serial, R_BF, f_B, J_geo, dh_actual, theta_BF] = franka_forward_kinematics_with_jacobian(q_serial, delta_phi,nominal_params);

% 考虑安装误差
f_P = p_PB + R_PB * f_B;
R_PF = R_PB * R_BF;

% 在全局坐标系中的预测末端位姿
f_pred = p_j + R_Pj * f_P;
R_F_pred = R_Pj * R_PF;


% 串联部分残差
pos_residual = f_meas - f_pred;
delta_R = R_F_pred' * R_F_meas;
orient_residual = rotation_matrix_to_rotation_vector(delta_R);

serial_residual = [pos_residual; orient_residual];

%% 6. 串联部分雅可比计算
% 6.1 位置残差雅可比
J_serial = zeros(6, n_total_params_j);
J_serial(1:3, idx_p) = -eye(3);

J_serial(1:3, idx_theta) =  R_Pj * skew(f_P);

J_serial(1:3, idx_delta_psi(1:3)) = -R_Pj;

J_serial(1:3, idx_delta_psi(4:6)) =  R_Pj * R_PB * skew(f_B);

J_geo_f = J_geo(1:3, :);  % 位置部分的几何雅可比
J_serial(1:3, idx_deltaphi) = -R_Pj * R_PB * J_geo_f;


%% 6.2 姿态残差雅可比（基于右误差模型的正确形式）
J_r_inv = inverse_right_jacobian_so3(-orient_residual);
J_geo_omega = J_geo(4:6, :);  % 3x6: mapping delta_phi -> small-rot-vector in B-frame

C = - J_r_inv * (R_F_pred') ;

% 1. 对平台姿态的偏导 (delta_phi_Pj 已在G系)
J_serial(4:6, idx_theta)= -J_r_inv * (R_PF');

% 2. 对安装姿态误差的偏导 (delta_phi_B 在P系，需R_Pj转到G系)
J_serial(4:6, idx_delta_psi(4:6)) = -J_r_inv * (R_BF');

% 3. 对串联臂几何参数的偏导 (delta_phi_F 在B系，需 R_Pj*R_PB 转到G系)
J_serial(4:6, idx_deltaphi) = C * R_Pj * R_PB *  J_geo_omega; 

% 确保位置参数雅可比为零
J_serial(4:6, idx_delta_psi(1:3)) = zeros(3,3);


%% 7. 组合结果
residual_j = [limb_residual; serial_residual];
J_j = [J_limb; J_serial];

end











function [f_global, R_GF] = transform_measurement_to_global(f_meas, theta_meas_rpy, T_G_M)
% % theta_meas_rpy: [roll, pitch, yaw] (r, p, y)
% % Returns theta_global in same ordering: [roll, pitch, yaw]
% 
% R_G_M = T_G_M(1:3,1:3);
% t_G_M = T_G_M(1:3,4);
% 
% % Convert [roll,pitch,yaw] -> [yaw,pitch,roll] for ZYX constructor
% eul_for_matlab = [ theta_meas_rpy(3), theta_meas_rpy(2), theta_meas_rpy(1) ]; % [z y x]
% 
% % rotation matrix of measured frame F in measurement coords
% R_F_meas = eul2rotm(eul_for_matlab, 'ZYX');
% 
% % transform position and rotation to global
% f_global = R_G_M * f_meas + t_G_M
% R_F_global = R_G_M * R_F_meas;
% 
% % convert back to Euler (MATLAB returns [z y x] for 'ZYX')
% eul_global_zyx = rotm2eul(R_F_global, 'ZYX'); % -> [yaw, pitch, roll]
% % reorder back to [roll, pitch, yaw] to keep same external convention
% theta_global = [ eul_global_zyx(3), eul_global_zyx(2), eul_global_zyx(1) ]


% 提取旋转矩阵和平移向量
R_G_M = T_G_M(1:3, 1:3);
t_G_M = T_G_M(1:3, 4)';  

% 原始数据（测量坐标系M下的数据）
positions_M = f_meas';
euler_angles_M = theta_meas_rpy';

% 将位置从M系变换到G系
positions_G = zeros(size(positions_M));
for i = 1:size(positions_M, 1)
    pos_M = positions_M(i, :);  % 
    pos_G = (R_G_M * pos_M' + t_G_M')';  % 
    positions_G(i, :) = pos_G;
end

dRF = [1,0,0;0,-1,0;0,0,-1];
% 将欧拉角从M系变换到G系
for i = 1:size(euler_angles_M, 1)
    R_M = eul2rotm(euler_angles_M(i,:), 'ZYX');
    R_GF = dRF * R_G_M* R_M;
end


f_global = positions_G';

end



function phi = rotation_matrix_to_rotation_vector(R)
% SO(3) logarithm map: R -> rotation vector phi
% guarantees: log(R') = -log(R)

    epsilon = 1e-12;

    cos_theta = (trace(R) - 1) / 2;
    cos_theta = max(min(cos_theta,1),-1);
    theta = acos(cos_theta);

    if theta < 1e-8

        % Near zero rotation
        phi = zeros(3,1);

    elseif abs(theta - pi) < 1e-4

        % Special handling near pi
        A = (R + eye(3)) / 2;

        % find the column with largest diagonal
        [~, idx] = max(diag(A));

        axis = zeros(3,1);
        axis(idx) = sqrt(A(idx,idx));

        j = mod(idx,3)+1;
        k = mod(idx+1,3)+1;

        axis(j) = A(j,idx)/axis(idx);
        axis(k) = A(k,idx)/axis(idx);

        axis = axis / norm(axis);

        phi = theta * axis;

    else

        % Standard formula
        phi = (theta / (2*sin(theta))) * ...
              [R(3,2)-R(2,3);
               R(1,3)-R(3,1);
               R(2,1)-R(1,2)];
    end

end


function u = safe_unit_vector(v, epsilon)
% 安全单位向量计算
v_norm = norm(v);
if v_norm < epsilon
    u = [1;0;0];  % 默认方向
else
    u = v / v_norm;
end
end


function J_r_inv = inverse_right_jacobian_so3(phi)
% More numerically stable version of J_r^{-1} for SO(3)

theta = norm(phi);
phi_skew = skew(phi);

if theta < 1e-8
    % Second-order Taylor expansion
    J_r_inv = eye(3) - 0.5 * phi_skew + (1/12) * (phi_skew^2);
else
    A = (1 + cos(theta)) / (2 * theta * sin(theta));
    J_r_inv = eye(3) ...
            - 0.5 * phi_skew ...
            + (1/theta^2 - A) * (phi_skew^2);
end

end




function R = expSO3(phi)
    % Rodrigues' formula, robust for small and moderate phi
    th = norm(phi);
    if th < 1e-12
        R = eye(3) + skew(phi); % linear approx
    else
        k = phi / th;
        K = skew(k);
        R = eye(3) + sin(th)*K + (1-cos(th))*(K*K);
    end
end

function S = skew(v)
    S = [   0   -v(3)  v(2);
          v(3)   0    -v(1);
         -v(2)  v(1)    0 ];
end

