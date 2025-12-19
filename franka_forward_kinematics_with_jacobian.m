function [T0e, R_PF, f_P, J_geo, dh_actual, theta_PF] = franka_forward_kinematics_with_jacobian(q, delta_phi,nominal_params)
% Franka机械臂正运动学（基于MDH模型），同时计算几何参数雅可比
% 输入:
%   q - 7×1关节角度向量
%   delta_phi - 6×1几何参数误差向量 [Δa4, Δa5, Δa7, Δd3, Δd5, Δd7]
% 输出:
%   T0e - 4×4齐次变换矩阵
%   R_PF - 3×3旋转矩阵
%   f_P - 3×1位置向量
%   J_geo - 6×6几何雅可比矩阵
%   dh_actual - 实际MDH参数
%   theta_PF - 末端欧拉角

mdh_nominal=nominal_params.mdh_nominal;

% 应用参数误差 - 修改为6个参数
dh_actual = mdh_nominal;
% a参数误差
dh_actual(4, 1) = mdh_nominal(4, 1) + delta_phi(1);  % a4
dh_actual(5, 1) = mdh_nominal(5, 1) + delta_phi(2);  % a5  
dh_actual(7, 1) = mdh_nominal(7, 1) + delta_phi(3);  % a7
% d参数误差（移除Δd1，只保留Δd3, Δd5, Δd7）
dh_actual(3, 2) = mdh_nominal(3, 2) + delta_phi(4);  % d3
dh_actual(5, 2) = mdh_nominal(5, 2) + delta_phi(5);  % d5
dh_actual(7, 2) = mdh_nominal(7, 2) + delta_phi(6);  % d7（末端法兰偏移）


% 计算正运动学（基于MDH模型）
[T0e, R_PF, f_P, theta_PF] = franka_forward_kinematics(q, dh_actual);

% 计算几何参数雅可比（基于MDH模型）
J_geo = compute_geometric_jacobian_mdh(q, mdh_nominal, delta_phi);
end

function J_geo = compute_geometric_jacobian_mdh(q, mdh_nominal, delta_phi)
% 基于MDH模型的解析法计算机械臂末端位姿对几何参数的雅可比矩阵
% 输入:
%   q - 7×1关节角度向量
%   mdh_nominal - 7×4标称MDH参数矩阵 [a, d, alpha, theta_offset]
%   delta_phi - 6×1几何参数误差向量 [Δa4, Δa5, Δa7, Δd3, Δd5, Δd7]
% 输出:
%   J_geo - 6×6几何雅可比矩阵，前3行是位置导数，后3行是姿态导数

%% 构建当前的实际MDH参数
mdh_actual = mdh_nominal;
% 应用参数误差（6个参数）
mdh_actual(4, 1) = mdh_nominal(4, 1) + delta_phi(1);  % a4
mdh_actual(5, 1) = mdh_nominal(5, 1) + delta_phi(2);  % a5  
mdh_actual(7, 1) = mdh_nominal(7, 1) + delta_phi(3);  % a7
mdh_actual(3, 2) = mdh_nominal(3, 2) + delta_phi(4);  % d3
mdh_actual(5, 2) = mdh_nominal(5, 2) + delta_phi(5);  % d5
mdh_actual(7, 2) = mdh_nominal(7, 2) + delta_phi(6);  % d7（末端法兰偏移）

% 提取MDH参数
a = mdh_actual(:, 1);
d = mdh_actual(:, 2);
alpha = mdh_actual(:, 3);
theta_offset = mdh_actual(:, 4);
theta = q(:) + theta_offset;  % 实际关节角度

%% 预计算所有MDH变换矩阵
T_i = cell(7, 1);    % 每个连杆的MDH变换矩阵 T_{i-1,i}
T_0i = cell(8, 1);   % 从基座到第i个坐标系的变换 T_{0,i}
T_iN = cell(8, 1);   % 从第i个坐标系到末端的变换 T_{i,N}

% 基坐标系
T_0i{1} = eye(4);

% 前向计算 T_0i（MDH模型）- 7个连杆
for i = 1:7
    T_i{i} = mdh_transform(a(i), d(i), alpha(i), theta(i));
    T_0i{i+1} = T_0i{i} * T_i{i};
end

% 末端变换矩阵（已包含末端法兰，通过d7参数）
T_0N = T_0i{8};
R_0N = T_0N(1:3, 1:3);

% 反向计算 T_iN
T_iN{8} = eye(4);  % 末端坐标系
for i = 7:-1:1
    T_iN{i} = T_i{i} * T_iN{i+1};
end

%% 初始化雅可比矩阵（现在是6×6）
J_geo = zeros(6, 6);

% 修正的参数映射：6个参数对应的连杆和参数类型
param_mapping = {
    4, 'a';  % delta_phi(1) -> 连杆4的a参数
    5, 'a';  % delta_phi(2) -> 连杆5的a参数
    7, 'a';  % delta_phi(3) -> 连杆7的a参数
    3, 'd';  % delta_phi(4) -> 连杆3的d参数
    5, 'd';  % delta_phi(5) -> 连杆5的d参数
    7, 'd'   % delta_phi(6) -> 连杆7的d参数（末端法兰偏移）
};

%% 对每个几何参数计算偏导数（现在是6个参数）
for k = 1:6
    link_idx   = param_mapping{k, 1};
    param_type = param_mapping{k, 2};

    dT_i  = mdh_parameter_derivative(param_type, theta(link_idx), alpha(link_idx));
    dT_0N = T_0i{link_idx} * dT_i * T_iN{link_idx+1};

    % position part in base frame
    J_geo(1:3, k) = dT_0N(1:3, 4);

    % orientation part in base frame
    dR = dT_0N(1:3, 1:3);
    S  = dR * R_0N';
    S  = 0.5 * (S - S');

    if norm(S, 'fro') < 1e-12
        J_geo(4:6, k) = zeros(3,1);
    else
        J_geo(4:6, k) = [S(3,2); S(1,3); S(2,1)];
    end
end


end

function T = mdh_transform(a, d, alpha, theta)
% MDH变换矩阵计算
% T = Trans_x(a) * Rot_x(alpha) * Trans_z(d) * Rot_z(theta)
    
    ca = cos(alpha);
    sa = sin(alpha);
    ct = cos(theta);
    st = sin(theta);
    
    T = [ct, -st, 0, a;
         st*ca, ct*ca, -sa, -d*sa;
         st*sa, ct*sa, ca, d*ca;
         0, 0, 0, 1];
end

function dT = mdh_parameter_derivative(param_type, theta, alpha)
% MDH变换矩阵对几何参数的偏导数
    
    ct = cos(theta);
    st = sin(theta);
    ca = cos(alpha);
    sa = sin(alpha);
    
    switch param_type
        case 'a'  % 对a的偏导
            dT = [0, 0, 0, 1;
                  0, 0, 0, 0;
                  0, 0, 0, 0;
                  0, 0, 0, 0];
                  
        case 'd'  % 对d的偏导
            dT = [0, 0, 0, 0;
                  0, 0, 0, -sa;
                  0, 0, 0, ca;
                  0, 0, 0, 0];
                  
        case 'alpha'  % 对alpha的偏导
            dT = [0, 0, 0, 0;
                  -st*sa, -ct*sa, -ca, -d*ca;
                  st*ca, ct*ca, -sa, -d*sa;
                  0, 0, 0, 0];
                  
        case 'theta'  % 对theta的偏导
            dT = [-st, -ct, 0, 0;
                  ct*ca, -st*ca, 0, 0;
                  ct*sa, -st*sa, 0, 0;
                  0, 0, 0, 0];
    end
end