function [T0e, R, f, theta] = franka_forward_kinematics(q, dh_actual)
% franka_fk_mdh  计算 Franka Research 3 机械臂的末端执行器相对于基座的齐次变换矩阵
% 输入:
%   q: 7x1 向量, joint angles [theta1; theta2; ...; theta7] (单位 rad)
%   dh_actual: 7x4 实际MDH参数矩阵 [a, d, alpha, theta_offset]
% 输出:
%   T0e: 4x4 齐次变换矩阵, base_frame -> end-effector
%   R: 3x3 末端旋转矩阵
%   f: 3x1 位置向量
%   theta: 3x1 末端欧拉角 [roll, pitch, yaw]

    assert(numel(q)==7, 'q must be 7x1 vector');
    assert(all(size(dh_actual) == [7, 4]), 'dh_actual must be 7x4 matrix');

    % 从dh_actual中提取MDH参数
    a = dh_actual(:, 1);
    d = dh_actual(:, 2);
    alpha = dh_actual(:, 3);
    theta_offset = dh_actual(:, 4);
    
    % 实际关节角度 = 名义关节角度 + 偏移量
    theta_joints = q(:) + theta_offset;

    % 初始化变换矩阵
    T0e = eye(4);

    % 计算每个连杆的MDH变换矩阵（包含末端法兰，通过d7参数）
    for i = 1:7
        ca = cos(alpha(i));
        sa = sin(alpha(i));
        ct = cos(theta_joints(i));
        st = sin(theta_joints(i));

        % MDH变换矩阵: T = Trans_x(a) * Rot_x(alpha) * Trans_z(d) * Rot_z(theta)
        Ti = [ ct, -st, 0, a(i);
               st*ca, ct*ca, -sa, -d(i)*sa;
               st*sa, ct*sa, ca, d(i)*ca;
               0, 0, 0, 1 ];
        T0e = T0e * Ti;
    end

    % 提取旋转矩阵和位置向量
    R = T0e(1:3, 1:3);
    f = T0e(1:3, 4);

    % 将旋转矩阵转换为欧拉角
    theta = rotation_matrix_to_euler_zyx(R);
end