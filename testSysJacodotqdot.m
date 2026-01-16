clc; clear;

%% ===================== 0) 测试状态 =====================
q_arm  = [-0.000697544 -0.78446439 0.004060929  -2.356732332 0.000256461 1.570657504 0.785891378].'; % rad
l_para = (0.5 + [0 0 0 0 0 0]).';  % m

%% ===================== 1) J 验证参数 =====================
eps_q = 1e-6;   % rad
eps_l = 1e-5;   % m
dt_vel = 1e-5;  % s（速度验证用）

checkColumns = true;
checkRandomV = true;
NrandV = 5;
qdotScaleV = 0.5;   % rad/s
ldotScaleV = 0.02;  % m/s
seed = 2;
verbose = true;

%% ===================== 2) dotJ*qdot 验证参数 =====================
% 给一组固定测试速度（你要求“给出测试值”）
qdot_arm_test = [ 0.20; -0.15; 0.05; 0.10; -0.12; 0.18; -0.08 ];  % rad/s
ldot_test     = [ 0.010; -0.008; 0.006; -0.004; 0.012; -0.009 ];  % m/s

checkJdotQdot_single = true;

% 随机多次验证（可选）
checkJdotQdot_random = true;
NrandA = 10;
qdotScaleA = 0.5;   % rad/s
ldotScaleA = 0.02;  % m/s
dt_acc = 2e-5;      % s（加速度验证用；二阶差分更建议用更大 dt）

%% ---------- 1) 计算解析 J ----------
[J0, out0] = system_Jdotqdot(q_arm, l_para);   % 你已经验证过 J 正确
T0 = global_fk_from_out(out0);

if verbose
    fprintf('\n===== Hybrid Jacobian & Jdot*qdot Verification =====\n');
    fprintf('State: ||q_arm||=%.3g, ||l_para||=%.3g\n', norm(q_arm), norm(l_para));
    if isfield(out0,'fkInfo')
        fprintf('Stewart FK: iter=%d, residual=%.3e\n', out0.fkInfo.iter, out0.fkInfo.residual_norm);
    end
    fprintf('J size: %dx%d\n', size(J0,1), size(J0,2));
end

%% =====================================================================
% 3) 列验证：逐列对比 J(:,i) 与有限差分
% =====================================================================
if checkColumns
    colAbsErr = zeros(13,1);
    colRelErr = zeros(13,1);

    for i = 1:13
        q_plus  = q_arm;  l_plus  = l_para;
        q_minus = q_arm;  l_minus = l_para;

        if i <= 7
            q_plus(i)  = q_plus(i)  + eps_q;
            q_minus(i) = q_minus(i) - eps_q;
            eps = eps_q;
        else
            idx = i-7;
            l_plus(idx)  = l_plus(idx)  + eps_l;
            l_minus(idx) = l_minus(idx) - eps_l;
            eps = eps_l;
        end

        [~, outP] = system_Jdotqdot(q_plus,  l_plus);
        [~, outM] = system_Jdotqdot(q_minus, l_minus);

        TP = global_fk_from_out(outP);
        TM = global_fk_from_out(outM);

        V_num = twistGeomFromTwoPoses(TP, TM, eps);   % 数值列 (6x1)
        e = V_num - J0(:,i);

        colAbsErr(i) = norm(e);
        colRelErr(i) = norm(e) / max(1e-12, norm(J0(:,i)));
    end

    if verbose
        fprintf('\n--- [J] Column check (finite difference) ---\n');
        fprintf('eps_q=%.1e rad, eps_l=%.1e m\n', eps_q, eps_l);
        fprintf('max abs err = %.3e\n', max(colAbsErr));
        fprintf('max rel err = %.3e\n', max(colRelErr));
        [~, idxMax] = max(colAbsErr);
        fprintf('worst column = %d  (abs %.3e, rel %.3e)\n', idxMax, colAbsErr(idxMax), colRelErr(idxMax));
        if idxMax <= 7
            fprintf(' -> joint q_%d column\n', idxMax);
        else
            fprintf(' -> cylinder l_%d column\n', idxMax-7);
        end
    end
end

%% =====================================================================
% 4) 随机速度验证：比较 J*qdot 与数值 twist
% =====================================================================
if checkRandomV
    rng(seed);

    randAbsErr = zeros(NrandV,1);
    randRelErr = zeros(NrandV,1);

    for k = 1:NrandV
        qdot_arm = qdotScaleV * (2*rand(7,1)-1);
        ldot     = ldotScaleV * (2*rand(6,1)-1);
        qdot_all = [qdot_arm; ldot];

        V_ana = J0 * qdot_all;

        qP = q_arm + qdot_arm*dt_vel;
        lP = l_para + ldot*dt_vel;
        qM = q_arm - qdot_arm*dt_vel;
        lM = l_para - ldot*dt_vel;

        [~, outP] = system_Jdotqdot(qP, lP);
        [~, outM] = system_Jdotqdot(qM, lM);

        TP = global_fk_from_out(outP);
        TM = global_fk_from_out(outM);

        V_num = twistGeomFromTwoPoses(TP, TM, dt_vel);

        e = V_num - V_ana;
        randAbsErr(k) = norm(e);
        randRelErr(k) = norm(e) / max(1e-12, norm(V_num));

        if verbose
            fprintf('J Random vel test %d/%d: abs %.3e, rel %.3e\n', ...
                k, NrandV, randAbsErr(k), randRelErr(k));
        end
    end

    if verbose
        fprintf('\n--- [J] Random velocity check ---\n');
        fprintf('dt_vel=%.1e s, qdotScale=%.3g rad/s, ldotScale=%.3g m/s\n', dt_vel, qdotScaleV, ldotScaleV);
        fprintf('max abs err = %.3e\n', max(randAbsErr));
        fprintf('max rel err = %.3e\n', max(randRelErr));
    end
end

%% =====================================================================
% 5) 验证 dot(J)*dot(q)：用“零加速度输入”下的数值二阶差分加速度对比
%    理论：Xddot = J*qddot + dotJ*qdot
%    当 qddot=0 时：Xddot = dotJ*qdot
% =====================================================================

if checkJdotQdot_single
    % 解析值（system_Jdotqdot 输出的 bias 项）
    [~, outA, Jdot_qdot_ana] = system_Jdotqdot(q_arm, l_para, qdot_arm_test, ldot_test);

    % 数值加速度：用 t-2dt, t, t+2dt 的 pose 先构造 Xdot(t±dt)，再中心差分得到 Xddot(t)
    [TP2, T0_forA, TM2] = poses_at_t_pm2dt(q_arm, l_para, qdot_arm_test, ldot_test, dt_acc);

    Xddot_num = accelGeomFromThreePoses(TP2, T0_forA, TM2, dt_acc);

    e = Xddot_num - Jdot_qdot_ana;

    if verbose
        fprintf('\n--- [dotJ*qdot] Single test (qddot=0, lddot=0) ---\n');
        fprintf('dt_acc=%.1e s\n', dt_acc);
        fprintf('||Xddot_num - Jdot_qdot_ana|| = %.3e\n', norm(e));
        fprintf('rel err = %.3e\n', norm(e)/max(1e-12, norm(Xddot_num)));
        fprintf('num  = [%.3e %.3e %.3e %.3e %.3e %.3e]^T\n', Xddot_num);
        fprintf('ana  = [%.3e %.3e %.3e %.3e %.3e %.3e]^T\n', Jdot_qdot_ana);
    end
end

if checkJdotQdot_random
    rng(seed);

    absErrA = zeros(NrandA,1);
    relErrA = zeros(NrandA,1);

    for k = 1:NrandA
        qdot_arm = qdotScaleA * (2*rand(7,1)-1);
        ldot     = ldotScaleA * (2*rand(6,1)-1);

        [~, ~, Jdot_qdot_ana] = system_Jdotqdot(q_arm, l_para, qdot_arm, ldot);

        [TP2, T0_forA, TM2] = poses_at_t_pm2dt(q_arm, l_para, qdot_arm, ldot, dt_acc);
        Xddot_num = accelGeomFromThreePoses(TP2, T0_forA, TM2, dt_acc);

        e = Xddot_num - Jdot_qdot_ana
        absErrA(k) = norm(e);
        relErrA(k) = absErrA(k) / max(1e-12, norm(Xddot_num));

        if verbose
            fprintf('dotJ*qdot Random accel test %d/%d: abs %.3e, rel %.3e\n', ...
                k, NrandA, absErrA(k), relErrA(k));
        end
    end

    if verbose
        fprintf('\n--- [dotJ*qdot] Random bias-acceleration check ---\n');
        fprintf('dt_acc=%.1e s, qdotScaleA=%.3g rad/s, ldotScaleA=%.3g m/s\n', dt_acc, qdotScaleA, ldotScaleA);
        fprintf('max abs err = %.3e\n', max(absErrA));
        fprintf('max rel err = %.3e\n', max(relErrA));
    end
end

fprintf('\n===== Done. =====\n');

%% ========================================================================
%                               helper functions
% ========================================================================

function [TP2, T0, TM2] = poses_at_t_pm2dt(q_arm, l_para, qdot_arm, ldot, dt)
% 生成 t-2dt, t, t+2dt 的末端位姿（在 {G}）
% 线性更新 q(t)=q0+qdot*t, l(t)=l0+ldot*t, 假设 qddot=0, lddot=0

q_arm  = q_arm(:);  l_para = l_para(:);
qdot_arm = qdot_arm(:); ldot = ldot(:);

% 检查长度正值
lP2 = l_para + 2*ldot*dt;
lM2 = l_para - 2*ldot*dt;
if any(lP2 <= 0) || any(lM2 <= 0)
    error('Actuator length became non-positive under +/-2dt. Reduce dt_acc or ldot.');
end

% t
[~, out0] = system_Jdotqdot(q_arm, l_para);
T0 = global_fk_from_out(out0);

% t+2dt
qP2 = q_arm + 2*qdot_arm*dt;
[~, outP2] = system_Jdotqdot(qP2, lP2);
TP2 = global_fk_from_out(outP2);

% t-2dt
qM2 = q_arm - 2*qdot_arm*dt;
[~, outM2] = system_Jdotqdot(qM2, lM2);
TM2 = global_fk_from_out(outM2);
end

function Xddot = accelGeomFromThreePoses(TP2, T0, TM2, dt)
% 用 3 个 pose：t+2dt, t, t-2dt 计算 “几何定义”下的加速度
% 先估计 Xdot(t+dt)、Xdot(t-dt)，再中心差分：
%   Xddot(t) ≈ (Xdot(t+dt)-Xdot(t-dt)) / (2dt)

Xdot_plus  = twistGeomFromTwoPoses(TP2, T0,  dt);  % 约等于 Xdot(t+dt)
Xdot_minus = twistGeomFromTwoPoses(T0,  TM2, dt);  % 约等于 Xdot(t-dt)

Xddot = (Xdot_plus - Xdot_minus) / (2*dt);
end

function TGF = global_fk_from_out(out)
% 从 out 构造全局末端位姿 T_GF
R = out.R_GP;
p = out.p_GP;
T_BF = out.T_BF;

p_B_in_P = out.delta_dB;  % 你的 out.delta_dB 是 3x1 向量（[0;0;0.12]）

p_GB = p + R * p_B_in_P;
T_GB = [R, p_GB;
        0 0 0 1];

TGF = T_GB * T_BF;
end

function V = twistGeomFromTwoPoses(TP, TM, eps)
% 与几何 Jacobian 一致的 twist: [p_dot; omega]，都在 {G}
Rp = TP(1:3,1:3); pp = TP(1:3,4);
Rm = TM(1:3,1:3); pm = TM(1:3,4);

p_dot = (pp - pm) / (2*eps);

Rdelta = Rp * Rm.';                 % R(t+eps)R(t-eps)^T
omega  = so3Log(Rdelta) / (2*eps);  % 空间角速度（在 {G}）

V = [p_dot; omega];
end

function w = so3Log(R)
cos_theta = (trace(R)-1)/2;
cos_theta = max(-1, min(1, cos_theta));
theta = acos(cos_theta);

if theta < 1e-12
    w = 0.5 * vee3(R - R.');
else
    w = (theta/(2*sin(theta))) * vee3(R - R.');
end
end

function v = vee3(S)
v = [S(3,2); S(1,3); S(2,1)];
end
