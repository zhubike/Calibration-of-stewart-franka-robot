function [J, out, Jdot_qdot] = system_Jdotqdot(q_arm, l_para, qdot_arm, ldot)

%  Stewart(6UPS) + Franka(7DoF) 串并混联系统 Jacobian + dot(J)*dot(q)
%
%  Twist： Xdot = [v; w]  (线速度在前，角速度在后)
%
%  输入:
%    q_arm  : 7x1  Franka 关节角(rad)
%    l_para : 6x1  Stewart 六个电缸长度(m) 绝对长度
%    qdot_arm (可选): 7x1  Franka 关节速度(rad/s)
%    ldot     (可选): 6x1  电缸速度(m/s)
%
%  输出:
%    J         : 6x13  使得 Xdot_F_in_G = J * [qdot_arm; ldot]
%    out       : 中间量
%    Jdot_qdot : 6x1   dot(J)*dot(q)，仅当提供 qdot_arm, ldot 时计算；否则返回 []

%% ---------------- 是否计算 dotJ*qdot ----------------
doJdot = (nargin >= 4) && ~isempty(qdot_arm) && ~isempty(ldot);
if doJdot
    qdot_arm = qdot_arm(:);
    ldot     = ldot(:);
    assert(numel(qdot_arm)==7, 'qdot_arm must be 7x1');
    assert(numel(ldot)==6,     'ldot must be 6x1');
else
    Jdot_qdot = [];
end

%% ========== Stewart 平台几何参数（单位 m） ==========
% 要根据实际的坐标系定义，修改上下平台铰链点的位置,理论上只需要修改Binit和Ainit即可
% 以机械臂基座的x、y、z轴作为标准
% 6个铰链点3等分，即A1（B1）、A3（B3）、A5（B5）之间的夹角为120°；
% A2（B2）、A4（B4）、A6（B6）之间的夹角为120°
% 相邻铰链点之间的夹角dt为30°（A1与A2之间为30°，B1与B2之间夹角为30°）
% A1与B1之间、A2与B2、...之间的角度差也为30°

% 平台半径
R_upper = 0.1868; 
R_lower = 0.2256; 
dt = 30;  % 相邻铰链点之间的夹角

Binit=30;  % B1铰链点与x轴的夹角为30°，刚好B4铰链点的x轴坐标为0
height = 0.05;  % A铰链点距离上平台0.05m，B铰链点距离下平台0.05m
thetaB = [Binit, Binit+240+dt, Binit+240, ...
          Binit+120+dt, Binit+120, Binit+dt] * pi/180;
B_G = [-R_lower*sin(thetaB(:)), R_lower*cos(thetaB(:)), height*ones(6,1)]';
Ainit=0; % A1铰链点与x轴的夹角为30°，A1铰链点的x轴坐标为0
thetaA = [Ainit, Ainit-dt, Ainit-120, Ainit-120-dt, ...
              Ainit+120, Ainit+120-dt] * pi/180;
A_P = [-R_upper*sin(thetaA(:)), R_upper*cos(thetaA(:)), -height*ones(6,1)]';

%% ========== 机械臂基座B在Stewart上平台中心点P的安装位置 {P}->{B} 固定安装偏置 ==========
delta_dB = [0;0;0.12];

%% 以下参数无需修改
%% ========== Franka 机械臂 MDH 参数 ==========
% 每行: [a, d, alpha, theta_offset]
% 采用 MDH 形式: Tx(a)*Rx(alpha)*Tz(d)*Rz(theta)
dh = [ ...
    0.0000, 0.3330, 0.0000,    0.0000;   % joint1
    0.0000, 0.0000, -pi/2,     0.0000;   % joint2
    0.0000, 0.3160,  pi/2,     0.0000;   % joint3
    0.0825, 0.0000,  pi/2,     0.0000;   % joint4
   -0.0825, 0.3840, -pi/2,     0.0000;   % joint5
    0.0000, 0.0000,  pi/2,     0.0000;   % joint6
    0.0880, 0.1070,  pi/2,     0.0000];  % joint7 (0.1070 含工具安装长度)

%% ========== stewart 正运动学 默认输入 ==========
pose_guess.p = [0;0;0.6];  % 计算stewart正解的迭代初值（P的默认位置）
pose_guess.R = eye(3); % 计算stewart正解的迭代初值（P的默认姿态）
maxIter    = 50;  % 计算stewart正解的最大迭代次数
tol        = 1e-10;  % 计算stewart正解的残差阈值

q_arm  = q_arm(:);
l_para = l_para(:);

assert(numel(q_arm)==7,  'q_arm must be 7x1');
assert(numel(l_para)==6, 'l_para must be 6x1');


%% ========== 1) Stewart 正运动学计算 由于stewart没有FK解析解，因此这里用这个函数计算 ==========
[p_GP, R_GP, fkInfo] = stewart_fk_LM(A_P, B_G, l_para, pose_guess.p, pose_guess.R, maxIter, tol);

%% ========== 2) Stewart 速度雅可比 J_para ==========
[Gq, u_all, rA_G_all, legVec_all] = stewart_Gq(A_P, B_G, p_GP, R_GP);

if rcond(Gq) > 1e-12
    J_para = Gq \ eye(6);
else
    warning('Gq is near singular (rcond=%g). Using pinv(Gq).', rcond(Gq));
    J_para = pinv(Gq);
end

%% ========== 3) Franka Jacobian (in {B}) ==========
[J_arm_B, T_BF, f_B] = franka_jacobian_mdh(q_arm, dh);

%% ========== 4) 组合总 Jacobian ==========
R_B = R_GP;

r_PF = R_GP * delta_dB + R_B * f_B;

G_PF = [ eye(3), -skew3(r_PF);
         zeros(3), eye(3) ];

Rbar_B = blkdiag(R_B, R_B);

J = [Rbar_B * J_arm_B,  G_PF * J_para];

%% ========== 5) 计算 dot(J)*dot(q) ==========
if doJdot
    % 1) 平台速度 Xdot_P
    Xdot_P = J_para * ldot;
    v_P = Xdot_P(1:3);
    w_P = Xdot_P(4:6);

    % 2) 平台 bias 加速度（ddot_l=0）: Xddot_P = -J_para*h_para
    h_para = zeros(6,1);
    for i = 1:6
        ui  = u_all(:,i);
        rAi = rA_G_all(:,i);
        li  = l_para(i);
        ldi = ldot(i);

        vAi = v_P + cross(w_P, rAi);

        h_para(i) = ui.' * cross(w_P, cross(w_P, rAi)) ...
                  + (vAi.'*vAi - ldi^2) / li;
    end
    Xddot_P = -J_para * h_para;
    dotv_P = Xddot_P(1:3);
    dotw_P = Xddot_P(4:6);

    % 3) 机械臂相对 twist in {B}
    %    注意：这里我们需要 Tlist，所以 franka_jacobian_mdh 要返回第4输出
    [J_arm_B, T_BF, f_B, Tlist] = franka_jacobian_mdh(q_arm, dh);

    Xdot_rel_B = J_arm_B * qdot_arm;     % [v_rel_B; w_rel_B]
    v_rel_B = Xdot_rel_B(1:3);
    w_rel_B = Xdot_rel_B(4:6);

    % 4) 机械臂相对 bias 加速度 in {B}：^B(dotJ*qdot)
    Xddot_rel_B = armJdotQdot_geometric_fromTlist(J_arm_B, Tlist, qdot_arm);
    dotv_rel_B = Xddot_rel_B(1:3);
    dotw_rel_B = Xddot_rel_B(4:6);

    % 5) 转到 {G}
    R_B = R_GP;
    v_rel_G    = R_B * v_rel_B;
    w_rel_G    = R_B * w_rel_B;
    dotv_rel_G = R_B * dotv_rel_B;
    dotw_rel_G = R_B * dotw_rel_B;

    % 6) r_PF
    r_PF = R_GP * delta_dB + R_B * f_B;

    % 7) 末端 bias 加速度（就是 dotJ*qdot）
    dotw_F = dotw_P ...
           + cross(w_P, w_rel_G) ...
           + dotw_rel_G;

    dotv_F = dotv_P ...
           + cross(dotw_P, r_PF) ...
           + cross(w_P, cross(w_P, r_PF)) ...
           + 2*cross(w_P, v_rel_G) ...
           + dotv_rel_G;

    Jdot_qdot = [dotv_F; dotw_F];
else
    Jdot_qdot = [];
end


%% ========== 输出打包 ==========
out = struct();
out.A_P = A_P; out.B_G = B_G;
out.dh  = dh;

out.p_GP = p_GP;
out.R_GP = R_GP;
out.fkInfo = fkInfo;

out.Gq = Gq;
out.J_para = J_para;
out.u_all = u_all;
out.rA_G_all = rA_G_all;
out.legVec_all = legVec_all;

out.T_BF = T_BF;
out.f_B  = f_B;
out.J_arm_B = J_arm_B;
out.delta_dB = delta_dB;

out.R_B = R_B;
out.r_PF = r_PF;
out.G_PF = G_PF;
out.Rbar_B = Rbar_B;

end

%% ========================================================================
%                               子函数
% ========================================================================

function [p, R, info] = stewart_fk_LM(A_P, B_G, l, p0, R0, maxIter, tol)
p = p0(:);
R = R0;

lambda = 1e-4;
resPrev = inf;

for it = 1:maxIter
    [g, Gq, ~] = stewart_residual_and_Gq(A_P, B_G, p, R, l);
    res = norm(g);
    if res < tol
        break;
    end

    H = Gq.'*Gq + lambda*eye(6);
    delta = -(H \ (Gq.'*g));

    accepted = false;
    for trial = 1:10
        p_new = p + delta(1:3);
        R_new = so3_exp(delta(4:6)) * R;

        g_new = stewart_residual_only(A_P, B_G, p_new, R_new, l);
        res_new = norm(g_new);

        if res_new < res
            p = p_new;
            R = R_new;
            lambda = max(lambda/5, 1e-12);
            accepted = true;
            break;
        else
            lambda = lambda*5;
            H = Gq.'*Gq + lambda*eye(6);
            delta = -(H \ (Gq.'*g));
        end
    end

    if ~accepted
        break;
    end
    if abs(resPrev - res)/max(1,res) < 1e-12
        break;
    end
    resPrev = res;
end

info = struct();
info.iter = it;
info.residual_norm = res;
info.lambda_final = lambda;
end

function g = stewart_residual_only(A_P, B_G, p, R, l)
S = R*A_P + p - B_G;
L = sqrt(sum(S.^2,1)).';
g = L - l(:);
end

function [g, Gq, cache] = stewart_residual_and_Gq(A_P, B_G, p, R, l)
S = R*A_P + p - B_G;
L = sqrt(sum(S.^2,1));
u = S ./ L;

g = L.' - l(:);

Gq = zeros(6,6);
rA_G = R*A_P;
for i=1:6
    ui = u(:,i);
    rAi = rA_G(:,i);
    Gq(i,1:3) = ui.';
    Gq(i,4:6) = (ui.' * (-skew3(rAi)));
end

cache = struct();
cache.S = S;
cache.L = L;
cache.u = u;
cache.rA_G = rA_G;
end

function [Gq, u_all, rA_G_all, legVec_all] = stewart_Gq(A_P, B_G, p, R)
S = R*A_P + p - B_G;
L = sqrt(sum(S.^2,1));
u = S ./ L;
rA_G = R*A_P;

Gq = zeros(6,6);
for i=1:6
    ui = u(:,i);
    rAi = rA_G(:,i);
    Gq(i,1:3) = ui.';
    Gq(i,4:6) = (ui.' * (-skew3(rAi)));
end

u_all = u;
rA_G_all = rA_G;
legVec_all = S;
end

function [J, T_BF, f_B, Tlist] = franka_jacobian_mdh(q, dh)
q = q(:);
theta = q + dh(:,4);

T = eye(4);
Tlist = cell(7,1);

for i=1:7
    a = dh(i,1); d = dh(i,2); alpha = dh(i,3); th = theta(i);
    Ti = mdh_T(a,d,alpha,th);
    T = T * Ti;
    Tlist{i} = T;
end

T_BF = T;
f_B = T_BF(1:3,4);

J = zeros(6,7);
p_e = f_B;

for i=1:7
    zi = Tlist{i}(1:3,3);
    pi = Tlist{i}(1:3,4);
    J(1:3,i) = cross(zi, (p_e - pi));
    J(4:6,i) = zi;
end
end

function T = mdh_T(a, d, alpha, theta)
ca = cos(alpha); sa = sin(alpha);
ct = cos(theta); st = sin(theta);

T = [ ct,    -st,     0,   a;
      st*ca, ct*ca,  -sa, -d*sa;
      st*sa, ct*sa,   ca,  d*ca;
      0,      0,      0,   1 ];
end

function S = skew3(v)
S = [   0, -v(3),  v(2);
      v(3),   0,  -v(1);
     -v(2), v(1),   0  ];
end

function R = so3_exp(w)
th = norm(w);
if th < 1e-12
    R = eye(3) + skew3(w);
    return;
end
k = w / th;
K = skew3(k);
R = eye(3) + sin(th)*K + (1-cos(th))*(K*K);
end

%% ======= 解析计算 (dotJ_arm_B)*qdot_arm 的辅助函数（不显式输出 dotJ） =======

function Jdotq = armJdotQdot_geometric_fromTlist(J_arm_B, Tlist, qdot)
% 解析计算 ^B(dotJ*qdot)，与当前几何Jacobian列定义完全一致
% 输入:
%   J_arm_B : 6x7 [v;w] in {B}
%   Tlist   : cell(7,1), T0i in {B}
%   qdot    : 7x1
% 输出:
%   Jdotq   : 6x1 [vdot; wdot] in {B}，假设 qddot=0

qdot = qdot(:);
n = 7;

% 提取 z_i 和 p_i （与你的 Jacobian 列构造完全一致）
z = zeros(3,n);
p = zeros(3,n);
for i = 1:n
    Ti = Tlist{i};
    z(:,i) = Ti(1:3,3);
    p(:,i) = Ti(1:3,4);
end

p_e = p(:,n);
v_e = J_arm_B(1:3,:) * qdot;   % 末端线速度 in {B}

% 计算每个 frame i 的角速度 ω_i 和线速度 v_i（都 in {B}）
omega_i = zeros(3,n);
v_i     = zeros(3,n);

for i = 1:n
    % ω_i = sum_{k=1..i} z_k qdot_k
    omega_i(:,i) = z(:,1:i) * qdot(1:i);

    % v_i = sum_{k=1..i} z_k × (p_i - p_k) qdot_k
    vi = zeros(3,1);
    for k = 1:i
        vi = vi + cross(z(:,k), (p(:,i) - p(:,k))) * qdot(k);
    end
    v_i(:,i) = vi;
end

% 计算 bias：Jdot*qdot（qddot=0）
vdot_bias = zeros(3,1);
wdot_bias = zeros(3,1);

for i = 1:n
    zi = z(:,i);
    pi = p(:,i);

    % zdot_i = ω_i × z_i
    z_dot = cross(omega_i(:,i), zi);

    % wdot 部分
    wdot_bias = wdot_bias + z_dot * qdot(i);

    % vdot 部分
    vdot_bias = vdot_bias + ( ...
        cross(z_dot, (p_e - pi)) + ...
        cross(zi, (v_e - v_i(:,i))) ) * qdot(i);
end

Jdotq = [vdot_bias; wdot_bias];
end

