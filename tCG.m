%'信赖域半径为正无穷时，信赖域子问题就等同于求解牛顿方程'
%'输入信息：迭代点$x$，梯度grad，海瑟矩阵hess，信赖域半径 $\Delta$，算法参数结构体opts'
%'输出信息：信赖域子问题的解$\eta$（即迭代点处的共轭（下降）方向）'
%'Heta为海瑟矩阵在点$x$作用在方向$\eta$上的结果'
%'即 $\nabla^2f(x^k)\eta^k$，迭代信息结构体out，退出原因stop_tCG'
function [eta, Heta, out, stop_tCG] ...
    = tCG(x, grad, hess, Delta, opts)
if ~isfield(opts,'kappa');     opts.kappa = 0.1;   end%'牛顿方程求解精度的参数'
if ~isfield(opts,'theta');     opts.theta = 1;   end%'牛顿方程求精精度的参数'
if ~isfield(opts,'maxit');     opts.maxit = length(x);   end%'最大迭代次数，对于共轭梯度法而言默认等于变量的维度'
if ~isfield(opts,'minit');     opts.minit = 5;   end%'最小迭代次数'
%'参数复制'
theta = opts.theta;
kappa = opts.kappa;
%'初始化，eta（共轭方向）为优化变量，初始为全0向量$\mathbf{0}$， $r_0$ '
%'初始化为目标函数的梯度 $\nablaf(x^k)$'
eta = zeros(size(x));
Heta = eta;
r = grad;%'目标函数的梯度'
e_Pe = 0;
r_r = r'*r;
norm_r = sqrt(r_r);
norm_r0 = norm_r;
%'共轭梯度法初始时刻的共轭方向 $p^0=-g$ 并在代码中以mdelta表示 $-p^k$ (minus delta)'
mdelta = r;
d_Pd = r_r;
e_Pd = 0;
%'共轭梯度法优化的目标函数'
model_fun = @(eta, Heta) eta'*grad + .5*(eta'*Heta);
model_value = 0;
%'当迭代以最大迭代步数停止时记为stop_tCG = 5'
stop_tCG = 5;
%'迭代主循环'
for j = 1 : opts.maxit
    Hmdelta = hess(mdelta);  %'$g$ 表示梯度，$H$ 表示海瑟矩阵。此处计算$-Hp^k$'  
    d_Hd = mdelta'*Hmdelta;  %'计算曲率$(p^k)^\top Hp^k$'
    alpha = r_r/d_Hd;  %'共轭梯度法的步长$\displaystyle\alpha_{k}=\frac{\|r^k\|_2^2}{(p^k)^\topHp^k}$'
    %'新的共轭方向$\eta^{k+1}=\eta^{k}+\alpha_{k} p^k$'
    %'计算'$(\eta^{k+1})^\top \eta^{k+1}=\alpha_{k}^2(p^k)^\top p^k+2\alpha_{k}p^k\eta^k+(\eta^k)^\top \eta^k$
    e_Pe_new = e_Pe + 2.0*alpha*e_Pd + alpha*alpha*d_Pd;
   %'曲率≤0$(p^{k})^\top Hp^{k}\le0$ 说明海瑟矩阵不正定（再求解下去得到的方向不一定是下降方向），则停止共轭梯度算法'
   %'$\|\eta^{k+1}\|_2^2\ge\Delta^2$ 时，迭代点超出可行域边界，停止算法'
   %'计算$\tau$ 使得 $\|\eta^k+\tau p^k\|_2^2=\Delta^2$'
   %'以$\eta=\eta^k+\tau p^k$ 为最终的迭代结果。更新$H\eta^{k+1}=H\eta^k -\tauH\eta^k$'
   if d_Hd <= 0 || e_Pe_new >= Delta^2
        tau = (-e_Pd + sqrt(e_Pd*e_Pd + d_Pd*(Delta^2-e_Pe))) / d_Pd;
        eta  = eta - tau*mdelta;
        Heta = Heta -tau*Hmdelta;
   %'以负曲率退出记为1，以超出边界退出记为2.退出迭代'
        if d_Hd <= 0
            stop_tCG = 1;
        else
            stop_tCG = 2;
        end
        break;
   end
  %'更新$\eta^{k+1}$'   
    e_Pe = e_Pe_new;
    new_eta  = eta-alpha*mdelta;
    new_Heta = Heta -alpha*Hmdelta;
%'计算$\eta^{k+1}$处的函数值'
%'更新后的目标函数值没有下降，退出迭代，拒绝该步更新。stop_tCG==6表示目标函数值非降而停止'
    new_model_value = model_fun(new_eta, new_Heta);
    if new_model_value >= model_value
        stop_tCG = 6;
        break;
    end
% '如果没有达到上述两种情况，接受更新的 $\eta^{k+1}$，利用新的$\eta^{k+1}$ 更新变量' 
    eta = new_eta;
    Heta = new_Heta;
    model_value = new_model_value;
% '更新残差 $r^{k+1}=r^{k}+\alpha_{k}H p^k$，及其范数，特别的记录上一步的$\|r^k\|_2$为r_rold ' 
    r = r -alpha*Hmdelta;
    r_rold = r_r;
    r_r = r'*r;
    norm_r = sqrt(r_r);
%标准的共轭梯度法的收敛条件：当达到最小迭代次数，且$\|r\|_2\le\|r_0\|_2\min(\kappa,\|r_0\|_2^\theta)$ 时，认为算法收敛，停止迭代
%如果 $\kappa < \|r_0\|^\theta$，则说明条件 $\|r^k\|_2\le\kappa\|r^0\|_2$
%为更严格的条件，说明此时外层牛顿法或者信赖域方法处于线性收敛阶段
%反之，条件$\|r^k\|_2\le\|r_0\|^{1+\theta}$ 更严格，说明此时 $\|r^0\|$
%已经较小，此时对应外层牛顿法或信赖域方法处于超线性收敛阶段
if j >= opts.minit && norm_r <= norm_r0*min(norm_r0^theta, kappa)
        if kappa < norm_r0^theta
            stop_tCG = 3;
        else
            stop_tCG = 4;
        end
        break;
    end
    % 计算新的搜索方向$\beta_k=\frac{\|r^{k+1}\|^2}{\|r^k\|^2}$,
    % $p^{k+1}=-r^{k+1}+\beta_k p^k$?     
    beta = r_r/r_rold;
    mdelta = r + beta*mdelta;
    % 更新 $\eta^{k+1}p^{k+1}=(\eta^k+\alpha_k p^k)^\top(-r^{k+1}+\beta_{k+1}p^k)
    % =\beta_{k+1}(p^k)^\top(\eta^k+\alpha_k p^k)$，这是由于% $\eta^0=\mathbf{0}$ 且$\eta^k$ 为$p^0,\dots,p^k$ 的线性组合，
    %则由 $(r^{k+1})^\top p^j=0,\ j=0,1,\dots,k$ 可知 $(r^{k+1})^\top (\eta^k+\alpha_k p^k)=0$
    % 以及 $(p^{k+1})^\top p^{k+1}=(-r^{k+1}+\beta_k p^k)^\top(-r^{k+1}+\beta_k p^k)$，
    %注意到 $(r^{k+1})^\top p^k=0$，有 $(p^{k+1})^\top p^{k+1}=(r^{k+1})^\top r^{k+1}+\beta_k^2(p^k)^\top p^k$  
    e_Pd = beta*(e_Pd + alpha*d_Pd);
    d_Pd = r_r + beta*beta*d_Pd;
end
% 退出循环，记录退出信息?
out.iter = j;
out.stop_tCG = stop_tCG;
end